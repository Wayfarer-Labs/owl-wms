"""
Dimensionality Reduction for Velocity Matching in Diffusion Models
Focused on computational efficiency and training stability
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class VelocityEncoder(nn.Module):
    """
    Encoder to reduce velocity dimensionality while preserving OT-relevant structure
    """
    def __init__(self, 
                 input_dim: int,
                 latent_dim: int,
                 hidden_dims: list = [512, 256, 128],
                 use_spectral_norm: bool = True):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        for h_dim in hidden_dims:
            if use_spectral_norm:
                layers.append(nn.utils.spectral_norm(nn.Linear(prev_dim, h_dim)))
            else:
                layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.LayerNorm(h_dim))
            layers.append(nn.GELU())
            prev_dim = h_dim
            
        # Final projection to latent space
        if use_spectral_norm:
            layers.append(nn.utils.spectral_norm(nn.Linear(prev_dim, latent_dim)))
        else:
            layers.append(nn.Linear(prev_dim, latent_dim))
            
        self.encoder = nn.Sequential(*layers)
        
    def forward(self, v: torch.Tensor) -> torch.Tensor:
        """Encode velocities to latent space"""
        return self.encoder(v)


class VelocityDecoder(nn.Module):
    """
    Decoder to reconstruct full velocities from latent representation
    """
    def __init__(self, 
                 latent_dim: int,
                 output_dim: int,
                 hidden_dims: list = [128, 256, 512],
                 use_spectral_norm: bool = True):
        super().__init__()
        
        layers = []
        prev_dim = latent_dim
        
        for h_dim in hidden_dims:
            if use_spectral_norm:
                layers.append(nn.utils.spectral_norm(nn.Linear(prev_dim, h_dim)))
            else:
                layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.LayerNorm(h_dim))
            layers.append(nn.GELU())
            prev_dim = h_dim
            
        # Final projection back to velocity space
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.decoder = nn.Sequential(*layers)
        
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent representation to velocities"""
        return self.decoder(z)


class PCAVelocityReducer:
    """
    Efficient PCA-based velocity reduction with incremental updates
    Good for online training scenarios
    """
    def __init__(self, 
                 n_components: int = 128,
                 whiten: bool = True,
                 incremental: bool = True):
        self.n_components = n_components
        self.whiten = whiten
        self.incremental = incremental
        
        self.mean = None
        self.components = None
        self.explained_variance = None
        self.n_samples_seen = 0
        
    def partial_fit(self, velocities: torch.Tensor):
        """
        Incrementally update PCA components
        Efficient for streaming data during training
        """
        batch_size = velocities.shape[0]
        velocities_flat = velocities.reshape(batch_size, -1)
        
        if self.mean is None:
            self.mean = torch.zeros(velocities_flat.shape[1], device=velocities.device)
            self.components = None
            
        # Update mean incrementally
        old_n = self.n_samples_seen
        new_n = old_n + batch_size
        
        if old_n == 0:
            self.mean = velocities_flat.mean(dim=0)
        else:
            self.mean = (old_n * self.mean + velocities_flat.sum(dim=0)) / new_n
            
        self.n_samples_seen = new_n
        
        # Update components using randomized SVD for efficiency
        if self.n_samples_seen >= self.n_components * 2:
            centered = velocities_flat - self.mean
            
            # Use randomized SVD for large matrices
            if centered.shape[1] > 1000:
                # Random projection for efficiency
                n_oversamples = 10
                random_matrix = torch.randn(
                    centered.shape[1], 
                    self.n_components + n_oversamples,
                    device=centered.device
                )
                Q, _ = torch.linalg.qr(centered @ random_matrix)
                B = Q.T @ centered
                
                # SVD on smaller matrix
                U_small, s, Vt = torch.linalg.svd(B, full_matrices=False)
                U = Q @ U_small
            else:
                U, s, Vt = torch.linalg.svd(centered, full_matrices=False)
                
            # Update components
            self.components = Vt[:self.n_components]
            self.explained_variance = (s[:self.n_components] ** 2) / (new_n - 1)
            
    def transform(self, velocities: torch.Tensor) -> torch.Tensor:
        """Project velocities to lower dimensional space"""
        original_shape = velocities.shape
        velocities_flat = velocities.reshape(velocities.shape[0], -1)
        
        if self.components is None:
            raise ValueError("PCA must be fitted before transform")
            
        centered = velocities_flat - self.mean
        projected = centered @ self.components.T
        
        if self.whiten:
            projected = projected / torch.sqrt(self.explained_variance + 1e-8)
            
        return projected
        
    def inverse_transform(self, z: torch.Tensor, original_shape: tuple) -> torch.Tensor:
        """Reconstruct velocities from latent representation"""
        if self.whiten:
            z = z * torch.sqrt(self.explained_variance + 1e-8)
            
        velocities_flat = z @ self.components + self.mean
        return velocities_flat.reshape(original_shape)


class EfficientOTCFM:
    """
    OT-CFM with dimensionality reduction for efficiency and stability
    """
    def __init__(self,
                 teacher_model,
                 student_model,
                 velocity_dim_reducer,
                 latent_dim: int = 128,
                 ot_method: str = 'minibatch',
                 reg: float = 0.05,
                 stability_weight: float = 0.1,
                 device: str = 'cuda'):
        
        self.teacher = teacher_model
        self.student = student_model
        self.reducer = velocity_dim_reducer
        self.latent_dim = latent_dim
        self.ot_method = ot_method
        self.reg = reg
        self.stability_weight = stability_weight
        self.device = device
        
        # Stability tracking
        self.velocity_ema = None
        self.velocity_std_ema = None
        self.ema_decay = 0.999
        
    def compute_reduced_ot(self, v0_latent: torch.Tensor, v1_latent: torch.Tensor):
        """
        Compute OT in reduced dimension space
        Much more efficient than full dimensional OT
        """
        n = v0_latent.shape[0]
        
        if self.ot_method == 'minibatch':
            # Simple random coupling
            perm = torch.randperm(n, device=self.device)
            return perm
            
        elif self.ot_method == 'sinkhorn':
            # Sinkhorn in reduced space - much faster!
            C = torch.cdist(v0_latent, v1_latent, p=2).pow(2)
            
            # Use log-domain sinkhorn for stability
            log_a = torch.zeros(n, device=self.device)
            log_b = torch.zeros(n, device=self.device)
            
            # Log-space Sinkhorn iterations
            for _ in range(50):  # Fewer iterations needed in low-dim
                log_K = -C / self.reg
                log_a = -torch.logsumexp(log_K + log_b.unsqueeze(0), dim=1)
                log_b = -torch.logsumexp(log_K.T + log_a.unsqueeze(0), dim=1)
                
            # Sample from transport plan
            log_pi = log_a.unsqueeze(1) + log_K + log_b.unsqueeze(0)
            
            # Gumbel-softmax sampling for differentiability
            gumbel = -torch.log(-torch.log(torch.rand_like(log_pi) + 1e-8) + 1e-8)
            indices = torch.argmax(log_pi + gumbel, dim=1)
            
            return indices
            
    def update_stability_metrics(self, velocities: torch.Tensor):
        """
        Track velocity statistics for training stability
        """
        with torch.no_grad():
            v_mean = velocities.mean()
            v_std = velocities.std()
            
            if self.velocity_ema is None:
                self.velocity_ema = v_mean
                self.velocity_std_ema = v_std
            else:
                self.velocity_ema = self.ema_decay * self.velocity_ema + (1 - self.ema_decay) * v_mean
                self.velocity_std_ema = self.ema_decay * self.velocity_std_ema + (1 - self.ema_decay) * v_std
                
    def stability_regularization(self, velocities: torch.Tensor) -> torch.Tensor:
        """
        Add regularization to prevent velocity explosion/vanishing
        """
        # Spectral normalization of velocities
        v_flat = velocities.reshape(velocities.shape[0], -1)
        _, s, _ = torch.linalg.svd(v_flat, full_matrices=False)
        spectral_penalty = torch.mean((s - 1.0) ** 2)
        
        # Variance regularization
        v_std = velocities.std()
        variance_penalty = (v_std - 1.0) ** 2
        
        return self.stability_weight * (spectral_penalty + variance_penalty)
        
    def velocity_matching_loss(self, 
                             batch_data: torch.Tensor,
                             batch_noise: Optional[torch.Tensor] = None,
                             t: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute velocity matching loss with dimensionality reduction
        """
        batch_size = batch_data.shape[0]
        
        if batch_noise is None:
            batch_noise = torch.randn_like(batch_data)
        if t is None:
            t = torch.rand(batch_size, 1, device=self.device)
            
        # Get teacher velocities
        with torch.no_grad():
            # Teacher forward pass
            x_t_teacher = (1 - t) * batch_noise + t * batch_data
            v_teacher = self.teacher(x_t_teacher, t)
            
            # Update stability metrics
            self.update_stability_metrics(v_teacher)
            
            # Reduce dimensionality
            if isinstance(self.reducer, PCAVelocityReducer):
                # Update PCA with new velocities
                self.reducer.partial_fit(v_teacher)
                v_teacher_reduced = self.reducer.transform(v_teacher)
            else:
                # Neural encoder
                v_teacher_reduced = self.reducer(v_teacher.reshape(batch_size, -1))
                
        # Get student velocities  
        v_student = self.student(x_t_teacher, t)
        
        # Reduce student velocities
        if isinstance(self.reducer, PCAVelocityReducer):
            v_student_reduced = self.reducer.transform(v_student)
        else:
            v_student_reduced = self.reducer(v_student.reshape(batch_size, -1))
            
        # Compute OT in reduced space (much faster!)
        if self.ot_method != 'none':
            indices = self.compute_reduced_ot(v_teacher_reduced, v_student_reduced)
            v_teacher_reduced = v_teacher_reduced[indices]
            v_teacher = v_teacher[indices]
            
        # Matching loss in reduced space
        reduced_loss = F.mse_loss(v_student_reduced, v_teacher_reduced)
        
        # Optional: reconstruction loss to ensure good representation
        if hasattr(self, 'decoder') and self.decoder is not None:
            v_student_recon = self.decoder(v_student_reduced)
            recon_loss = F.mse_loss(v_student_recon, v_student.reshape(batch_size, -1))
            reduced_loss = reduced_loss + 0.1 * recon_loss
            
        # Add stability regularization
        stability_loss = self.stability_regularization(v_student)
        
        return reduced_loss + stability_loss


# Practical training function with all optimizations
def train_efficient_ot_cfm(teacher_model,
                          student_model,
                          train_loader,
                          n_epochs: int = 100,
                          latent_dim: int = 128,
                          lr: float = 1e-3,
                          gradient_clip: float = 1.0,
                          device: str = 'cuda'):
    """
    Train with dimensionality reduction for efficiency and stability
    """
    # Setup PCA reducer (most efficient option)
    velocity_reducer = PCAVelocityReducer(
        n_components=latent_dim,
        whiten=True,
        incremental=True
    )
    
    # Alternative: Neural encoder/decoder
    # velocity_dim = 256 * 256 * 3  # Example for 256x256 RGB velocities
    # velocity_encoder = VelocityEncoder(velocity_dim, latent_dim).to(device)
    # velocity_decoder = VelocityDecoder(latent_dim, velocity_dim).to(device)
    
    # Setup efficient OT-CFM
    ot_cfm = EfficientOTCFM(
        teacher_model=teacher_model,
        student_model=student_model,
        velocity_dim_reducer=velocity_reducer,
        latent_dim=latent_dim,
        ot_method='sinkhorn',  # Now feasible in reduced space!
        stability_weight=0.1,
        device=device
    )
    
    # Optimizer with gradient clipping
    optimizer = torch.optim.AdamW(
        student_model.parameters(), 
        lr=lr,
        weight_decay=0.01,
        betas=(0.9, 0.99)  # More stable than default
    )
    
    # Learning rate scheduler for stability
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2
    )
    
    # Training loop with checkpointing
    for epoch in range(n_epochs):
        epoch_loss = 0
        n_batches = 0
        
        for batch_idx, batch_data in enumerate(train_loader):
            batch_data = batch_data[0].to(device)
            
            # Forward pass with gradient accumulation for stability
            if batch_data.shape[0] > 64:
                # Split large batches
                accumulation_steps = batch_data.shape[0] // 64
                optimizer.zero_grad()
                
                for i in range(accumulation_steps):
                    start_idx = i * 64
                    end_idx = min((i + 1) * 64, batch_data.shape[0])
                    sub_batch = batch_data[start_idx:end_idx]
                    
                    loss = ot_cfm.velocity_matching_loss(sub_batch)
                    loss = loss / accumulation_steps
                    loss.backward()
                    
            else:
                optimizer.zero_grad()
                loss = ot_cfm.velocity_matching_loss(batch_data)
                loss.backward()
                
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(student_model.parameters(), gradient_clip)
            
            # Optimizer step
            optimizer.step()
            scheduler.step()
            
            epoch_loss += loss.item()
            n_batches += 1
            
            # Log stability metrics
            if batch_idx % 100 == 0 and ot_cfm.velocity_ema is not None:
                print(f"Velocity stats - Mean: {ot_cfm.velocity_ema:.4f}, "
                      f"Std: {ot_cfm.velocity_std_ema:.4f}")
                      
        avg_loss = epoch_loss / n_batches
        print(f"Epoch {epoch}: Loss = {avg_loss:.4f}, LR = {scheduler.get_last_lr()[0]:.6f}")
        
    return student_model


# Memory-efficient velocity collection with chunking
class ChunkedVelocityCollector:
    """
    Collect velocities in chunks to avoid OOM
    """
    def __init__(self, chunk_size: int = 1000, device: str = 'cuda'):
        self.chunk_size = chunk_size
        self.device = device
        self.velocity_chunks = []
        
    def collect_velocities(self, model, data_loader, n_timesteps: int = 10):
        """
        Collect velocities across multiple timesteps efficiently
        """
        all_velocities = []
        
        with torch.no_grad():
            for batch_data in data_loader:
                batch_data = batch_data.to(self.device)
                batch_size = batch_data.shape[0]
                
                # Sample multiple timesteps per batch
                for _ in range(n_timesteps):
                    t = torch.rand(batch_size, 1, device=self.device)
                    noise = torch.randn_like(batch_data)
                    x_t = (1 - t) * noise + t * batch_data
                    
                    # Get velocities in chunks to save memory
                    velocities = []
                    for i in range(0, batch_size, self.chunk_size):
                        chunk_x = x_t[i:i+self.chunk_size]
                        chunk_t = t[i:i+self.chunk_size]
                        chunk_v = model(chunk_x, chunk_t)
                        velocities.append(chunk_v.cpu())  # Move to CPU to save GPU memory
                        
                    all_velocities.extend(velocities)
                    
        return torch.cat(all_velocities, dim=0)


if __name__ == "__main__":
    # Example usage with memory-efficient setup
    device = 'cpu'  # Force CPU due to CUDA compatibility issue
    
    # Dummy models (replace with your actual models)
    from torch.utils.data import DataLoader, TensorDataset
    
    class DummyVelocityModel(nn.Module):
        def __init__(self, dim=2):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(dim + 1, 128),
                nn.ReLU(),
                nn.Linear(128, dim)
            )
            
        def forward(self, x, t):
            return self.net(torch.cat([x, t], dim=-1))
    
    # Create models
    teacher = DummyVelocityModel(dim=2).to(device)
    student = DummyVelocityModel(dim=2).to(device)
    
    # Create dummy data
    dummy_data = torch.randn(1000, 2)
    train_loader = DataLoader(
        TensorDataset(dummy_data), 
        batch_size=64, 
        shuffle=True
    )
    
    # Train with efficient OT-CFM
    trained_student = train_efficient_ot_cfm(
        teacher, student, train_loader,
        n_epochs=10,
        latent_dim=32,  # Much smaller than original dimension
        device=device
    )