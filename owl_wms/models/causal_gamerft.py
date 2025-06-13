"""
Causal GameRFT model with KV cache support for efficient autoregressive generation
"""

import torch
from torch import nn
import torch.nn.functional as F
import einops as eo

from ..nn.embeddings import (
    TimestepEmbedding,
    ControlEmbedding,
    LearnedPosEnc
)
from ..nn.attn import UViT, FinalLayer
from ..nn.kv_cache import KVCache

class CausalGameRFTCore(nn.Module):
    """
    Core model with causal attention support and KV caching
    """
    def __init__(self, config):
        super().__init__()

        # Modify config for causal attention
        self.causal = config.causal
        self.tokens_per_frame = config.tokens_per_frame
        self.n_frames = config.n_frames
        
        # Initialize transformer with causal flag
        config_copy = deepcopy(config)
        config_copy.causal = self.causal
        self.transformer = UViT(config_copy)
        
        self.control_embed = ControlEmbedding(config.n_buttons, config.d_model)
        self.t_embed = TimestepEmbedding(config.d_model)

        self.proj_in = nn.Linear(config.channels, config.d_model, bias=False)
        self.proj_out = FinalLayer(config.sample_size, config.d_model, config.channels)

        self.pos_enc = LearnedPosEnc(config.tokens_per_frame * config.n_frames, config.d_model)
        
        # For caching context frames
        self.cached_frames = None
        self.cached_frame_count = 0

    def forward(self, x, t, mouse, btn, kv_cache=None, use_cache=False):
        """
        Forward pass with optional KV caching
        
        Args:
            x: Input frames [b,n,c,h,w]
            t: Timesteps [b,n]
            mouse: Mouse inputs [b,n,2]
            btn: Button inputs [b,n,n_buttons]
            kv_cache: Optional KVCache object for caching
            use_cache: Whether to use cached context
        """
        # Handle caching for autoregressive generation
        if use_cache and self.cached_frames is not None:
            # Concatenate cached frames with new input
            x = torch.cat([self.cached_frames, x], dim=1)
            t = torch.cat([self.cached_timesteps, t], dim=1)
            mouse = torch.cat([self.cached_mouse, mouse], dim=1)
            btn = torch.cat([self.cached_btn, btn], dim=1)
            
            # Update cache with new frames
            self.cached_frames = x
            self.cached_timesteps = t
            self.cached_mouse = mouse
            self.cached_btn = btn
        elif use_cache:
            # Initialize cache
            self.cached_frames = x
            self.cached_timesteps = t
            self.cached_mouse = mouse
            self.cached_btn = btn

        # Standard forward pass
        ctrl_cond = self.control_embed(mouse, btn)
        t_cond = self.t_embed(t)

        cond = ctrl_cond + t_cond # [b,n,d]
        
        b, n, c, h, w = x.shape
        x = eo.rearrange(x, 'b n c h w -> b (n h w) c')

        x = self.proj_in(x)
        x = self.pos_enc(x)
        
        # Pass KV cache to transformer if provided
        x = self.transformer(x, cond, kv_cache=kv_cache)
        
        x = self.proj_out(x, cond) # -> [b,n*hw,c]
        x = eo.rearrange(x, 'b (n h w) c -> b n c h w', n=n, h=h, w=w)

        # If using cache, only return the newly generated frames
        if use_cache and self.cached_frame_count > 0:
            x = x[:, -1:]  # Return only last frame
        
        return x
    
    def reset_cache(self):
        """Reset the cached context frames"""
        self.cached_frames = None
        self.cached_timesteps = None
        self.cached_mouse = None
        self.cached_btn = None
        self.cached_frame_count = 0

class CausalGameRFT(nn.Module):
    """
    Causal GameRFT model for autoregressive video generation
    """
    def __init__(self, config):
        super().__init__()

        self.core = CausalGameRFTCore(config)
        self.cfg_prob = config.cfg_prob
        self.causal = config.causal
    
    def forward(self, x, mouse, btn, return_dict=False, cfg_prob=None, kv_cache=None, use_cache=False):
        """
        Forward pass with diffusion loss computation
        
        For training: Standard diffusion loss
        For generation: Can use KV cache for efficiency
        """
        b, n, c, h, w = x.shape

        # Apply classifier-free guidance dropout
        if cfg_prob is None:
            cfg_prob = self.cfg_prob
        if cfg_prob > 0.0 and self.training:
            mask = torch.rand(b, device=x.device) <= cfg_prob
            null_mouse = torch.zeros_like(mouse)
            null_btn = torch.zeros_like(btn)
            
            # Where mask is True, replace with zeros
            mouse = torch.where(mask.unsqueeze(-1).unsqueeze(-1), null_mouse, mouse)
            btn = torch.where(mask.unsqueeze(-1).unsqueeze(-1), null_btn, btn)
        
        if self.training:
            # Standard diffusion training
            with torch.no_grad():
                ts = torch.rand(b, n, device=x.device, dtype=x.dtype).sigmoid()
                
                ts_exp = eo.repeat(ts, 'b n -> b n 1 1 1')
                z = torch.randn_like(x)

                lerpd = x * (1. - ts_exp) + z * ts_exp
                target = z - x
            
            pred = self.core(lerpd, ts, mouse, btn)
            diff_loss = F.mse_loss(pred, target)

            if not return_dict:
                return diff_loss
            else:
                return {
                    'diffusion_loss': diff_loss,
                    'lerpd': lerpd, 
                    'pred': pred,
                    'ts': ts,
                    'z': z
                }
        else:
            # Generation mode - can use KV cache
            return self.core(x, torch.zeros(b, n, device=x.device), mouse, btn, 
                           kv_cache=kv_cache, use_cache=use_cache)

    def generate_next_frame(self, context, mouse, btn, num_steps=50, cfg_scale=1.3, kv_cache=None):
        """
        Generate the next frame given context frames
        
        Args:
            context: Context frames [b, n_context, c, h, w]
            mouse: Mouse inputs for context + 1 new frame [b, n_context+1, 2]
            btn: Button inputs for context + 1 new frame [b, n_context+1, n_buttons]
            num_steps: Number of denoising steps
            cfg_scale: Classifier-free guidance scale
            kv_cache: Optional KV cache for efficiency
            
        Returns:
            next_frame: Generated next frame [b, 1, c, h, w]
        """
        b = context.shape[0]
        device = context.device
        
        # Initialize next frame with noise
        next_frame = torch.randn(b, 1, *context.shape[2:], device=device)
        
        # Combine context and noisy next frame
        full_sequence = torch.cat([context, next_frame], dim=1)
        
        # Denoising loop
        for step in range(num_steps):
            t = (1.0 - step / num_steps) * torch.ones(b, full_sequence.shape[1], device=device)
            
            # Zero timestep for context frames (they're clean)
            t[:, :-1] = 0.0
            
            with torch.no_grad():
                # Conditional prediction
                pred_cond = self.core(full_sequence, t, mouse, btn, kv_cache=kv_cache)
                
                if cfg_scale > 1.0:
                    # Unconditional prediction for CFG
                    null_mouse = torch.zeros_like(mouse)
                    null_btn = torch.zeros_like(btn)
                    pred_uncond = self.core(full_sequence, t, null_mouse, null_btn, kv_cache=kv_cache)
                    
                    # Apply CFG
                    pred = pred_uncond + cfg_scale * (pred_cond - pred_uncond)
                else:
                    pred = pred_cond
                
                # Update only the last frame
                noise_level = t[:, -1:, None, None, None]
                full_sequence[:, -1:] = full_sequence[:, -1:] - pred[:, -1:] * noise_level / num_steps
        
        return full_sequence[:, -1:]

    def generate_sequence(self, initial_frame, mouse, btn, num_frames, 
                         num_steps=50, cfg_scale=1.3, use_kv_cache=True):
        """
        Generate a full sequence autoregressively
        
        Args:
            initial_frame: Starting frame [b, 1, c, h, w]
            mouse: Mouse inputs for full sequence [b, num_frames, 2]
            btn: Button inputs for full sequence [b, num_frames, n_buttons]
            num_frames: Number of frames to generate
            num_steps: Denoising steps per frame
            cfg_scale: CFG scale
            use_kv_cache: Whether to use KV caching for efficiency
            
        Returns:
            sequence: Generated sequence [b, num_frames, c, h, w]
        """
        generated = [initial_frame]
        
        # Initialize KV cache if requested
        kv_cache = KVCache(self.core.transformer.config) if use_kv_cache else None
        if kv_cache:
            kv_cache.reset(initial_frame.shape[0])
        
        # Reset any internal caching
        self.core.reset_cache()
        
        for frame_idx in range(1, num_frames):
            # Get context and actions up to current frame
            context = torch.cat(generated, dim=1)
            frame_mouse = mouse[:, :frame_idx+1]
            frame_btn = btn[:, :frame_idx+1]
            
            # Generate next frame
            next_frame = self.generate_next_frame(
                context, frame_mouse, frame_btn,
                num_steps=num_steps, cfg_scale=cfg_scale,
                kv_cache=kv_cache
            )
            
            generated.append(next_frame)
        
        return torch.cat(generated, dim=1)


if __name__ == "__main__":
    from ..configs import Config
    from copy import deepcopy

    # Test causal model
    cfg = Config.from_yaml("configs/basic.yml").model
    cfg.causal = True
    model = CausalGameRFT(cfg).cuda().bfloat16()

    # Test training forward pass
    with torch.no_grad():
        x = torch.randn(2, 30, 128, 4, 4, device='cuda', dtype=torch.bfloat16)
        mouse = torch.randn(2, 30, 2, device='cuda', dtype=torch.bfloat16)
        btn = torch.randn(2, 30, 11, device='cuda', dtype=torch.bfloat16)
        
        loss = model(x, mouse, btn)
        print(f"Training loss: {loss.item()}")
        
    # Test generation
    model.eval()
    with torch.no_grad():
        initial = torch.randn(2, 1, 128, 4, 4, device='cuda', dtype=torch.bfloat16)
        mouse_seq = torch.randn(2, 10, 2, device='cuda', dtype=torch.bfloat16)
        btn_seq = torch.randn(2, 10, 11, device='cuda', dtype=torch.bfloat16)
        
        generated = model.generate_sequence(
            initial, mouse_seq, btn_seq, 
            num_frames=10, num_steps=20, cfg_scale=1.3
        )
        print(f"Generated sequence shape: {generated.shape}")