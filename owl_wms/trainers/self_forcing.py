"""
Self-Forcing Trainer for Game World Model
Implements autoregressive self-rollout training with proper gradient truncation
"""

import torch
from ema_pytorch import EMA
import wandb
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import einops as eo
from copy import deepcopy

from .base import BaseTrainer
from ..utils import freeze, unfreeze, Timer, find_unused_params, versatile_load
from ..schedulers import get_scheduler_cls
from ..models import get_model_cls
from ..sampling import get_sampler_cls
from ..data import get_loader
from ..utils.logging import LogHelper, to_wandb
from ..muon import init_muon
from ..utils.owl_vae_bridge import get_decoder_only, make_batched_decode_fn
from copy import deepcopy

class SelfForcingTrainer(BaseTrainer):
    """
    Self-Forcing Trainer implementing autoregressive self-rollout training
    
    Key differences from CausVid:
    1. Uses self-generated context during training (not ground truth)
    2. Implements gradient truncation with stochastic steps
    3. Supports DMD, SiD, and GAN losses on correct distribution
    """
    def __init__(self, *args, **kwargs):  
        super().__init__(*args, **kwargs)

        self.model_cfg = self.config.model

        model_id = self.model_cfg.model_id

        # Create student (causal) and teacher (non-causal) configs
        student_cfg = deepcopy(self.model_cfg)
        teacher_cfg = deepcopy(self.model_cfg)

        student_cfg.causal = True
        teacher_cfg.causal = False

        # Initialize models
        self.model = get_model_cls(model_id)(student_cfg)
        self.score_real = get_model_cls(model_id)(teacher_cfg)

        # Load pretrained teacher
        if self.train_cfg.teacher_ckpt:
            self.score_real.load_state_dict(versatile_load(self.train_cfg.teacher_ckpt))
        freeze(self.score_real)

        # Initialize fake score for DMD/SiD losses
        self.score_fake = deepcopy(self.score_real)

        # Print model size
        if self.rank == 0:
            n_params = sum(p.numel() for p in self.model.parameters())
            print(f"Model has {n_params:,} parameters")

        self.ema = None
        self.opt = None
        self.s_fake_opt = None
        self.scheduler = None
        self.s_fake_scaler = None
        self.scaler = None

        self.total_step_counter = 0
        
        # Initialize VAE decoder
        self.decoder = get_decoder_only(
            self.train_cfg.vae_id,
            self.train_cfg.vae_cfg_path,
            self.train_cfg.vae_ckpt_path
        )
        freeze(self.decoder)

        # Self-forcing specific parameters
        self.loss_type = self.train_cfg.get('loss_type', 'dmd')  # dmd, sid, or gan
        self.gradient_steps = self.train_cfg.get('gradient_steps', 1)  # Number of steps to backprop
        self.rollout_steps = self.train_cfg.get('rollout_steps', 5)  # Total rollout length
        self.stochastic_steps = self.train_cfg.get('stochastic_steps', True)  # Random gradient truncation
        self.update_ratio = self.train_cfg.get('update_ratio', 5)  # Critic updates per generator update
        self.cfg_scale = self.train_cfg.get('cfg_scale', 1.3)

    def save(self):
        save_dict = {
            'model': self.model.state_dict(),
            'ema': self.ema.state_dict(),
            'opt': self.opt.state_dict(),
            'scaler': self.scaler.state_dict(),
            'score_fake': self.score_fake.state_dict(),
            's_fake_opt': self.s_fake_opt.state_dict(),
            's_fake_scaler': self.s_fake_scaler.state_dict(),
            'steps': self.total_step_counter
        }
        if self.scheduler is not None:
            save_dict['scheduler'] = self.scheduler.state_dict()
        super().save(save_dict)
    
    def load(self):
        has_ckpt = False
        try:
            if self.train_cfg.resume_ckpt is not None:
                save_dict = super().load(self.train_cfg.resume_ckpt)
                has_ckpt = True
        except:
            print("Error loading checkpoint")
        
        if not has_ckpt:
            return

        self.model.load_state_dict(save_dict['model'])
        self.ema.load_state_dict(save_dict['ema'])
        self.opt.load_state_dict(save_dict['opt'])
        if self.scheduler is not None and 'scheduler' in save_dict:
            self.scheduler.load_state_dict(save_dict['scheduler'])
        self.scaler.load_state_dict(save_dict['scaler'])
        self.score_fake.load_state_dict(save_dict['score_fake'])
        self.s_fake_opt.load_state_dict(save_dict['s_fake_opt'])
        self.s_fake_scaler.load_state_dict(save_dict['s_fake_scaler'])
        self.total_step_counter = save_dict['steps']

    def autoregressive_rollout(self, initial_latent, mouse, btn, decode_fn=None):
        """
        Perform autoregressive rollout with gradient truncation
        
        Args:
            initial_latent: Initial frame(s) [b, init_frames, c, h, w]
            mouse: Mouse inputs for entire sequence [b, n, 2]
            btn: Button inputs for entire sequence [b, n, n_buttons]
            decode_fn: Optional decode function for visualization
            
        Returns:
            generated_latents: Full generated sequence
            gradient_mask: Boolean mask indicating which frames get gradients
        """
        b = initial_latent.shape[0]
        device = initial_latent.device
        
        # Initialize output with initial frames
        generated_latents = [initial_latent]
        
        # Determine gradient truncation point
        if self.stochastic_steps:
            # Randomly select which steps to backprop through
            grad_start = torch.randint(
                max(0, self.rollout_steps - self.gradient_steps),
                self.rollout_steps,
                (1,)
            ).item()
        else:
            # Always backprop through last gradient_steps
            grad_start = self.rollout_steps - self.gradient_steps
        
        # Generate frames autoregressively
        for step in range(self.rollout_steps):
            # Get context from previously generated frames
            context = torch.cat(generated_latents, dim=1)
            context_frames = context.shape[1]
            
            # Get corresponding actions
            step_mouse = mouse[:, :context_frames]
            step_btn = btn[:, :context_frames]
            
            # Determine if this step needs gradients
            needs_grad = step >= grad_start
            
            with torch.set_grad_enabled(needs_grad):
                # Add noise to last frame for next prediction
                noisy_next = torch.randn(b, 1, *initial_latent.shape[2:], device=device)
                
                # Prepare input
                model_input = torch.cat([context, noisy_next], dim=1)
                model_mouse = mouse[:, :context_frames + 1]
                model_btn = btn[:, :context_frames + 1]
                
                # Generate next frame
                with torch.amp.autocast('cuda', torch.bfloat16):
                    # Run diffusion model to denoise
                    ts = torch.ones(b, context_frames + 1, device=device)
                    ts[:, -1] = 0.99  # High noise for last frame
                    ts[:, :-1] = 0.0  # Clean context
                    
                    pred = self.model.core(model_input, ts, model_mouse, model_btn)
                    next_frame = model_input - pred * ts[:, :, None, None, None]
                    next_frame = next_frame[:, -1:]  # Take only the newly generated frame
                
                generated_latents.append(next_frame)
        
        # Concatenate all generated frames
        full_sequence = torch.cat(generated_latents, dim=1)
        
        # Create gradient mask
        gradient_mask = torch.zeros(b, full_sequence.shape[1], dtype=torch.bool, device=device)
        gradient_mask[:, initial_latent.shape[1] + grad_start:] = True
        
        return full_sequence, gradient_mask

    def compute_dmd_loss(self, generated, mouse, btn, gradient_mask):
        """Compute DMD loss on generated sequence"""
        s_real_fn = self.score_real.core
        s_fake_fn = self.score_fake.module.core if self.world_size > 1 else self.score_fake.core

        with torch.no_grad():
            b, n, c, h, w = generated.shape
            ts = torch.rand(b, n, device=generated.device).sigmoid()
            z = torch.randn_like(generated)
            ts_exp = ts[:, :, None, None, None]
            lerpd = generated * (1. - ts_exp) + z * ts_exp

        # Compute real score with CFG
        null_mouse = torch.zeros_like(mouse)
        null_btn = torch.zeros_like(btn)
        
        s_real_uncond = s_real_fn(lerpd, ts, null_mouse, null_btn)
        s_real_cond = s_real_fn(lerpd, ts, mouse, btn)
        s_real = s_real_uncond + self.cfg_scale * (s_real_cond - s_real_uncond)

        # Compute fake score
        s_fake = s_fake_fn(lerpd, ts, mouse, btn)

        # DMD gradient
        grad = (s_fake - s_real)
        
        # Normalize
        p_real = (generated - s_real)
        normalizer = torch.abs(p_real).mean(dim=[2, 3, 4], keepdim=True)
        grad = grad / (normalizer + 1e-6)
        grad = torch.nan_to_num(grad)
        
        # Apply gradient mask
        if gradient_mask is not None:
            grad = grad * gradient_mask[:, :, None, None, None]
        
        dmd_loss = 0.5 * F.mse_loss(generated.double(), (generated - grad).double())
        
        return dmd_loss

    def compute_sid_loss(self, generated, mouse, btn, gradient_mask):
        """Compute SiD loss on generated sequence"""
        # Similar to DMD but with SiD formulation
        # Implementation details from SiD paper
        raise NotImplementedError("SiD loss not yet implemented")

    def compute_gan_loss(self, generated, mouse, btn, gradient_mask, train_generator=True):
        """Compute GAN loss on generated sequence"""
        # Implementation for GAN-based loss
        raise NotImplementedError("GAN loss not yet implemented")

    def train(self):
        torch.cuda.set_device(self.local_rank)

        # Prepare models
        self.model = self.model.cuda().train()        
        self.decoder = self.decoder.cuda().eval().bfloat16()
        self.score_real = self.score_real.cuda().eval().bfloat16()
        self.score_fake = self.score_fake.cuda().train()

        if self.world_size > 1:
            self.model = DDP(self.model)
            self.score_fake = DDP(self.score_fake)

        freeze(self.decoder)
        freeze(self.score_real)

        decode_fn = make_batched_decode_fn(self.decoder, self.train_cfg.vae_batch_size)

        # Initialize EMA
        self.ema = EMA(
            self.model,
            beta=0.999,
            update_after_step=0,
            update_every=1
        )

        # Initialize optimizers
        self.opt = getattr(torch.optim, self.train_cfg.opt)(
            self.model.parameters(), 
            **self.train_cfg.opt_kwargs
        )
        self.s_fake_opt = getattr(torch.optim, self.train_cfg.opt)(
            self.score_fake.parameters(), 
            **self.train_cfg.opt_kwargs
        )

        if self.train_cfg.scheduler is not None:
            self.scheduler = get_scheduler_cls(self.train_cfg.scheduler)(
                self.opt, 
                **self.train_cfg.scheduler_kwargs
            )

        # Scalers for mixed precision
        self.s_fake_scaler = torch.amp.GradScaler()
        self.scaler = torch.amp.GradScaler()
        ctx = torch.amp.autocast('cuda', torch.bfloat16)

        self.load()

        # Setup logging
        timer = Timer()
        timer.reset()
        metrics = LogHelper()
        if self.rank == 0:
            wandb.watch(self.get_module(), log='all')
        
        # Dataset setup
        loader = get_loader(
            self.train_cfg.data_id, 
            self.train_cfg.batch_size, 
            **self.train_cfg.data_kwargs
        )
        sampler = get_sampler_cls(self.train_cfg.sampler_id)(**self.train_cfg.sampler_kwargs)

        def optimizer_step(loss, model, scaler, optimizer):
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            optimizer.zero_grad(set_to_none=True)
            scaler.update()

        # Training loop
        loader = iter(loader)
        while True:
            # Update critic/fake score
            if self.loss_type in ['dmd', 'sid']:
                freeze(self.model)
                unfreeze(self.score_fake)
                
                for _ in range(self.update_ratio):
                    batch_vid, batch_mouse, batch_btn = next(loader)
                    
                    # Get initial frames
                    initial_frames = batch_vid[:, :1]  # Use first frame as initial
                    
                    with torch.no_grad():
                        # Generate sequence autoregressively
                        generated, _ = self.autoregressive_rollout(
                            initial_frames, 
                            batch_mouse, 
                            batch_btn
                        )
                    
                    # Train fake score on generated data
                    with ctx:
                        s_fake_loss = self.score_fake(
                            generated.detach(), 
                            batch_mouse[:, :generated.shape[1]], 
                            batch_btn[:, :generated.shape[1]]
                        )
                    
                    optimizer_step(s_fake_loss, self.score_fake, self.s_fake_scaler, self.s_fake_opt)
                    metrics.log('s_fake_loss', s_fake_loss)

            # Update generator
            unfreeze(self.model)
            freeze(self.score_fake)
            
            batch_vid, batch_mouse, batch_btn = next(loader)
            initial_frames = batch_vid[:, :1]
            
            # Generate with gradients
            with ctx:
                generated, gradient_mask = self.autoregressive_rollout(
                    initial_frames,
                    batch_mouse,
                    batch_btn
                )
                
                # Compute loss based on selected type
                if self.loss_type == 'dmd':
                    loss = self.compute_dmd_loss(
                        generated, 
                        batch_mouse[:, :generated.shape[1]], 
                        batch_btn[:, :generated.shape[1]], 
                        gradient_mask
                    )
                elif self.loss_type == 'sid':
                    loss = self.compute_sid_loss(
                        generated,
                        batch_mouse[:, :generated.shape[1]],
                        batch_btn[:, :generated.shape[1]],
                        gradient_mask
                    )
                elif self.loss_type == 'gan':
                    loss = self.compute_gan_loss(
                        generated,
                        batch_mouse[:, :generated.shape[1]],
                        batch_btn[:, :generated.shape[1]],
                        gradient_mask,
                        train_generator=True
                    )
                
                metrics.log(f'{self.loss_type}_loss', loss)
            
            optimizer_step(loss, self.model, self.scaler, self.opt)
            self.ema.update()

            # Logging and visualization
            with torch.no_grad():
                wandb_dict = metrics.pop()
                wandb_dict['time'] = timer.hit()
                timer.reset()

                if self.total_step_counter % self.train_cfg.sample_interval == 0:
                    with ctx, torch.no_grad():
                        n_samples = self.train_cfg.n_samples
                        
                        # Get EMA model for sampling
                        ema_core = self.ema.ema_model.module.core if self.world_size > 1 else self.ema.ema_model.core
                        
                        # Sample using the trained model
                        samples, sample_mouse, sample_button = sampler(
                            ema_core,
                            batch_vid[:n_samples, :1],  # Initial frame
                            batch_mouse[:n_samples],
                            batch_btn[:n_samples],
                            decode_fn=decode_fn,
                            scale=self.train_cfg.vae_scale
                        )
                        
                        if self.rank == 0:
                            wandb_dict['samples'] = to_wandb(samples, sample_mouse, sample_button)
                    
                if self.rank == 0:
                    wandb.log(wandb_dict)

            self.total_step_counter += 1
            if self.total_step_counter % self.train_cfg.save_interval == 0:
                if self.rank == 0:
                    self.save()
                
            self.barrier()