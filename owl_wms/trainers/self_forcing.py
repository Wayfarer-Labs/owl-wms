import random
from copy import deepcopy
from ema_pytorch import EMA
import torch ; from torch import Tensor
from typing import Callable
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
from .base import BaseTrainer
from ..models import get_model_cls
from ..utils import (
    freeze, unfreeze,
    Timer, versatile_load
)
from ..data import get_loader
from ..utils.owl_vae_bridge import get_decoder_only, make_batched_decode_fn
from ..sampling import get_sampler_cls
from ..schedulers import get_scheduler_cls
from torch.nn.parallel import DistributedDataParallel
import wandb
from ..utils.logging import LogHelper, to_wandb


# NOTE Schedule for which denoising timestep to sample for DMD loss
T_SCHEDULE = [1000, 750, 500, 250]

def alpha(t: int) -> Tensor:            # signal coefficient
    return torch.cos(torch.tensor(t, device='cuda') * torch.pi / 2 / 1000)
def sigma(t: int) -> Tensor:            # noise coefficient
    return torch.sin(torch.tensor(t, device='cuda') * torch.pi / 2 / 1000)
def q_sample(x: Tensor, t: int) -> Tensor:     # add noise to input at timestep
    return alpha(t) * x + sigma(t) * (eps := torch.randn_like(x)), eps
def action_conditioning(*, mouse: Tensor, btn: Tensor) -> dict[str, Tensor]:
    return {'mouse': mouse, 'button': btn}


class KV_Cache:
    """
    Rolling buffer that stores one KV tensor **per frame**.
    It evicts the oldest entry once *max_size* is exceeded so the memory footprint stays O(max_size).
    The cache itself is just a Python list; *model* decides how to concatenate / attend over it.
    """
    def __init__(self, max_size: int):
        self._buf: list[Tensor] = []
        self.max_size           = max_size

    def append(self, kv: Tensor) -> None:
        self._buf.append(kv)
        if self.max_size and len(self._buf) > self.max_size:
            self._buf.pop(0)

    def get(self) -> list[Tensor]: return self._buf

    def __len__(self): return len(self._buf)

    def clear(self): self._buf.clear()



class Loss_DistributionMatchingDistillation(nn.Module):

    def __init__(self,
                 teacher_model: nn.Module,
                 q_sample_fn: Callable[[Tensor, int], Tensor] = q_sample,
                 teacher_score_fn: Callable[[Tensor, int], Tensor] = None):

        super().__init__()
        self.teacher_model = teacher_model
        self.teacher_score_fn: Callable[[Tensor, int], Tensor] = teacher_score_fn

        if not self.teacher_score_fn and hasattr(teacher_model, 'score_fn'):
            self.teacher_score_fn = teacher_model.score_fn

        assert self.teacher_score_fn is not None

        self.device = next(teacher_model.parameters()).device
        self.q_sample_fn = q_sample_fn


    def forward(self,
            student_model:      nn.Module, *,
            student_clip:       Tensor,
            groundtruth_clip:   Tensor,
            t:                  int | Tensor,
            student_score_fn:   Callable[[Tensor, int], Tensor] | None = None
        ) -> Tensor:
        t = torch.tensor(t, device=self.device) if isinstance(t, int) else t

        student_score_fn = student_score_fn or student_model.score_fn
        assert student_score_fn is not None

        noisy_student, _     = self.q_sample_fn(student_clip, t)
        noisy_groundtruth, _ = self.q_sample_fn(groundtruth_clip, t)

        with torch.no_grad():
            score_groundtruth_teacher = self.teacher_score_fn(noisy_groundtruth, t)

        score_student = student_score_fn(noisy_student, t)
        loss = 0.5 * ((score_student - score_groundtruth_teacher.detach()) ** 2).mean()
        return loss


class SelfForcingTrainer(BaseTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_grad_norm = getattr(self.train_cfg, 'max_grad_norm', 1.0)
        
        model_id = self.model_cfg.model_id

        student_cfg = deepcopy(self.model_cfg)
        teacher_cfg = deepcopy(self.model_cfg)

        student_cfg.causal = True
        teacher_cfg.causal = False
        # -- models
        self.causal_model: nn.Module        = get_model_cls(model_id)(student_cfg)
        self.causal_model.load_state_dict(versatile_load(self.train_cfg.student_ckpt))
        unfreeze(self.causal_model)

        self.bidirectional_model: nn.Module = get_model_cls(model_id)(teacher_cfg)
        self.bidirectional_model.load_state_dict(versatile_load(self.train_cfg.teacher_ckpt))
        freeze(self.bidirectional_model)

        self.decoder:    nn.Module = get_decoder_only()
        self.decoder_fn            = make_batched_decode_fn(self.decoder, self.train_cfg.vae_batch_size)
        freeze(self.decoder)

        if self.rank == 0:
            n_params = sum(p.numel() for p in self.causal_model.parameters())
            print(f"Model has {n_params:,} parameters")

        # -- loss
        self.t_schedule = T_SCHEDULE
        self.loss_fn    = Loss_DistributionMatchingDistillation(self.bidirectional_model)

        # -- optim tomfoolery, shenanigans, etc.
        self.ema        = EMA(self.causal_model, beta=0.999, update_after_step=0, update_every=1)
        self.opt        = torch.optim.AdamW(self.causal_model.parameters(), **self.train_cfg.opt_kwargs)
        self.scaler     = torch.amp.GradScaler()
        self.ctx        = torch.amp.autocast('cuda', torch.bfloat16)
        self.scheduler  = get_scheduler_cls(self.train_cfg.scheduler)(self.opt, **self.train_cfg.scheduler_kwargs)
        
        # -- metrics & logging
        self.metrics    = LogHelper()
        if self.rank == 0:
            wandb.watch(self.get_module(), log = 'all')

        # -- data
        self.train_loader   = get_loader(self.train_cfg.data_id,
                                         self.train_cfg.batch_size,
                                         **self.train_cfg.data_kwargs)
        # -- sampler
        # TODO TODO TODO TODO TODO TODO TODO Make this
        self.context_len    = self.train_cfg.context_length
        self.sampler        = get_sampler_cls(self.train_cfg.sampler_id)()

        # -- hardware
        self.device = torch.device("cuda", self.local_rank)
        self.init_hardware()
        # -- auto-load
        self.load()
        # -- ddp
        if self.world_size > 1:
            self.causal_model    = DistributedDataParallel(self.causal_model)

    def init_hardware(self):
        self.causal_model        = self.causal_model.cuda().train()
        self.bidirectional_model = self.bidirectional_model\
                                                    .cuda().eval().bfloat16()
        self.loss_fn             = self.loss_fn     .cuda().eval().bfloat16()

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
        
        self.causal_model       .load_state_dict(save_dict['causal_model'])
        self.bidirectional_model.load_state_dict(save_dict['bidirectional_model'])
        self.ema                .load_state_dict(save_dict['ema'])
        self.opt                .load_state_dict(save_dict['opt'])
        self.scaler             .load_state_dict(save_dict['scaler'])
        if 'scheduler' in save_dict:
            self.scheduler      .load_state_dict(save_dict['scheduler'])
        self.decoder            .load_state_dict(save_dict['decoder'])
        self.total_step_counter = save_dict['steps']

    def save(self):
        save_dict = {
            'causal_model':        self.causal_model        .state_dict(),
            'bidirectional_model': self.bidirectional_model .state_dict(),
            'ema':                 self.ema                 .state_dict(),
            'opt':                 self.opt                 .state_dict(),
            'scaler':              self.scaler              .state_dict(),
            'decoder':             self.decoder             .state_dict(),
            'steps':               self.total_step_counter,
            **({'scheduler':       self.scheduler           .state_dict()}
               if self.scheduler is not None else {})
        }

        super().save(save_dict)

    def autoregressive_rollout(self, *,
                               use_teacher: bool = False,
                               context_len: int = None,
                               action_conditioning: dict[str, Tensor] = None,
                               latent_conditioning: Tensor = None,
                               frame_gradient_cutoff: int = 0) -> list[Tensor]:
        """Roll out *num_frames* latent frames with gradient-truncated denoising.

        Returns
        -------
        list[Tensor]
            length = N ; each tensor has shape **(B, C, F, H, W)**
            These are the clean x̂_{i,0} latents used later by compute_dmd_loss.
        """
        device               = self.device
        batch_size           = self.train_cfg.batch_size
        latent_shape_cfhw    = self.train_cfg.latent_shape          # (C,F,H,W)
        num_frames_n         = self.train_cfg.num_frames            # N
        t_schedule           = self.t_schedule                      # [t_T … t_1]
        model                = self.bidirectional_model if use_teacher else self.causal_model

        cache                       = KV_Cache(max_size=context_len or self.context_len)
        latents_bcFHW: list[Tensor] = [] # collects outputs

        # pick which denoising step keeps gradients for this rollout. because this is stochastic,
        # in expectation each timestep will get gradient signal over the entire training run.
        grad_step_s = random.randint(1, len(t_schedule))

        for _ in range(num_frames_n):
            # 1) initialise latent with pure noise at the *max* timestep t_T
            latent_t_bcFHW = torch.randn(batch_size, *latent_shape_cfhw, device=device)

            # 2) denoise backward through all timesteps
            for step_idx, t in enumerate(reversed(t_schedule), start=1):
                keep_grad = (step_idx == grad_step_s)

                with torch.set_grad_enabled(keep_grad):
                    x_hat_bcFHW = model.forward(latent_t_bcFHW, t=t, kv_cache=cache.get())

                if keep_grad:
                    # store clean frame for outer loss + push KV to cache
                    latents_bcFHW.append(x_hat_bcFHW)
                    kv_single = model.get_kv(x_hat_bcFHW, t=0)  # model decides shape
                    cache.append(kv_single)
                else:
                    x_hat_bcFHW = x_hat_bcFHW.detach()

                # 3) re-noise unless we just reached t=0
                if step_idx < len(t_schedule):
                    prev_t = t_schedule[-(step_idx + 1)]
                    eps    = torch.randn_like(x_hat_bcFHW)
                    latent_t_bcFHW = alpha(prev_t) * x_hat_bcFHW + sigma(prev_t) * eps

        return latents_bcFHW

    def _format_batch(self):
        try: return next(self.train_loader)
        except StopIteration: self.train_loader = iter(self.train_loader) ; return self._format_batch()


    def _train_step(self):
        # NOTE: Auto-regressive rollout
        student_clip_bcnhw      = self.autoregressive_rollout()
        groundtruth_clip_bcnhw, mouse, btn = self._format_batch()
        t: int                  = random.choice(self.t_schedule)

        loss = self.loss_fn.forward(
            student_model=self.causal_model,
            student_clip=student_clip_bcnhw,
            groundtruth_clip=groundtruth_clip_bcnhw,
            t=t,
            student_score_fn=self.causal_model.score_fn
        )
        
        self.scaler.scale(loss).backward() ; self.scaler.unscale_(self.opt)
        grad_norm = clip_grad_norm_(self.causal_model.parameters(), self.max_grad_norm)
        self.scaler.step(self.opt)         ; self.opt.zero_grad() ; self.scaler.update()

        return {
            'student_clip':     student_clip_bcnhw,
            'groundtruth_clip': groundtruth_clip_bcnhw,
            'mouse':            mouse,
            'btn':              btn,
            't':                t,
            'grad_norm':        grad_norm,
            'loss':             loss,
        }

    @property
    def should_log(self):
        return self.rank == 0 and self.total_step_counter % self.train_cfg.log_interval == 0
    
    @property
    def should_sample(self):
        return self.rank == 0 and self.total_step_counter % self.train_cfg.sample_interval == 0

    @property
    def should_save(self):
        return self.rank == 0 and self.total_step_counter % self.train_cfg.save_interval == 0
    
    @property
    def should_break(self):
        return hasattr(self.train_cfg, 'max_steps') and self.total_step_counter >= self.train_cfg.max_steps


    @torch.no_grad()
    def evaluate(self):
        if self.rank != 0: return
        self.causal_model.eval()
        try:
            _, mouse, btn = self._format_batch()
            # TODO How does this get conditioning from the groundtruth? I am guessing we need some shenanigans with
            # latents for groundtruth.
            student_clip = self.autoregressive_rollout()
            eval_video = to_wandb(student_clip, mouse, btn,  gather=False, max_samples=4)
            wandb.log({'eval_samples': eval_video}, step=self.total_step_counter)
        
        except Exception as e:  print(f"Evaluation failed: {e}")
        finally:                self.causal_model.train()

    @torch.no_grad()
    def _log_step(self, info: dict):
        if self.should_log:
            wandb.log(self.metrics.pop(), step=self.total_step_counter)
            if self.scheduler is not None: wandb.log({'lr': self.scheduler.get_last_lr()[0]},
                                                        step=self.total_step_counter)
            try:                  
                wandb.log({
                    'student_samples':      to_wandb(info['student_clip'], info['mouse'], info['btn'],
                                                        gather=True, max_samples=8),
                    'groundtruth_samples':  to_wandb(info['groundtruth_clip'], info['mouse'], info['btn'],
                                                        gather=True, max_samples=8),
                }, step=self.total_step_counter)

            except Exception as e: print(f"Warning: Failed to log videos: {e}")

        if self.should_sample: self.evaluate()


    def train(self):
        timer = Timer()
        while not self.should_break:
            timer.reset()
            with self.ctx:
                info = self._train_step()
                self.ema.update() ; info['time'] = timer.hit()
            
            self.metrics.log_dict({
                'loss':         info['loss'],
                'grad_norm':    info['grad_norm'],
                'time':         info['time'],
                't':            info['t']
            })

            self.total_step_counter += 1
            self._log_step(info)
            self.barrier()
            if self.should_save: self.save()

        self.save()