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
from ..configs import TrainingConfig, TransformerConfig as ModelConfig, WANDBConfig as LoggingConfig
from torch.nn.parallel import DistributedDataParallel
import wandb
from ..utils.logging import LogHelper, to_wandb
from owl_wms.sampling.self_forcing_sampler import SelfForcingSampler, alpha, sigma
from owl_wms.muon import Muon

# NOTE Schedule for which denoising timestep to sample for DMD loss
T_SCHEDULE = [1000, 750, 500, 250]


def q_sample(x: Tensor, t: int) -> Tensor:     # add noise to input at timestep
    return alpha(t) * x + sigma(t) * (eps := torch.randn_like(x)), eps


class Loss_DistributionMatchingDistillation(nn.Module):

    def __init__(self,
                 teacher_model: nn.Module,
                 q_sample_fn: Callable[[Tensor, int], Tensor] = q_sample,
                 teacher_score_fn: Callable[[Tensor, int], Tensor] = None):

        super().__init__()
        self.teacher_model = teacher_model
        # -- teacher score fn
        self.teacher_score_fn: Callable[[Tensor, int], Tensor] = teacher_score_fn
        if not self.teacher_score_fn and hasattr(teacher_model, 'score_fn'):
            self.teacher_score_fn = teacher_model.score_fn
        assert self.teacher_score_fn is not None

        self.teacher_score_fn = torch.no_grad()(self.teacher_score_fn)

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
    def __init__(
            self,
            train_cfg:      TrainingConfig,
            logging_cfg:    LoggingConfig,
            model_cfg:      ModelConfig,
            global_rank:    int = 0,
            local_rank:     int = 0,
            world_size:     int = 1
        ):
        super().__init__(train_cfg, logging_cfg, model_cfg, global_rank, local_rank, world_size)
        self.max_grad_norm = getattr(self.train_cfg, 'max_grad_norm', 1.0)
        
        model_id = self.model_cfg.model_id

        student_cfg = deepcopy(self.model_cfg)
        teacher_cfg = deepcopy(self.model_cfg)

        student_cfg.causal = True
        teacher_cfg.causal = False
        # -- models
        self.causal_model: nn.Module        = get_model_cls(model_id)(student_cfg)
        if self.train_cfg.resume_ckpt is not None:
            assert self.train_cfg.student_ckpt is not None
            student_state_dict = versatile_load(self.train_cfg.student_ckpt)
            self.causal_model.core.load_state_dict(student_state_dict)
        unfreeze(self.causal_model)

        self.bidirectional_model: nn.Module = get_model_cls(model_id)(teacher_cfg)
        teacher_state_dict = versatile_load(self.train_cfg.teacher_ckpt)
        self.bidirectional_model.core.load_state_dict(teacher_state_dict)
        freeze(self.bidirectional_model)

        self.decoder:    nn.Module = get_decoder_only(self.train_cfg.vae_id,
                                                      self.train_cfg.vae_cfg_path,
                                                      self.train_cfg.vae_ckpt_path)

        self.decoder_fn            = make_batched_decode_fn(self.decoder, self.train_cfg.vae_batch_size)
        freeze(self.decoder)

        if self.rank == 0:
            n_params = sum(p.numel() for p in self.causal_model.parameters())
            print(f"Model has {n_params:,} parameters")

        # -- loss
        self.t_schedule = self.train_cfg.t_schedule
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
        self.context_len           = self.model_cfg.context_length
        self.frame_gradient_cutoff = self.train_cfg.frame_gradient_cutoff
        self.sampler               = SelfForcingSampler(
            model=self.causal_model,
            model_config=self.model_cfg,
            train_config=self.train_cfg,
            latent_shape=self.train_cfg.latent_shape,
            t_schedule=self.t_schedule,
            context_len=self.context_len,
            frame_gradient_cutoff=self.frame_gradient_cutoff,
            training=True,
            autocast=torch.bfloat16
        )

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
        # TODO make a ckpt manually from the 1b model that conforms to the below structure
        self.causal_model       .load_state_dict(save_dict.get('causal_model') or save_dict['model'])
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


    def _format_batch(self):
        try: return next(self.train_loader)
        except StopIteration: self.train_loader = iter(self.train_loader) ; return self._format_batch()

    def _construct_primers(self,
                           clip_bcnhw: Tensor,
                           mouse: Tensor,
                           btn: Tensor,
                           primer_len: int) -> list[dict[str, Tensor]]:
        return [
            dict(latent =clip_bcnhw[:, i:i+1],
                 mouse  =mouse     [:, i:i+1],
                 btn    =btn       [:, i:i+1])
            for i in range(primer_len)
        ]

    def _train_step(self):
        # NOTE: Auto-regressive rollout
        clip_bcnhw, mouse, btn  = self._format_batch()
        latent_conditions       = self._construct_primers(clip_bcnhw, mouse, btn, self.context_len)
        num_gen_frames          = self.model_cfg.n_frames - self.context_len
        # TODO WTF is even going on here
        # NOTE The issue is that we need latent conditioning.
        # This could be the groundtruth latents, but they need to be in {'latent': ..., 'mouse': ..., 'btn': ...} format.
        # This means that we might need to chunk the groundtruth+mouse+btn into two - but is this correct?
        student_clip_bcnhw      = self.sampler.autoregressive_rollout(btn  [:, self.context_len:],
                                                                      mouse[:, self.context_len:],
                                                                      latent_conditioning=latent_conditions,
                                                                      num_frames=num_gen_frames)
        t: int                  = random.choice(self.t_schedule)

        loss = self.loss_fn.forward(
            student_model=self.causal_model,
            student_clip=student_clip_bcnhw,
            groundtruth_clip=clip_bcnhw,
            t=t,
            student_score_fn=self.causal_model.score_fn
        )
        
        self.scaler.scale(loss).backward() ; self.scaler.unscale_(self.opt)
        grad_norm = clip_grad_norm_(self.causal_model.parameters(), self.max_grad_norm)
        self.scaler.step(self.opt)         ; self.opt.zero_grad() ; self.scaler.update()

        return {
            'student_clip':     student_clip_bcnhw,
            'groundtruth_clip': clip_bcnhw,
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

    def get_module(self, ema = False):
        if ema: return self.ema.ema_model if self.world_size == 1 else self.ema.ema_model.module
        else:   return self.causal_model  if self.world_size == 1 else self.causal_model.module

    @torch.no_grad()
    def evaluate(self):
        if self.rank != 0: return
        self.causal_model.eval()
        try:
            groundtruth, mouse, btn = self._format_batch()
            # TODO How does this get conditioning from the groundtruth? I am guessing we need some shenanigans with
            # latents for groundtruth.
            student_clip = self.sampler.autoregressive_rollout(btn,
                                                               mouse,
                                                               latent_conditioning=groundtruth,
                                                               num_frames=self.model_cfg.n_frames)
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