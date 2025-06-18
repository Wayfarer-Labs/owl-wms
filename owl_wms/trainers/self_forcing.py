from __future__ import annotations
import random
from copy import deepcopy
from ema_pytorch import EMA
import torch ; from torch import Tensor
from typing import Callable
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
from .base import BaseTrainer
from ..models.gamerft_audio import GameRFTAudio
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
from ..utils.logging import LogHelper, to_wandb_av
from owl_wms.sampling.self_forcing_sampler import SelfForcingSampler, q_sample


class Loss_DistributionMatchingDistillation(nn.Module):

    def __init__(self,
                 teacher_model: nn.Module,
                 q_sample_fn: Callable[[Tensor, ...], tuple[Tensor, Tensor]] = q_sample,
                 teacher_score_fn: Callable[[Tensor, Tensor], Tensor] | None = None):

        super().__init__()
        self.teacher_model = teacher_model
        # -- teacher score fn
        if not teacher_score_fn and hasattr(teacher_model, 'score_fn'):
            self.teacher_score_fn = teacher_model.score_fn
        assert self.teacher_score_fn is not None
        self.teacher_score_fn: Callable[[Tensor, Tensor], Tensor] = torch.no_grad()(self.teacher_score_fn)
        self.q_sample_fn = q_sample_fn

    def forward(self,
            score_student_clip:  Tensor,  # [B, N, C, H, W]
            score_student_audio: Tensor,  # [B, N, C] 
            t:                   Tensor,  # [B, N] containing initial denoising timestep for its corresponding frame in scores
            mouse:               Tensor,  # [B, N, 2]
            btn:                 Tensor,  # [B, N, n_buttons]
        ) -> dict[str, Tensor]:
        noisy_clip,  _ = self.q_sample_fn(score_student_clip,  t)
        noisy_audio, _ = self.q_sample_fn(score_student_audio, t)
        
        with torch.no_grad():
            score_teacher_clip, score_teacher_audio = self.teacher_score_fn(noisy_clip, t,
                                                                            mouse, btn,
                                                                            noisy_audio)

        clip_loss  = 0.5 * ((score_student_clip  - score_teacher_clip.detach())  ** 2).mean()
        audio_loss = 0.5 * ((score_student_audio - score_teacher_audio.detach()) ** 2).mean()

        return {
            'clip_loss':  clip_loss,
            'audio_loss': audio_loss,
            'total_loss': clip_loss + audio_loss,
        }


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
        self.causal_model: GameRFTAudio = get_model_cls(model_id)(student_cfg)
        if self.train_cfg.student_ckpt is not None:
            assert self.train_cfg.student_ckpt is not None
            student_state_dict = versatile_load(self.train_cfg.student_ckpt)
            self.causal_model.core.load_state_dict(student_state_dict)
        unfreeze(self.causal_model)

        self.bidirectional_model: GameRFTAudio = get_model_cls(model_id)(teacher_cfg)
        teacher_state_dict = versatile_load(self.train_cfg.teacher_ckpt)
        self.bidirectional_model.core.load_state_dict(teacher_state_dict)
        freeze(self.bidirectional_model)

        self.decoder: nn.Module = get_decoder_only(self.train_cfg.vae_id,
                                                      self.train_cfg.vae_cfg_path,
                                                      self.train_cfg.vae_ckpt_path)
        self.decoder_fn         = make_batched_decode_fn(self.decoder, self.train_cfg.vae_batch_size)

        self.audio_decoder: nn.Module = get_decoder_only(self.train_cfg.audio_vae_id,
                                                      self.train_cfg.audio_vae_cfg_path,
                                                      self.train_cfg.audio_vae_ckpt_path)
        self.audio_decoder_fn   = make_batched_decode_fn(self.audio_decoder, self.train_cfg.audio_vae_batch_size)
        
        freeze(self.decoder)
        freeze(self.audio_decoder)

        if self.rank == 0:
            n_params = sum(p.numel() for p in self.causal_model.parameters())
            print(f"Model has {n_params:,} parameters")


        # -- metrics & logging
        self.metrics            = LogHelper()
        self.total_step_counter = 0
        if self.rank == 0:        wandb.watch(self.get_module(), log = 'all')

        # -- make sure that eval frequency is a multiple of log frequency
        self.train_cfg.sample_interval = (self.train_cfg.log_interval * (self.train_cfg.sample_interval // self.train_cfg.log_interval))

        # -- data
        self.train_loader   = get_loader(self.train_cfg.data_id,
                                         self.train_cfg.batch_size,
                                         **self.train_cfg.data_kwargs)
        self.train_loader   = iter(self.train_loader)
        
        # -- loss
        self.t_schedule = self.train_cfg.t_schedule
        self.loss_fn    = Loss_DistributionMatchingDistillation(self.bidirectional_model)

        # -- auto-load
        self.load_bidirectional()  # always has to load cause an assumption we make is that it always exists
        self.load_causal()         # only loads if we have a checkpoint to resume from or if student is specified
        
        # -- hardware - done after loading models so it casts to bfloat16
        self.device = torch.device("cuda", self.local_rank)
        self.init_hardware()

        # -- optim tomfoolery, shenanigans, etc. - needs to be done after init_hardware so .cuda() calls work as intended
        self.ema        = EMA(self.causal_model, beta=0.999, update_after_step=0, update_every=1)
        self.opt        = torch.optim.AdamW(self.causal_model.parameters(), **self.train_cfg.opt_kwargs)
        self.scaler     = torch.amp.GradScaler()
        self.ctx        = torch.amp.autocast('cuda', torch.float32)
        self.scheduler  = get_scheduler_cls(self.train_cfg.scheduler)(self.opt, **self.train_cfg.scheduler_kwargs)
        

        # -- sampler - initialize this after the hardware so it detects the device
        self.batch_size            = self.train_cfg.batch_size
        self.context_len           = self.model_cfg.context_length
        self.frame_gradient_cutoff = self.train_cfg.frame_gradient_cutoff
        self.num_gen_frames        = self.model_cfg.n_frames - self.context_len
        self.sampler               = SelfForcingSampler(
            model=self.causal_model,
            model_config=self.model_cfg,
            batch_size=self.batch_size,
            latent_shape=self.train_cfg.latent_shape,
            t_schedule=self.t_schedule,
            context_len=self.context_len,
            num_gen_frames=self.num_gen_frames,
            frame_gradient_cutoff=self.frame_gradient_cutoff,
            training=True,
            autocast=torch.bfloat16
        )

        # -- ddp
        self.causal_model          = DistributedDataParallel(self.causal_model) if self.world_size > 1 else self.causal_model

    def init_hardware(self):
        self.causal_model        = self.causal_model.cuda().train()
        self.bidirectional_model = self.bidirectional_model\
                                                    .cuda().eval().bfloat16()
        self.loss_fn             = self.loss_fn     .cuda().eval().bfloat16()

    def load_bidirectional(self):
        assert self.train_cfg.teacher_ckpt is not None
        self.bidirectional_model.core.load_state_dict(versatile_load(self.train_cfg.teacher_ckpt))

    def load_causal(self):
        has_ckpt = False
        try:
            if self.train_cfg.student_ckpt is not None:
                save_dict = super().load_causal(self.train_cfg.student_ckpt)
                has_ckpt = True
        except:
            print("Error loading checkpoint")
        
        if not has_ckpt:
            print("No checkpoint could be detected, initializing from scratch")
            return

        self.causal_model       .load_state_dict(save_dict.get('causal_model') or save_dict['model'])
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

    def _format_batch(self) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        try:
            batch: tuple[Tensor, ...] = next(self.train_loader)
            return tuple(item.to(self.device).float() for item in batch)
        except StopIteration:
            self.train_loader = iter(self.train_loader)
            return self._format_batch()

    def _construct_primers(self,
                           clip_bnchw: Tensor,
                           mouse: Tensor,
                           btn: Tensor,
                           audio: Tensor) -> list[dict[str, Tensor]]:

        return [ dict(latent = clip_bnchw[:, i:i+1],
                       mouse = mouse     [:, i:i+1],
                       btn   = btn       [:, i:i+1],
                       audio = audio     [:, i:i+1]) for i in range(btn.shape[1]) ]

    def _train_step(self):
        clip_bnchw, audio, mouse, btn   = self._format_batch()
        latent_conditions               = self._construct_primers(clip_bnchw, mouse, btn, audio)[   :self.num_gen_frames]
        scores_video, scores_audio, t_b = self.sampler.autoregressive_rollout(btn               [:, -self.num_gen_frames:],
                                                                             mouse              [:, -self.num_gen_frames:],
                                                                             audio              [:, -self.num_gen_frames:],
                                                                             latent_conditions)
        
        loss_info = self.loss_fn.forward(
            score_student_clip  = scores_video,  # fully denoised frame latent from t->0
            score_student_audio = scores_audio,  # fully denoised audio latent from t->0
            t                   = t_b,           # [B, 1] containing initial denoising timestep for its corresponding frame in scores
            mouse               = mouse[:, -self.num_gen_frames:],
            btn                 = btn  [:, -self.num_gen_frames:],
        )

        self.scaler.scale(loss_info['total_loss']).backward() ; self.scaler.unscale_(self.opt)
        grad_norm = clip_grad_norm_(self.causal_model.parameters(), self.max_grad_norm)
        self.scaler.step(self.opt)         ; self.opt.zero_grad() ; self.scaler.update()

        return {
            'groundtruth_clip': clip_bnchw,
            'groundtruth_audio':audio,
            'student_clip':     scores_video,
            'student_audio':    scores_audio,
            'mouse':            mouse,
            'btn':              btn,
            'grad_norm':        grad_norm,
            **loss_info,
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
    def evaluate(self, info: dict):
        if self.rank != 0: return
        try:
            self.causal_model.eval()
            # -- get relevant samples
            student_clip      = info['student_clip']
            student_audio     = info['student_audio']
            groundtruth_clip  = info['groundtruth_clip']
            groundtruth_audio = info['groundtruth_audio']
            mouse, btn        = info['mouse'], info['btn']
            # -- overlay student frames on groundtruth ones
            n_frames                        = student_clip.shape[1]
            overlay_student                 = groundtruth_clip.clone()
            overlay_student [:, -n_frames:] = student_clip             # replace last n frames of groundtruth with student frames
            overlay_student                 = self.decoder_fn(overlay_student.bfloat16())
            # -- overlay student audio on groundtruth ones
            overlay_audio                   = groundtruth_audio.clone()
            overlay_audio   [:, -n_frames:] = student_audio
            overlay_audio                   = self.audio_decoder_fn(overlay_audio.bfloat16())

            wandb.log({
                'student_samples':     to_wandb_av(overlay_student,  overlay_audio,     mouse, btn, gather=True, max_samples=8),
                'groundtruth_samples': to_wandb_av(groundtruth_clip, groundtruth_audio, mouse, btn, gather=True, max_samples=8),
            }, step=self.total_step_counter, commit=True)
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"Evaluation failed: {e}") ; breakpoint()
        finally:               self.causal_model.train()

    @torch.no_grad()
    def _log_step(self, info: dict):
        # TODO fix this garbaje. also make it only samplein evaluate & only log metrics in should_log
        if self.should_sample:  self.evaluate(info)
        if not self.should_log: return

        wandb.log(self.metrics.pop(), step=self.total_step_counter, commit=False)
        wandb.log({'lr': self.scheduler.get_last_lr()[0]}, step=self.total_step_counter, commit=False)
        return 

    def train(self):
        timer = Timer()
        while not self.should_break:
            timer.reset()
            with self.ctx:
                info = self._train_step()
                self.ema.update() ; info['time'] = timer.hit()
            
            self.metrics.log_dict({
                'total_loss':   info['total_loss'],
                'grad_norm':    info['grad_norm'],
                'time':         info['time']
            })

            self.total_step_counter += 1
            self._log_step(info)
            self.barrier()
            if self.should_save: self.save()

        self.save()