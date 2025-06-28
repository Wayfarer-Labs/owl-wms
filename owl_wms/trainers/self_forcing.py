from __future__ import annotations
from ast import BoolOp
import os
import random
from copy import deepcopy
from ema_pytorch import EMA
import torch ; from torch import Tensor
from contextlib import nullcontext
from typing import Callable, ContextManager, Optional, Any
import einops as eo
import torch.nn.functional as F
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_, get_total_norm
from torch.nn.parallel import DistributedDataParallel
from owl_wms.trainers.base import BaseTrainer
from functools import partial, wraps
from owl_wms.models.gamerft_audio import GameRFTAudio
from owl_wms.models import get_model_cls
from owl_wms.utils import (
    freeze, unfreeze,
    SamiTimer as Timer, versatile_load
)
from owl_wms.data import get_loader
from owl_wms.utils.owl_vae_bridge import get_decoder_only, make_batched_decode_fn, make_batched_audio_decode_fn
from owl_wms.schedulers import get_scheduler_cls
from owl_wms.configs import TrainingConfig, TransformerConfig as ModelConfig, WANDBConfig as LoggingConfig
import wandb
from owl_wms.utils.logging import LogHelper, to_wandb_av
from owl_wms.sampling.self_forcing_sampler import SelfForcingSampler, fwd_rectified_flow
from owl_wms.sampling.av_window import AVWindowSampler
from owl_wms.nn.kv_cache import KVCache

def remove_learned_abs_poc_enc(model: GameRFTAudio):
    model.core.pos_enc = nn.Identity()
    return model


class Loss_SelfForcing(nn.Module):

    def __init__(self,
            teacher:   GameRFTAudio,
            critic:    GameRFTAudio,
            student:   GameRFTAudio,
            teacher_cfg_weight: float = 1.3,
            student_cfg_weight: float = 0.0,
            critic_cfg_weight:  float = 0.0,
            q_sample_fn:        Callable[[Tensor, Tensor, bool], tuple[Tensor, Tensor]] = fwd_rectified_flow,
            normalize:          bool = False,
            normalize_eps: float      = 1e-6,
            debug_basic: bool = False,
        ):
        super().__init__()
        self.q_sample_fn        = q_sample_fn
        self.critic    = critic
        self.teacher   = teacher ; freeze(self.teacher)
        self.student   = student

        # -- cfg
        self.teacher_cfg_weight = teacher_cfg_weight
        self.critic_cfg_weight  = critic_cfg_weight
        self.student_cfg_weight = student_cfg_weight
        # -- 
        self.normalize          = normalize
        self.normalize_eps      = normalize_eps
        self.debug_basic        = debug_basic

    def loss_distribution_matching_distillation(self,
            causal_latent_video: Tensor,  # [B, N, C, H, W]
            causal_latent_audio: Tensor,  # [B, N, C] 
            t:                   Tensor,  # [B, N] containing initial denoising timestep for its corresponding frame in scores
            mouse:               Tensor,  # [B, N, 2]
            btn:                 Tensor,  # [B, N, n_buttons]
            normalize:           Optional[bool] = None,
            mask_frame_idx:      Optional[int]  = None, # index uptil frames are ignored. "2" will ignore the first two frames
        ) -> dict[str, Tensor]:

        assert sum(x.numel() for x in self.critic.parameters() if x.requires_grad) == 0, 'critic must be frozen'
        assert sum(x.numel() for x in self.student.parameters() if x.requires_grad) > 0, 'student must not be frozen'
        assert sum(x.numel() for x in self.teacher.parameters() if x.requires_grad) == 0, 'teacher must be frozen'
        
        normalize               = normalize if normalize is not None else self.normalize

        if mask_frame_idx is not None:
            causal_latent_video = causal_latent_video[:, mask_frame_idx:] # -- causvid: 178: vid
            causal_latent_audio = causal_latent_audio[:, mask_frame_idx:]
            mouse               = mouse              [:, mask_frame_idx:]
            t                   = t                  [:, mask_frame_idx:]
            btn                 = btn                [:, mask_frame_idx:]

        noisy_causal_clip,  *_  = self.q_sample_fn(causal_latent_video, eo.repeat(t, 'b n -> b n 1 1 1'))
        noisy_causal_audio, *_  = self.q_sample_fn(causal_latent_audio, eo.repeat(t, 'b n -> b n 1'))
        
        # -- teacher predicts on groundtruth noisy data, with cfg
        velocity_clip_teacher, velocity_audio_teacher = self.teacher.velocity_fn(noisy_causal_clip, t,
                                                                        mouse, btn, noisy_causal_audio,
                                                                        cfg_weight=self.teacher_cfg_weight) # -- causvid: 200: s_real
        if self.debug_basic:
            velocity_clip_teacher = torch.zeros_like(velocity_clip_teacher)
            velocity_audio_teacher = torch.zeros_like(velocity_audio_teacher)

        # -- critic score predicts on student noisy data, without cfg
        velocity_clip_critic, velocity_audio_critic   = self.critic.velocity_fn(noisy_causal_clip, t,
                                                                        mouse, btn, noisy_causal_audio,
                                                                        cfg_weight=self.critic_cfg_weight) # -- casuvid: 202: s_fake
        
        grad_clip  = (velocity_clip_critic  - velocity_clip_teacher)
        grad_audio = (velocity_audio_critic - velocity_audio_teacher)
        
        if normalize:
            p_real_clip     = (causal_latent_video - velocity_clip_teacher)
            normalizer_clip = torch.abs(p_real_clip) .mean(dim=[1,2,3,4], keepdim=True)
            grad_clip       = grad_clip / (normalizer_clip + self.normalize_eps)

            p_real_audio     = (causal_latent_audio - velocity_audio_teacher)
            normalizer_audio = torch.abs(p_real_audio).mean(dim=[1,2],     keepdim=True)
            grad_audio       = grad_audio / (normalizer_audio + self.normalize_eps)

        clip_loss  = 0.5 * F.mse_loss(causal_latent_video.double(), 
                                      causal_latent_video.double() - grad_clip.double())  # -- causvid: 212
        audio_loss = 0.5 * F.mse_loss(causal_latent_audio.double(),
                                      causal_latent_audio.double() - grad_audio.double()) # -- causvid: 212

        return {
            'clip_loss':  clip_loss,
            'audio_loss': audio_loss,
            'total_loss': clip_loss + audio_loss,
        }

    def loss_rectified_flow(self,         
            causal_latent_video: Tensor, # [B, N, C, H, W]
            causal_latent_audio: Tensor, # [B, N, C] 
            t:                   Tensor, # [B, N] containing initial denoising timestep for its corresponding frame in scores
            mouse:               Tensor, # [B, N, 2]
            btn:                 Tensor, # [B, N, n_buttons]
        ) -> dict[str, Tensor]:
        assert sum(x.numel() for x in self.critic.parameters() if x.requires_grad) > 0, 'critic must not be frozen'
        assert sum(x.numel() for x in self.student.parameters() if x.requires_grad) == 0, 'student must be frozen'
        assert sum(x.numel() for x in self.teacher.parameters() if x.requires_grad) == 0, 'teacher must be frozen'
        
        noisy_causal_clip,  noise_c = self.q_sample_fn(causal_latent_video, eo.repeat(t, 'b n -> b n 1 1 1'))
        noisy_causal_audio, noise_a = self.q_sample_fn(causal_latent_audio, eo.repeat(t, 'b n -> b n 1'))
        # TODO SAMI Might be a bit more complicated cause the causal video must be generated with no-grad?

        velocity_clip_critic, velocity_audio_critic   = self.critic.velocity_fn(noisy_causal_clip, t,
                                                                            mouse, btn, noisy_causal_audio,
                                                                            cfg_weight=self.critic_cfg_weight, kv_cache = None) # no need for kv cause critic is bidirectional
        
        loss_clip  = 0.5 * F.mse_loss(velocity_clip_critic,  noise_c - causal_latent_video)
        loss_audio = 0.5 * F.mse_loss(velocity_audio_critic, noise_a - causal_latent_audio)
        total_loss = loss_clip + loss_audio
    
        return {
            'clip_loss':  loss_clip,
            'audio_loss': loss_audio,
            'total_loss': total_loss,
        }
    
    def loss_ode_regression(self,
            teacher_clip_inputs: Tensor,   # [(b*d), n, c, h, w] where d is denoising steps. contains denoised iamges and not velocities
            teacher_clip_targets: Tensor,  # ^
            teacher_audio_inputs: Tensor,  # [(b*d), n, c, h, w]
            teacher_audio_targets: Tensor, # ^
            timesteps_start: Tensor,       # [(b*d), n, 1]
            timesteps_end: Tensor,         # [(b*d), n, 1]
            mouse: Tensor, # [(b*d) n 2]
            btn: Tensor,   # [(b*d) n 11]
            student_kv_cache:    KVCache,
            cfg_override: float = 0.0,
        ) -> dict[str, Tensor]:
        assert sum(x.numel() for x in self.student.parameters() if x.requires_grad) > 0, 'student must not be frozen'
        assert sum(x.numel() for x in self.teacher.parameters() if x.requires_grad) == 0, 'teacher must be frozen'

        dt = timesteps_start - timesteps_end # -- should all be 0.05. we denoise from high t to low t so it's start - end for a positive number
        cfg_weight = self.teacher_cfg_weight if cfg_override == 0.0 else cfg_override
        student_kv_cache.disable_cache_updates()

        # -- cant calculate the loss directly since the kv cache was warmed up with a different batch-size that
        # does not include the denoising steps. therefore we sliding-window over the batch-dimension where window_size=batch_size
        denoising_steps = teacher_clip_inputs.shape[1]
        clip_loss  = torch.tensor(0., device=teacher_clip_inputs.device)
        audio_loss = torch.tensor(0., device=teacher_clip_inputs.device)

        for d in range(denoising_steps):
            x_t_d   = teacher_clip_inputs [:, d, ::]
            t_d     = timesteps_start     [:, d, ::]
            mouse_d = mouse               [:, d, ::]
            btn_d   = btn                 [:, d, ::]
            audio_d = teacher_audio_inputs[:, d, ::]
            dt_d    = dt                  [0, d, 0].item()

            velocity_clip_student, velocity_audio_student = self.student.velocity_fn(x_t = x_t_d, t = t_d,
                                                                                    mouse = mouse_d, btn = btn_d, audio = audio_d,
                                                                                    kv_cache = student_kv_cache, cfg_weight = cfg_weight)
            # trying to minimize where the student would have taken us vs where the teacher ended up
            clip_loss  += 0.5 * F.mse_loss(teacher_clip_targets [:, d, ::], x_t_d   - (dt_d * velocity_clip_student))
            audio_loss += 0.5 * F.mse_loss(teacher_audio_targets[:, d, ::], audio_d - (dt_d * velocity_audio_student))

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
        self.max_grad_norm = self.train_cfg.max_grad_norm
        
        model_id = self.model_cfg.model_id

        student_cfg = deepcopy(self.model_cfg)
        teacher_cfg = deepcopy(self.model_cfg)

        student_cfg.causal = True
        teacher_cfg.causal = False
        # -- models. order of loading matters. 
        self.bidirectional_model: GameRFTAudio = get_model_cls(model_id)(teacher_cfg) ; self.load_bidirectional()
        self.causal_model:        GameRFTAudio = get_model_cls(model_id)(student_cfg) ; self.load_causal()
        self.critic_model:        GameRFTAudio = deepcopy(self.bidirectional_model)   ; self.load_critic()

        self.decoder: nn.Module         = get_decoder_only(self.train_cfg.vae_id,
                                                           self.train_cfg.vae_cfg_path,
                                                           self.train_cfg.vae_ckpt_path)
        self.decoder_fn                 = make_batched_decode_fn(self.decoder, self.train_cfg.vae_batch_size)

        self.audio_decoder: nn.Module   = get_decoder_only(self.train_cfg.audio_vae_id,
                                                           self.train_cfg.audio_vae_cfg_path,
                                                           self.train_cfg.audio_vae_ckpt_path)
        self.audio_decoder_fn           = make_batched_audio_decode_fn(self.audio_decoder, self.train_cfg.audio_vae_batch_size)

        unfreeze(self.causal_model)      ; unfreeze(self.critic_model) 
        freeze(self.bidirectional_model) ; freeze(self.decoder) ; freeze(self.audio_decoder)

        if self.rank == 0:
            n_params_causal = sum(p.numel() for p in self.causal_model.parameters())
            n_params_critic   = sum(p.numel() for p in self.critic_model.parameters())
            print(f"Total parameters: {(n_params_causal+n_params_critic):,}")
            print(f"\tCausal Model has {n_params_causal:,} parameters")
            print(f"\tCritic Score Model has {n_params_critic:,} parameters")

        # -- metrics & logging
        self.metrics                        = LogHelper()
        self.distill_step_counter           = 0
        self.ode_init_step_counter          = 0
        self.log_step_counter               = 0
        if self.rank == 0:        wandb.watch(self.get_module(), log = 'all')

        # -- make sure that eval frequency is a multiple of log frequency
        self.train_cfg.sample_interval = (self.train_cfg.log_interval * (self.train_cfg.sample_interval // self.train_cfg.log_interval))

        # -- data
        self.train_loader   = get_loader(self.train_cfg.data_id,
                                         self.train_cfg.batch_size,
                                         **self.train_cfg.data_kwargs)
        self.train_loader   = iter(self.train_loader)
        self.vae_scale      = self.train_cfg.vae_scale
        self.audio_scale    = self.train_cfg.audio_vae_scale
        self.latent_shape   = self.train_cfg.latent_shape
        self.audio_channels = self.model_cfg.audio_channels
        
        # -- dmd2-style loss (https://arxiv.org/pdf/2405.14867) 
        # - two time-scale update rule (section 4.2)
        # - TODO ensure whether the self-forcing paper has a GAN term (4.3), i don't think they do?
        # - multi-step generator, based on t-schedule (section 4.4) 
        # - i believe section 4.5 (reducing train-inference mismatch) is implicit in the autoregressive step of self-forcing as a whole
        self.update_ratio: int  = getattr(self.train_cfg, 'update_ratio', 25)
        # -- make sure critic learning rate is 5x less (corresponding to update ratio) than the causal (generator, in DMD) student
        self.causal_lr: float = self.train_cfg.opt_kwargs.lr
        self.critic_lr: float = self.causal_lr / self.update_ratio
        self.cfg_scale: float = getattr(self.train_cfg, 'cfg_scale',  1.3)  # cfg scale of the teacher
        self.denoising_steps  = self.train_cfg.n_steps
        self.ode_init_steps   = self.train_cfg.ode_init_steps
        self.loss_module      = Loss_SelfForcing(self.bidirectional_model,
                                                 self.critic_model,
                                                 self.causal_model,
                                                 teacher_cfg_weight=self.cfg_scale,
                                                 q_sample_fn=fwd_rectified_flow,
                                                 normalize=True)

        # -- hardware - done after loading models so it casts to bfloat16
        self.device = torch.device("cuda", self.local_rank)
        self.init_hardware()
        self.dtype = torch.bfloat16
        # -- optim tomfoolery, shenanigans, etc. - needs to be done after init_hardware so device casting calls work as intended
        self.ema                = EMA(self.causal_model, beta=0.999, update_after_step=200, update_every=1)
        self.opt_causal         = torch.optim.AdamW(self.causal_model.parameters(), **(dict(self.train_cfg.opt_kwargs) | {'lr': self.causal_lr}))
        self.opt_critic         = torch.optim.AdamW(self.critic_model.parameters(), **(dict(self.train_cfg.opt_kwargs) | {'lr': self.critic_lr}))
        self.scaler             = torch.amp.GradScaler(enabled = self.dtype != torch.bfloat16)
        self.ctx                = torch.amp.autocast(self.device.type, self.dtype)
        self.scheduler_causal   = get_scheduler_cls(self.train_cfg.scheduler)(self.opt_causal, **self.train_cfg.scheduler_kwargs)
        self.scheduler_critic   = get_scheduler_cls(self.train_cfg.scheduler)(self.opt_critic, **self.train_cfg.scheduler_kwargs)
        
        # -- sampler - initialize this after the hardware so it detects the device
        self.batch_size            = self.train_cfg.batch_size
        self.context_len           = self.model_cfg.context_length
        self.frame_gradient_cutoff = self.train_cfg.frame_gradient_cutoff
        self.num_gen_frames        = self.model_cfg.n_frames - self.context_len
        self.sampler               = SelfForcingSampler(
            causal_model=self.causal_model,
            bidirectional_model=self.bidirectional_model,
            model_config=self.model_cfg,
            batch_size=self.batch_size,
            latent_shape=self.train_cfg.latent_shape,
            denoising_steps=self.denoising_steps,
            context_len=self.context_len,
            num_gen_frames=self.num_gen_frames,
            frame_gradient_cutoff=self.frame_gradient_cutoff,
            training=True,
            autocast=self.dtype,
            cfg_scale=self.cfg_scale,
        )

        self.av_window_sampler = AVWindowSampler(
            n_steps=self.denoising_steps,
            cfg_scale=self.cfg_scale,
            window_length=self.context_len,
            num_frames=self.num_gen_frames,
            noise_prev=0.0,
            only_return_generated=False,
        )

        # -- ddp
        self.causal_model       = DistributedDataParallel(self.causal_model)     if self.world_size > 1 else self.causal_model
        self.critic_model       = DistributedDataParallel(self.critic_model) if self.world_size > 1 else self.critic_model

    def init_hardware(self):
        self.causal_model        = self.causal_model.to(self.device).train()
        self.critic_model        = self.critic_model.to(self.device).train()
        self.bidirectional_model = self.bidirectional_model\
                                                    .to(self.device).eval().bfloat16()
        self.loss_module         = self.loss_module .to(self.device).eval().bfloat16()

    def load_bidirectional(self):
        assert self.train_cfg.teacher_ckpt is not None
        self.bidirectional_model.core.load_state_dict(versatile_load(self.train_cfg.teacher_ckpt))

    def load_causal(self):
        if self.train_cfg.student_ckpt is None:
            print(f'Student checkpoint not specified, using bi-directional teacher weights.')
            # -- note don't use deepcopy just for debugging purposes,
            #  e.g. if config.causal=True vs False affects some architectural changes,
            #  this would catch them when the time comes, instead of introducing a sneaky bug
            self.causal_model.load_state_dict(self.bidirectional_model.state_dict())
            self.causal_model = remove_learned_abs_poc_enc(self.causal_model)
            return
        
        save_dict      = super().load_causal    (self.train_cfg.student_ckpt)
        self.causal_model       .load_state_dict(save_dict.get('causal_model') or save_dict['model'])
        self.ema                .load_state_dict(save_dict['ema'])
        self.opt_causal         .load_state_dict(save_dict['opt_causal'])
        self.opt_critic         .load_state_dict(save_dict['opt_critic'])
        self.scaler             .load_state_dict(save_dict['scaler'])
        self.scheduler_causal   .load_state_dict(save_dict['scheduler_causal'])
        self.scheduler_critic   .load_state_dict(save_dict['scheduler_critic'])
        self.decoder            .load_state_dict(save_dict['decoder'])
        self.distill_step_counter = save_dict['distillation_steps']
        self.ode_init_step_counter = save_dict['initialization_steps']
        self.log_step_counter   = save_dict['log_step_counter']

        self.causal_model = remove_learned_abs_poc_enc(self.causal_model)

    def load_critic(self):
        if self.train_cfg.critic_ckpt is None:
            print(f'critic checkpoint not specified, using bi-directional teacher weights.')
            self.critic_model.load_state_dict(self.bidirectional_model.state_dict())
            return
        
        save_dict = versatile_load(self.train_cfg.critic_ckpt)
        self.critic_model.load_state_dict(save_dict.get('critic_model'))
        return

    def save(self, suffix: str = ''):
        save_dict = {
            'causal_model':        self.causal_model       .state_dict(),
            'critic_model':        self.critic_model       .state_dict(),
            'bidirectional_model': self.bidirectional_model.state_dict(),
            'ema':                 self.ema                .state_dict(),
            'opt_causal':          self.opt_causal         .state_dict(),
            'opt_critic':          self.opt_critic         .state_dict(),
            'scaler':              self.scaler             .state_dict(),
            'decoder':             self.decoder            .state_dict(),
            'scheduler_causal':    self.scheduler_causal   .state_dict(),
            'scheduler_critic':    self.scheduler_critic   .state_dict(),
            'initialization_steps':self.ode_init_step_counter,
            'distillation_steps':  self.distill_step_counter,
         
            'log_step_counter':    self.log_step_counter,
        }
        os.makedirs(self.train_cfg.checkpoint_dir, exist_ok = True)
        fp = os.path.join(self.train_cfg.checkpoint_dir, f"step_{self.total_step_count}_{suffix}.pt")
        torch.save(save_dict, fp)


    def _format_batch(self) -> tuple[Tensor, ...]:
        try:
            batch: tuple[Tensor, ...] = next(self.train_loader)
            clip_bnchw, audio, mouse, btn = batch
            clip_bnchw /= self.vae_scale
            audio      /= self.audio_scale
            return (clip_bnchw  .to(device=self.device, dtype=self.dtype),
                    audio       .to(device=self.device, dtype=self.dtype),
                    mouse       .to(device=self.device, dtype=self.dtype),
                    btn         .to(device=self.device, dtype=self.dtype))
                
        except StopIteration:
            self.train_loader = iter(self.train_loader)
            return self._format_batch()


    @property
    def total_step_count(self):
        return self.ode_init_step_counter + self.distill_step_counter
    
    @property
    def should_log(self):
        return self.rank == 0 and self.total_step_count % self.train_cfg.log_interval == 0
    
    @property
    def should_sample(self):
        return self.rank == 0 and self.total_step_count % self.train_cfg.sample_interval == 0

    @property
    def should_save(self):
        return self.rank == 0 and self.total_step_count % self.train_cfg.save_interval == 0
    
    @property
    def should_distill(self):
        return self.distill_step_counter < self.train_cfg.ode_init_steps

    @property
    def should_ode_init(self):
        return self.ode_init_step_counter < self.ode_init_steps

    def get_module(self, ema = False): # TODO fix this shit
        if ema: return self.ema.ema_model if self.world_size == 1 else self.ema.ema_model
        else:   return self.causal_model  if self.world_size == 1 else self.causal_model

    @torch.no_grad()
    def evaluate(self, info: dict):
        try:
            sample_fn = partial(self.sampler.autoregressive_rollout,
                                clip_bnchw  = info['groundtruth_clip'],
                                audio_bcd   = info['groundtruth_audio'],
                                btn         = info['btn'],
                                mouse       = info['mouse'])

            self.causal_model       .eval()
            causal_samples = sample_fn(self.causal_model)
            causal_clip    = causal_samples['clean_latents_video']
            causal_audio   = causal_samples['clean_latents_audio']

            self.bidirectional_model.eval()
            bidirectional_samples = sample_fn(self.bidirectional_model)
            bidirectional_clip    = bidirectional_samples['clean_latents_video']
            bidirectional_audio   = bidirectional_samples['clean_latents_audio']

            self.critic_model       .eval()
            critic_samples = sample_fn(self.critic_model)
            critic_clip    = critic_samples['clean_latents_video']
            critic_audio   = critic_samples['clean_latents_audio']


            # -- get relevant samples
            groundtruth_clip  = info['groundtruth_clip']
            groundtruth_audio = info['groundtruth_audio']
            mouse, btn        = info['mouse'], info['btn']
            # -- build the dictionary
            n_frames = causal_clip.shape[1]
            log_dict = {
                'bidirectional_samples_video': None,
                'groundtruth_samples_video': groundtruth_clip,
                'student_samples_video': None,
                'critic_samples_video': None,
                'bidirectional_samples_audio': None,
                'groundtruth_samples_audio': groundtruth_audio,
                'student_samples_audio': None,
                'critic_samples_audio': None,
            }

            log_dict['student_samples_video']         = groundtruth_clip.clone()
            log_dict['bidirectional_samples_video']   = groundtruth_clip.clone()
            log_dict['critic_samples_video']          = groundtruth_clip.clone()

            log_dict['student_samples_video'][:, -n_frames:]       = causal_clip
            log_dict['bidirectional_samples_video'][:, -n_frames:] = bidirectional_clip
            log_dict['critic_samples_video'][:, -n_frames:]        = critic_clip

            log_dict['student_samples_video']       = self.decoder_fn(log_dict['student_samples_video'].bfloat16() * self.vae_scale)
            log_dict['bidirectional_samples_video'] = self.decoder_fn(log_dict['bidirectional_samples_video'].bfloat16() * self.vae_scale)
            log_dict['critic_samples_video']        = self.decoder_fn(log_dict['critic_samples_video'].bfloat16() * self.vae_scale)

            log_dict['bidirectional_samples_audio']   = groundtruth_audio.clone()
            log_dict['student_samples_audio']         = groundtruth_audio.clone()
            log_dict['critic_samples_audio']          = groundtruth_audio.clone()

            log_dict['student_samples_audio'][:, -n_frames:]       = causal_audio
            log_dict['bidirectional_samples_audio'][:, -n_frames:] = bidirectional_audio
            log_dict['critic_samples_audio'][:, -n_frames:]        = critic_audio

            log_dict['student_samples_audio']       = self.audio_decoder_fn(log_dict['student_samples_audio'].bfloat16() * self.audio_scale)
            log_dict['bidirectional_samples_audio'] = self.audio_decoder_fn(log_dict['bidirectional_samples_audio'].bfloat16() * self.audio_scale)
            log_dict['critic_samples_audio']        = self.audio_decoder_fn(log_dict['critic_samples_audio'].bfloat16() * self.audio_scale)

            # -- decode the groundtruth
            log_dict['groundtruth_samples_video'] = self.decoder_fn(groundtruth_clip.bfloat16() * self.vae_scale)
            log_dict['groundtruth_samples_audio'] = self.audio_decoder_fn(groundtruth_audio.bfloat16() * self.audio_scale)
            log_dict = {k: v.float() for k, v in log_dict.items()} # -- convert to float32 for wandb

            # -- to wandb:
            log_dict['student_samples_video']       = to_wandb_av(log_dict['student_samples_video'], log_dict['student_samples_audio'], mouse.float(), btn.float(), gather=False, max_samples=8, prefix='student_samples_')
            log_dict['bidirectional_samples_video'] = to_wandb_av(log_dict['bidirectional_samples_video'], log_dict['bidirectional_samples_audio'], mouse.float(), btn.float(), gather=False, max_samples=8, prefix='bidirectional_samples_')
            log_dict['critic_samples_video']        = to_wandb_av(log_dict['critic_samples_video'], log_dict['critic_samples_audio'], mouse.float(), btn.float(), gather=False, max_samples=8, prefix='critic_samples_')
            log_dict['groundtruth_samples_video']   = to_wandb_av(log_dict['groundtruth_samples_video'], log_dict['groundtruth_samples_audio'], mouse.float(), btn.float(), gather=False, max_samples=8, prefix='groundtruth_samples_')

            wandb.log({
                'student_samples': log_dict['student_samples_video'],
                'bidirectional_samples': log_dict['bidirectional_samples_video'],
                'critic_samples': log_dict['critic_samples_video'],
                'groundtruth_samples': log_dict['groundtruth_samples_video'],
            }, step=self.log_step_counter, commit=False)

        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"Evaluation failed: {e}")
        finally:
            self.causal_model.train()

    @torch.no_grad()
    def _log_step(self,
            info: dict[str, float | Tensor],
            maybe_evaluate: bool = True         # NOTE So that we can control only one of critic/causal to evaluate
        ):
        if self.should_sample and maybe_evaluate:
            self.evaluate(info)

        self.barrier() # -- evaluate can be time-consuming
        popped_metrics = self.metrics.pop()

        if self.should_log:
            wandb.log(
                popped_metrics,
                step=self.log_step_counter,
                commit=True)

    def _optimizer_step(self,
                        loss: Tensor,
                        optimizer: torch.optim.Optimizer,
                        model: GameRFTAudio,
                        clip_grad: bool = True) -> Tensor:
        self.scaler.scale(loss).backward()  ; self.scaler.unscale_(optimizer)
        grad_norm = clip_grad_norm_(model.parameters(), self.max_grad_norm) if clip_grad else get_total_norm(model.parameters())
        self.scaler.step(optimizer)         ; optimizer.zero_grad() ; self.scaler.update()
        return grad_norm

    def _train_step_self_forcing(self,
                    model: GameRFTAudio,
                    loss_fn: Callable[[Any], dict[str, Tensor]],
                    optim: torch.optim.Optimizer, 
                    clip_grad: bool = True,
                    debug_basic: bool = False):
    
        assert self.context_len + self.num_gen_frames == self.model_cfg.n_frames, \
            f'context_len + num_gen_frames must equal n_frames: {self.context_len=} + {self.num_gen_frames=} != {self.model_cfg.n_frames=}'
        
        optim.zero_grad(set_to_none=True)

        clip_bnchw, audio, mouse, btn = self._format_batch()
        if debug_basic:
            clip_bnchw = torch.zeros_like(clip_bnchw)
            audio = torch.zeros_like(audio)
            mouse = torch.zeros_like(mouse)
            btn = torch.zeros_like(btn)

        # -- generate the frames, while warming up the kv-cache
        rollout_info = self.sampler.autoregressive_rollout(
            model       = model,
            clip_bnchw  = clip_bnchw,
            audio_bcd   = audio,
            btn         = btn,
            mouse       = mouse)

        loss_info = loss_fn(causal_latent_video = rollout_info['clean_latents_video'],
                            causal_latent_audio = rollout_info['clean_latents_audio'],
                            t                   = rollout_info['selected_timesteps'],
                            mouse               = mouse,
                            btn                 = btn)

        grad_norm = self._optimizer_step(loss_info['total_loss'], optim, model, clip_grad=clip_grad)

        return {
            'groundtruth_clip':  clip_bnchw,
            'groundtruth_audio': audio,
            'clip':              rollout_info['clean_latents_video'],
            'audio':             rollout_info['clean_latents_audio'],
            'mouse':             mouse,
            'btn':               btn,
            'grad_norm':         grad_norm,       
            **loss_info
        }

    def _train_causal_step(self):
        return self._train_step_self_forcing(self.causal_model,
                                partial(self.loss_module.loss_distribution_matching_distillation,
                                        normalize=True,
                                        mask_frame_idx=1), # ignore the first frame
                                self.opt_causal,
                                debug_basic=False)

    def _train_critic_step(self):
        return self._train_step_self_forcing(self.critic_model, # -- but optim the critic 
                                self.loss_module.loss_rectified_flow,
                                self.opt_critic,
                                clip_grad=True,
                                debug_basic=False)

    def _train_step_ode_init(self) -> dict[str, Tensor]:
        clip_bnchw, audio, mouse, btn = self._format_batch()

        rollout_info = self.sampler.ode_initialization_rollout(
            teacher_model = self.bidirectional_model,
            clip_bnchw    = clip_bnchw,
            audio_bcd     = audio,
            btn           = btn,
            mouse         = mouse)
        
        num_denoising_steps   = rollout_info['trajectories_ts'].shape[1]
        # -- match pre- and post-denoising by an offset-by-1. also flatten into batch dim and re-add temporal dim
        teacher_clip_inputs   = rollout_info['trajectories_clip'] [:, :num_denoising_steps-1, ::]
        teacher_clip_targets  = rollout_info['trajectories_clip'] [:, 1:num_denoising_steps,   ::]
        teacher_audio_inputs  = rollout_info['trajectories_audio'][:,  :num_denoising_steps-1, ::]
        teacher_audio_targets = rollout_info['trajectories_audio'][:, 1:num_denoising_steps,   ::]
        timesteps_start       = rollout_info['trajectories_ts']   [:,  :num_denoising_steps-1, ::]
        timesteps_end         = rollout_info['trajectories_ts']   [:,  1:num_denoising_steps,  ::]

        # -- repeat the mouse and btn alongside the d dimension
        mouse_d = eo.repeat(mouse[:, self.context_len:, ::], 'b n x -> b d n x', d = (num_denoising_steps-1))
        btn_d   = eo.repeat(btn  [:, self.context_len:, ::], 'b n x -> b d n x', d = (num_denoising_steps-1))

        # -- warmup the student's kv cache on the clean context, so that the student & teacher are conditioned on the same frames
        student_kv_cache = KVCache(self.model_cfg, rank=self.device.index).to(device=self.device.type, rank=self.device.index)
        self.sampler._warmup_kv(self.causal_model, student_kv_cache,
                                clip_bnchw  [:, :self.context_len, ::],
                                audio       [:, :self.context_len, ::],
                                mouse       [:, :self.context_len, ::],
                                btn         [:, :self.context_len, ::])

        loss = self.loss_module.loss_ode_regression(
            teacher_clip_inputs   = teacher_clip_inputs,  # noisy frames 
            teacher_clip_targets  = teacher_clip_targets, # corresponding clean frames
            teacher_audio_inputs  = teacher_audio_inputs, 
            teacher_audio_targets = teacher_audio_targets, 
            timesteps_start       = timesteps_start, # timesteps of noisy frames
            timesteps_end         = timesteps_end, # timesteps of clean frames
            mouse                 = mouse_d,
            btn                   = btn_d,
            cfg_override          = 0.,
            student_kv_cache      = student_kv_cache
        )

        grad_norm = self._optimizer_step(loss['total_loss'], self.opt_causal, self.causal_model, clip_grad=True)

        return {**loss, 'grad_norm': grad_norm}


    def train(self):
        timer = Timer()

        # ---- ODE initialization (section 4 paragraph 1)
        unfreeze(self.causal_model)
        while self.should_ode_init:
            with self.ctx:
                info = self._train_step_ode_init()
            self.ode_init_step_counter += 1
            self.metrics.log_dict({
                'ode_init_loss':        info['total_loss'],
                'ode_init_audio_loss':  info['audio_loss'],
                'ode_init_clip_loss':   info['clip_loss'],
                'ode_init_grad_norm':   info['grad_norm'],
                'ode_init_lr':          self.scheduler_causal.get_last_lr()[0],
                'ode_init_time':        timer.hit(),
                'ode_init_step':        self.ode_init_step_counter
            })
            self._log_step(info, maybe_evaluate=False) ; self.log_step_counter += int(self.should_log)

        self.save(suffix='_ode_init_done')

        # ---- distillation training
        while self.should_distill:
            critic_time: list[float] = []
            info: list[dict]         = []
            with self.ctx:
                unfreeze(self.critic_model) ; freeze(self.causal_model)
                for _ in range(self.update_ratio):
                    info    += [self._train_critic_step()] ; critic_time += [timer.hit()]

                unfreeze(self.causal_model) ; freeze(self.critic_model)
                info        += [self._train_causal_step()] ; gen_time     = timer.hit()

                self.ema.update()
            
            *critic_info, causal_info = info

            self.metrics.log_dict({
                'causal_total_loss': causal_info['total_loss'],
                'causal_audio_loss': causal_info['audio_loss'],
                'causal_clip_loss':  causal_info['clip_loss'],
                'causal_grad_norm':  causal_info['grad_norm'],
                'causal_time':       gen_time,
                'causal_lr':         self.scheduler_causal.get_last_lr()[0],
                'distill_step':      self.distill_step_counter
            })

            self._log_step(causal_info, maybe_evaluate=True) ; self.log_step_counter += int(self.should_log)

            for step_info, _time in zip(critic_info, critic_time):
                self.metrics.log_dict({
                    'critic_total_loss': step_info['total_loss'],
                    'critic_audio_loss': step_info['audio_loss'],
                    'critic_clip_loss':  step_info['clip_loss'],
                    'critic_grad_norm':  step_info['grad_norm'],
                    'critic_lr':         self.scheduler_critic.get_last_lr()[0],
                    'critic_time':       _time,
                    'distill_step':      self.distill_step_counter
                })
                self._log_step(step_info, maybe_evaluate=False) ; self.log_step_counter += int(self.should_log)

            self.barrier()
            if self.should_save: self.save()
            print(f'Distillation step {self.distill_step_counter} - critic loss {step_info["total_loss"]} - causal loss {causal_info["total_loss"]}')
            self.distill_step_counter += 1

        self.save(suffix='_distill_done')

    def test_self_forcing_sampler(self, model: GameRFTAudio):
        with self.ctx:
            TOTAL_FRAMES        = []
            TOTAL_LATENT_FRAMES = []
            N = self.num_gen_frames
            (clip_bnchw, audio, mouse, btn) = self._format_batch()
            sampler = deepcopy(self.sampler)
            sampler.causal_model = model

            history_clip  = clip_bnchw  [:, :self.context_len]
            history_audio = audio       [:, :self.context_len]
            # -- we use history for these because av_sampler extends it by default
            history_mouse = mouse       [:, :self.context_len]
            history_btn   = btn         [:, :self.context_len]

            
            sampler._warmup_kv(history_clip, history_audio, history_mouse, history_btn)

            # -- emulate what av sampler does to get fake mouse data
            from owl_wms.utils import batch_permute_to_length

            fake_mouse, fake_btn = batch_permute_to_length(history_mouse, history_btn, N + self.context_len)

            # -- use the fake mouse data that would've been generated in the av_sampler anyways
            future_mouse = mouse       [:, self.context_len:]
            future_btn   = btn         [:, self.context_len:]

            info = sampler.old_autoregressive_rollout(clip_bnchw  = clip_bnchw,
                                                 audio_bcd   = audio,
                                                 btn         = btn,
                                                 mouse       = mouse) # audio only used for channel size


            print(f'{len(TOTAL_LATENT_FRAMES)=} {info['clean_latents_video'].shape=}')
            TOTAL_FRAMES = self.decoder_fn(info['clean_latents_video'] * self.vae_scale)
            return TOTAL_FRAMES

    def test_hail_mary_sampler(self, model: GameRFTAudio):
        with self.ctx:
            TOTAL_FRAMES        = []
            TOTAL_LATENT_FRAMES = []
            N = self.num_gen_frames
            (clip_bnchw, audio, mouse, btn) = self._format_batch()
            sampler = deepcopy(self.sampler)
            sampler.causal_model = model

            info = sampler.autoregressive_rollout(
                model       = self.causal_model,
                clip_bnchw  = clip_bnchw,
                audio_bcd   = audio,
                btn         = btn,
                mouse       = mouse)


            print(f'{len(TOTAL_LATENT_FRAMES)=} {info['clean_latents_video'].shape=}')
            TOTAL_FRAMES = self.decoder_fn(info['clean_latents_video'] * self.vae_scale)
            return TOTAL_FRAMES

    def test_av_window_sampler(self, model: GameRFTAudio):
        with self.ctx:
            av_window_sampler = self.av_window_sampler
            (clip_bnchw, audio, mouse, btn) = self._format_batch()

            history_clip  = clip_bnchw  [:] # for some reason the sampler expects full contextlen+num_frames_gen and it chops off the num_frames_gen itself?
            history_audio = audio       [:]
            # -- we use history for these because av_sampler extends it by default
            history_mouse = mouse       [:]
            history_btn   = btn         [:]

            frames, *_ = av_window_sampler.__call__(model.core,
                                                    dummy_batch     = history_clip,
                                                    audio           = history_audio,
                                                    mouse           = history_mouse,
                                                    btn             = history_btn,
                                                    audio_decode_fn = self.audio_decoder_fn,
                                                    decode_fn       = self.decoder_fn,
                                                    image_scale     = self.vae_scale, # already divided in format_batch
                                                    audio_scale     = self.audio_scale)
            
            print(f'{len(frames)=} {frames[0].shape=}')
            return frames