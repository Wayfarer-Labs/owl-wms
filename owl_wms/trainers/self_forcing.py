from __future__ import annotations
from ast import BoolOp
import random
from copy import deepcopy
from ema_pytorch import EMA
import torch ; from torch import Tensor
from typing import Callable, Optional, Any
import einops as eo
import torch.nn.functional as F
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_, get_total_norm
from torch.nn.parallel import DistributedDataParallel
from owl_wms.base import BaseTrainer
from functools import partial
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
            t                   = t                  [:, mask_frame_idx:]
            mouse               = mouse              [:, mask_frame_idx:]
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
                                                                            cfg_weight=self.critic_cfg_weight)
        
        loss_clip  = F.mse_loss(velocity_clip_critic,  noise_c - causal_latent_video)
        loss_audio = F.mse_loss(velocity_audio_critic, noise_a - causal_latent_audio)
        total_loss = loss_clip + loss_audio
    
        return {
            'clip_loss':  loss_clip,
            'audio_loss': loss_audio,
            'total_loss': total_loss,
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
        self.metrics            = LogHelper()
        self.total_step_counter = 0
        self.log_step_counter   = 0
        if self.rank == 0:        wandb.watch(self.get_module(), log = 'all')

        # -- make sure that eval frequency is a multiple of log frequency
        self.train_cfg.sample_interval = (self.train_cfg.log_interval * (self.train_cfg.sample_interval // self.train_cfg.log_interval))

        # -- data
        self.train_loader   = get_loader(self.train_cfg.data_id,
                                         self.train_cfg.batch_size,
                                         **self.train_cfg.data_kwargs)
        self.train_loader   = iter(self.train_loader)
        
        # -- dmd2-style loss (https://arxiv.org/pdf/2405.14867) 
        # - two time-scale update rule (section 4.2)
        # - TODO ensure whether the self-forcing paper has a GAN term (4.3), i don't think they do?
        # - multi-step generator, based on t-schedule (section 4.4) 
        # - i believe section 4.5 (reducing train-inference mismatch) is implicit in the autoregressive step of self-forcing as a whole
        self.update_ratio: int  = getattr(self.train_cfg, 'update_ratio', 5)
        # -- make sure critic learning rate is 5x less (corresponding to update ratio) than the causal (generator, in DMD) student
        self.causal_lr: float = self.train_cfg.opt_kwargs.lr
        self.critic_lr: float = self.causal_lr / self.update_ratio
        self.cfg_scale: float = getattr(self.train_cfg, 'cfg_scale',  1.3)  # cfg scale of the teacher
        self.t_schedule       = self.train_cfg.t_schedule
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
            model=self.causal_model,
            model_config=self.model_cfg,
            batch_size=self.batch_size,
            latent_shape=self.train_cfg.latent_shape,
            t_schedule=self.t_schedule,
            context_len=self.context_len,
            num_gen_frames=self.num_gen_frames,
            frame_gradient_cutoff=self.frame_gradient_cutoff,
            training=True,
            autocast=self.dtype
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
        self.total_step_counter = save_dict['steps']
        self.log_step_counter   = save_dict['log_step_counter']

    def load_critic(self):
        if self.train_cfg.critic_ckpt is None:
            print(f'critic checkpoint not specified, using bi-directional teacher weights.')
            self.critic_model.load_state_dict(self.bidirectional_model.state_dict())
            return
        
        save_dict = versatile_load(self.train_cfg.critic_ckpt)
        self.critic_model.load_state_dict(save_dict.get('critic_bidir_model'))
        return

    def save(self):
        save_dict = {
            'causal_model':        self.causal_model       .state_dict(),
            'bidirectional_model': self.bidirectional_model.state_dict(),
            'ema':                 self.ema                .state_dict(),
            'opt_causal':          self.opt_causal         .state_dict(),
            'opt_critic':          self.opt_critic         .state_dict(),
            'scaler':              self.scaler             .state_dict(),
            'decoder':             self.decoder            .state_dict(),
            'scheduler_causal':    self.scheduler_causal   .state_dict(),
            'scheduler_critic':    self.scheduler_critic   .state_dict(),
            'steps':               self.total_step_counter,
            'log_step_counter':    self.log_step_counter,
        }

        super().save(save_dict)

    def _format_batch(self) -> tuple[Tensor, ...]:
        try:
            batch: tuple[Tensor, ...] = next(self.train_loader)
            return tuple(item.to(self.device).float() for item in batch)
        except StopIteration:
            self.train_loader = iter(self.train_loader)
            return self._format_batch()

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

    def get_module(self, ema = False): # TODO fix this shit
        if ema: return self.ema.ema_model if self.world_size == 1 else self.ema.ema_model
        else:   return self.causal_model  if self.world_size == 1 else self.causal_model

    @torch.no_grad()
    def evaluate(self, info: dict):
        try:
            self.causal_model.eval()
            # -- get relevant samples
            student_clip      = info['clip']
            student_audio     = info['audio']
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
            # -- decode the groundtruth
            groundtruth_v = self.decoder_fn(groundtruth_clip.bfloat16())
            groundtruth_a = self.audio_decoder_fn(groundtruth_audio.bfloat16())

            wandb.log({
                'student_samples':     to_wandb_av(overlay_student, overlay_audio, mouse, btn, gather=False, max_samples=8, prefix='student_'),
                'groundtruth_samples': to_wandb_av(groundtruth_v,   groundtruth_a, mouse, btn, gather=False, max_samples=8, prefix='groundtruth_')
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
                commit=True,
            )


    def _optimizer_step(self,
                        loss: Tensor,
                        optimizer: torch.optim.Optimizer,
                        model: GameRFTAudio,
                        clip_grad: bool = True) -> Tensor:
        self.scaler.scale(loss).backward()  ; self.scaler.unscale_(optimizer)
        grad_norm = clip_grad_norm_(model.parameters(), self.max_grad_norm) if clip_grad else get_total_norm(model.parameters())
        self.scaler.step(optimizer)         ; optimizer.zero_grad() ; self.scaler.update()
        return grad_norm

    def _train_step(self,
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

        # -- warmup the KV cache with the context frames
        self.sampler._warmup_kv(clip_bnchw  [:, :self.context_len],
                                audio       [:, :self.context_len],
                                mouse       [:, :self.context_len],
                                btn         [:, :self.context_len])
        
        # -- generate the frames, with the warm cache
        rollout_info = self.sampler.autoregressive_rollout(btn     [:, -self.num_gen_frames:],
                                                           mouse   [:, -self.num_gen_frames:],
                                                           audio   [:, -self.num_gen_frames:])
        loss_info = loss_fn(causal_latent_video = rollout_info['clean_latents_video'],
                            causal_latent_audio = rollout_info['clean_latents_audio'],
                            t                   = rollout_info['selected_timesteps'],
                            mouse               = mouse       [:, -self.num_gen_frames:],
                            btn                 = btn         [:, -self.num_gen_frames:])

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
        return self._train_step(self.causal_model,
                                partial(self.loss_module.loss_distribution_matching_distillation,
                                        normalize=True,
                                        mask_frame_idx=1), # ignore the first frame
                                self.opt_causal,
                                debug_basic=False)

    def _train_critic_step(self):
        return self._train_step(self.critic_model,
                                self.loss_module.loss_rectified_flow,
                                self.opt_critic,
                                clip_grad=True,
                                debug_basic=False)

    def train(self):
        timer = Timer()
        while not self.should_break:
            critic_time: list[float] = []
            info: list[dict]         = []
            with self.ctx:
                unfreeze(self.causal_model) ; freeze(self.critic_model)
                info        += [self._train_causal_step()] ; gen_time     = timer.hit()
                unfreeze(self.critic_model) ; freeze(self.causal_model)
                for _ in range(self.update_ratio):
                    info    += [self._train_critic_step()] ; critic_time += [timer.hit()]

                self.ema.update()
            
            causal_info, *critic_info = info

            self.metrics.log_dict({
                'causal_total_loss': causal_info['total_loss'],
                'causal_audio_loss': causal_info['audio_loss'],
                'causal_clip_loss':  causal_info['clip_loss'],
                'causal_grad_norm':  causal_info['grad_norm'],
                'causal_time':       gen_time,
                'causal_lr':         self.scheduler_causal.get_last_lr()[0]
            })

            self._log_step(causal_info, maybe_evaluate=True) ; self.log_step_counter += int(self.should_log)

            for step_info, _time in zip(critic_info, critic_time):
                self.metrics.log_dict({
                    'critic_total_loss': step_info['total_loss'],
                    'critic_audio_loss': step_info['audio_loss'],
                    'critic_clip_loss':  step_info['clip_loss'],
                    'critic_grad_norm':  step_info['grad_norm'],
                    'critic_lr':         self.scheduler_critic.get_last_lr()[0],
                    'critic_time':       _time
                })
                self._log_step(step_info, maybe_evaluate=False) ; self.log_step_counter += int(self.should_log)

            self.barrier()
            if self.should_save: self.save()
            print(f'Step {self.total_step_counter} - critic loss {step_info["total_loss"]} - causal loss {causal_info["total_loss"]}')
            self.total_step_counter += 1

        self.save()
