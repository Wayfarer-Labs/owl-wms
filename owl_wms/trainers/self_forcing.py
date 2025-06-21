from __future__ import annotations
import random
from copy import deepcopy
from ema_pytorch import EMA
import torch ; from torch import Tensor
from typing import Callable, Optional
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
from .base import BaseTrainer
from ..models.gamerft_audio import GameRFTAudio
from ..models import get_model_cls
from ..utils import (
    freeze, unfreeze,
    SamiTimer as Timer, versatile_load
)
from ..data import get_loader
from ..utils.owl_vae_bridge import get_decoder_only, make_batched_decode_fn, make_batched_audio_decode_fn
from ..schedulers import get_scheduler_cls
from ..configs import TrainingConfig, TransformerConfig as ModelConfig, WANDBConfig as LoggingConfig
from torch.nn.parallel import DistributedDataParallel
import wandb
from ..utils.logging import LogHelper, to_wandb_av
from owl_wms.sampling.self_forcing_sampler import SelfForcingSampler, q_sample, sigma



class Loss_SelfForcing(nn.Module):

    def __init__(self,
            teacher_score_fn:   Callable[[Tensor, Tensor], tuple[Tensor, Tensor]], # TODO is this necessary
            student_score_fn:   Callable[[Tensor, Tensor], tuple[Tensor, Tensor]],
            critic_score_fn:    Callable[[Tensor, Tensor], tuple[Tensor, Tensor]],
            teacher_cfg_weight: float = 3.0,
            student_cfg_weight: float = 0.0,
            critic_cfg_weight:  float = 0.0, # TODO is this necessary
            q_sample_fn:        Callable[[Tensor, Tensor], tuple[Tensor, Tensor]] = q_sample,
            normalize:          bool = False,
            normalize_eps: float      = 1e-6,
        ):
        super().__init__()
        self.q_sample_fn = q_sample_fn
        self.student_score_fn   = student_score_fn
        self.critic_score_fn    = critic_score_fn
        self.teacher_score_fn   = teacher_score_fn
        self.teacher_score_fn   = torch.no_grad()(self.teacher_score_fn)

        # -- cfg
        self.teacher_cfg_weight = teacher_cfg_weight
        self.critic_cfg_weight  = critic_cfg_weight
        self.student_cfg_weight = student_cfg_weight
        # -- 
        self.normalize          = normalize
        self.normalize_eps      = normalize_eps

    def loss_distribution_matching_distillation(self,
            causal_latent_video: Tensor,  # [B, N, C, H, W]
            causal_latent_audio: Tensor,  # [B, N, C] 
            t:                   Tensor,  # [B, N] containing initial denoising timestep for its corresponding frame in scores
            mouse:               Tensor,  # [B, N, 2]
            btn:                 Tensor,  # [B, N, n_buttons]
            normalize:  Optional[bool] = None,
        ) -> dict[str, Tensor]:

        normalize               = normalize if normalize is not None else self.normalize

        noisy_causal_clip,  *_  = self.q_sample_fn(causal_latent_video, t)
        noisy_causal_audio, *_  = self.q_sample_fn(causal_latent_audio, t)
        
        # -- critic score predicts on student noisy data
        score_clip_critic, score_audio_critic   = self.critic_score_fn(noisy_causal_clip, t,
                                                                        mouse, btn, noisy_causal_audio,
                                                                        cfg_weight=self.critic_cfg_weight)
        # -- teacher predicts on groundtruth noisy data
        score_clip_teacher, score_audio_teacher = self.teacher_score_fn(noisy_causal_clip, t,
                                                                        mouse, btn, noisy_causal_audio,
                                                                        cfg_weight=self.teacher_cfg_weight)
        grad_clip: Tensor  = score_clip_critic  - score_clip_teacher
        grad_audio: Tensor = score_audio_critic - score_audio_teacher

        if normalize:
            # normalize by magnitude of real prediction
            p_gt_clip   = causal_latent_video - score_clip_teacher.detach()
            p_gt_audio  = causal_latent_audio - score_audio_teacher.detach()
            # --
            normalizer_clip  = torch.abs(p_gt_clip) .mean(dim=[1,2,3,4], keepdim=True)
            normalizer_audio = torch.abs(p_gt_audio).mean(dim=[1,2],     keepdim=True)
            grad_clip .div_(normalizer_clip  + self.normalize_eps).nan_to_num_()
            grad_audio.div_(normalizer_audio + self.normalize_eps).nan_to_num_()

        clip_loss  = 0.5 * (grad_clip  ** 2).mean()
        audio_loss = 0.5 * (grad_audio ** 2).mean()

        return {
            'clip_loss':  clip_loss,
            'audio_loss': audio_loss,
            'total_loss': clip_loss + audio_loss,
        }

    def _flow(self,
            causal:       Tensor,  # in SF: [B, C, H, W], here, maybe [B, N, C, H, W] ?
            noisy_causal: Tensor,  # in SF: [B, C, H, W],
            sigma_:       Tensor,  # What should this be
            full_precision: bool = True
        ) -> Tensor:
        prev_dtype = causal.dtype
        calc_dtype = torch.double if full_precision else causal.dtype
        # -- cast to higher precision
        causal      .to(dtype=calc_dtype)
        noisy_causal.to(dtype=calc_dtype) 
        sigma_      .to(dtype=calc_dtype)

        return (noisy_causal - causal).div_(sigma_).to(dtype=prev_dtype)

    def loss_flow_prediction(self,         
            causal_latent_video: Tensor, # [B, N, C, H, W]
            causal_latent_audio: Tensor, # [B, N, C] 
            t:                   Tensor, # [B, N] containing initial denoising timestep for its corresponding frame in scores
            mouse:               Tensor, # [B, N, 2]
            btn:                 Tensor, # [B, N, n_buttons]
        ) -> dict[str, Tensor]:
        # see https://github.com/guandeh17/Self-Forcing/blob/a93f2f80ce60f4b022b0340d0026fca24d4f72a2/model/dmd.py#L237-L333
        noisy_causal_clip,  noise_c, *_, sigma_c = self.q_sample_fn(causal_latent_video, t)
        noisy_causal_audio, noise_a, *_, sigma_a = self.q_sample_fn(causal_latent_audio, t)

        # TODO SAMI - I don't understand this, but it seems central to flow loss, so it might be a Claude question:
        #               we are trying to match the velocity field of the *student* by using this as a target?
        # 
        #               My understanding is that this simply moves both the student and critic closer together.
        #               Therefore, if the critic is closer to the teacher, then we want to update it more frequently
        #                 (hence the update_ratio is 5:1 critic:causal)
        target_flow_clip  = noise_c - causal_latent_video
        target_flow_audio = noise_a - causal_latent_audio

        score_clip_critic, score_audio_critic = self.critic_score_fn(noisy_causal_clip, t,
                                                   mouse, btn, noisy_causal_audio,
                                                   cfg_weight=self.critic_cfg_weight)

        flow_clip  = self._flow(score_clip_critic,  noisy_causal_clip,  sigma_c, full_precision=True)
        flow_audio = self._flow(score_audio_critic, noisy_causal_audio, sigma_a, full_precision=True)

        return {
            'clip_loss':  (loss_clip  := torch.mean((flow_clip  - target_flow_clip)  ** 2)),
            'audio_loss': (loss_audio := torch.mean((flow_audio - target_flow_audio) ** 2)),
            'total_loss': loss_clip + loss_audio,
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
        self.critic_model:    GameRFTAudio = deepcopy(self.bidirectional_model)   ; self.load_critic()

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
        
        # -- loss & more
        self.update_ratio: int  = getattr(self.train_cfg, 'update_ratio', 5)  # how many steps to update the critic bidir model before running one causal model update
        self.cfg_scale: float   = getattr(self.train_cfg, 'cfg_scale',  1.3)  # cfg scale of the teacher
        self.t_schedule         = self.train_cfg.t_schedule
        self.loss_module        = Loss_SelfForcing(self.bidirectional_model.score_fn,
                                                   self.causal_model.score_fn,
                                                   self.critic_model.score_fn,
                                                   teacher_cfg_weight=self.cfg_scale,
                                                   normalize=True)

        # -- hardware - done after loading models so it casts to bfloat16
        self.device = torch.device("cuda", self.local_rank)
        self.init_hardware()

        # -- optim tomfoolery, shenanigans, etc. - needs to be done after init_hardware so device casting calls work as intended
        self.ema                = EMA(self.causal_model, beta=0.999, update_after_step=0, update_every=1)
        self.opt_causal         = torch.optim.AdamW(self.causal_model.parameters(), **self.train_cfg.opt_kwargs)
        self.opt_critic         = torch.optim.AdamW(self.critic_model.parameters(), **self.train_cfg.opt_kwargs)
        self.scaler             = torch.amp.GradScaler()
        self.ctx                = torch.amp.autocast(self.device.type, torch.float32)
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
            autocast=torch.bfloat16
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

    def _format_batch(self) -> tuple[Tensor, Tensor, Tensor, Tensor]:
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
            student_clip      = info['causal_clip']
            student_audio     = info['causal_audio']
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

    def _construct_primers(self,
                           clip_bnchw: Tensor,
                           mouse: Tensor,
                           btn: Tensor,
                           audio: Tensor) -> list[dict[str, Tensor]]:

        return [ dict(latent = clip_bnchw[:, i:i+1],
                       mouse = mouse     [:, i:i+1],
                       btn   = btn       [:, i:i+1],
                       audio = audio     [:, i:i+1]) for i in range(btn.shape[1]) ]

    def _optimizer_step(self,
                        loss: Tensor,
                        optimizer: torch.optim.Optimizer,
                        model: GameRFTAudio) -> None:
        self.scaler.scale(loss).backward()  ; self.scaler.unscale_(optimizer)
        grad_norm = clip_grad_norm_(model.parameters(), self.max_grad_norm)
        self.scaler.step(optimizer)         ; optimizer.zero_grad() ; self.scaler.update()
        return grad_norm


    def _train_causal_step(self):
        self.opt_causal.zero_grad(set_to_none=True)

        clip_bnchw, audio, mouse, btn = self._format_batch()
        latent_conditions             = self._construct_primers(clip_bnchw, mouse, btn, audio)[   :self.num_gen_frames] # 30, 60
        rollout_info                  = self.sampler.autoregressive_rollout(btn               [:, -self.num_gen_frames:],
                                                                            mouse             [:, -self.num_gen_frames:],
                                                                            audio             [:, -self.num_gen_frames:],
                                                                            latent_conditions)
        
        loss_info = self.loss_module.loss_distribution_matching_distillation(
            causal_latent_video = rollout_info['clean_latents_video'],
            causal_latent_audio = rollout_info['clean_latents_audio'],
            t                   = rollout_info['selected_timesteps'],
            mouse               = mouse     [:, -self.num_gen_frames:],
            btn                 = btn       [:, -self.num_gen_frames:],
        )

        grad_norm = self._optimizer_step(loss_info['total_loss'], self.opt_causal, self.causal_model)

        return {
            'groundtruth_clip':  clip_bnchw,
            'groundtruth_audio': audio,
            'causal_clip':       rollout_info['clean_latents_video'],
            'causal_audio':      rollout_info['clean_latents_audio'],
            'mouse':             mouse,
            'btn':               btn,
            'grad_norm':         grad_norm,
            **loss_info,
        }

    def _train_critic_step(self):
        self.opt_critic.zero_grad(set_to_none=True)

        clip_bnchw, audio, mouse, btn   = self._format_batch()
        latent_conditions               = self._construct_primers(clip_bnchw, mouse, btn, audio)[   :self.num_gen_frames] # 30, 60
        rollout_info                    = self.sampler.autoregressive_rollout(btn               [:, -self.num_gen_frames:],
                                                                             mouse              [:, -self.num_gen_frames:],
                                                                             audio              [:, -self.num_gen_frames:],
                                                                             latent_conditions)

        # -- technically, the loss fwd would only need the generated frames, timesteps, and conditionals
        # -- not sure if this abstraction fits the mental model of other researchers but it makes sense to me?
        loss_info = self.loss_module.loss_flow_prediction(
            causal_latent_video = rollout_info['clean_latents_video'],
            causal_latent_audio = rollout_info['clean_latents_audio'],
            t                   = rollout_info['selected_timesteps'],
            mouse               = mouse       [:, -self.num_gen_frames:],
            btn                 = btn         [:, -self.num_gen_frames:],
        )

        grad_norm = self._optimizer_step(loss_info['total_loss'], self.opt_critic, self.critic_model)

        return {
            'groundtruth_clip':  clip_bnchw,
            'groundtruth_audio': audio,
            'critic_clip':       rollout_info['clean_latents_video'],
            'critic_audio':      rollout_info['clean_latents_audio'],
            'mouse':             mouse,
            'btn':               btn,
            'grad_norm':         grad_norm,
            **loss_info,
        }

    def train(self):
        timer = Timer()
        while not self.should_break:
            critic_time: list[float] = []
            info: list[dict]         = []
            with self.ctx:
                info        += [self._train_causal_step()] ; gen_time     = timer.hit()
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
