"""
TODO this doesnt work right now
"""

import torch
from torch import nn
import torch.nn.functional as F
import einops as eo
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

import wandb
from ema_pytorch import EMA

from ..data import get_loader
from ..models import get_model_cls
from ..muon import init_muon
from ..utils import Timer, freeze, unfreeze, versatile_load
from ..utils.logging import LogHelper, to_wandb, to_wandb_av
from .base import BaseTrainer
from ..configs import Config
from ..sampling import get_sampler_cls
from ..sampling.schedulers import get_sd3_euler
from ..utils.owl_vae_bridge import get_decoder_only, make_batched_decode_fn, make_batched_audio_decode_fn

class DistillODETrainer(BaseTrainer):
    """
    Trainer for distilling a teacher WM into some student WM.
    New config values needed:
    - teacher_ckpt
    - teacher_cfg
    - cfg_scale (when getting teacher trajectories for distillation)
    - subsample (how many solution pairs to actually regress on)
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Load the teacher model
        teacher_ckpt_path = self.train_cfg.teacher_ckpt
        teacher_cfg_path = self.train_cfg.teacher_cfg
        teacher_ckpt = versatile_load(teacher_ckpt_path)
        teacher_cfg = Config.from_yaml(teacher_cfg_path).model
        teacher = get_model_cls(teacher_cfg.model_id)(teacher_cfg)
        teacher.load_state_dict(teacher_ckpt)
        self.teacher = teacher.core.to(self.device).bfloat16().eval()
    
        # Instantiate student
        student_cfg = self.model_cfg
        self.student = get_model_cls(student_cfg.model_id)(student_cfg)
        self.load_teacher_into_student(self.student, teacher_ckpt, teacher_cfg, student_cfg)
        self.student = self.student.core.to(self.device).train()

        # Model size and init everything else
        if self.rank == 0:
            model_params = sum(p.numel() for p in self.student.parameters())
            print(f"Student model parameters: {model_params:,}")

        self.opt = None
        self.scaler = None
        self.total_step_counter = 0
        self.ema = None

        self.decoder = get_decoder_only(
            self.train_cfg.vae_id,
            self.train_cfg.vae_cfg_path,
            self.train_cfg.vae_ckpt_path
        )

        self.audio_decoder = get_decoder_only(
            self.train_cfg.audio_vae_id,
            self.train_cfg.audio_vae_cfg_path,
            self.train_cfg.audio_vae_ckpt_path
        )

        freeze(self.decoder)
        freeze(self.audio_decoder)
        self.decoder = self.decoder.bfloat16().to(self.device).eval()
        self.audio_decoder = self.audio_decoder.bfloat16().to(self.device).eval()

    def load_teacher_into_student(self, student_model, teacher_state_dict, teacher_cfg, student_cfg):
        """
        Improved weight-transfer from teacher → student.

        1. Handles 'module.' / 'core.' prefixes in the teacher checkpoint.
        2. Copies projection / embedding / final layers verbatim.
        3. Copies a subset of transformer blocks such that the first and last
           blocks are retained and the intermediate ones are chosen with
           uniform spacing.
        4. Works for both DiffusionDecoder (wrapper) and DiffusionDecoderCore.
        """

        # ── 0. get the real core we want to load into ────────────────────────────
        student_core = student_model.core if hasattr(student_model, "core") else student_model

        # ── 1. clean the teacher state-dict prefixes ────────────────────────────
        clean_teacher = {}
        for k, v in teacher_state_dict.items():
            if k.startswith("module."):
                k = k[len("module."):]
            if k.startswith("core."):
                k = k[len("core."):]
            clean_teacher[k] = v

        # ── 2. determine transformer depth of teacher & student ────────────────
        n_teacher = getattr(teacher_cfg, "n_layers", None)
        n_student = getattr(student_cfg, "n_layers", None)
        if n_teacher is None or n_student is None:
            raise ValueError("Both cfgs must expose `n_layers`")

        # Map indices: always keep first & last, interpolate the rest
        if n_student == 1:
            index_map = {0: 0}
        else:
            t_ids = [
                round(i * (n_teacher - 1) / (n_student - 1))
                for i in range(n_student)
            ]
            index_map = {s_idx: t_idx for s_idx, t_idx in enumerate(t_ids)}

        new_state_dict = {}

        # ── 3. copy projection / embedding / final layers verbatim ─────────────
        for k, v in clean_teacher.items():
            if not k.startswith("transformer.blocks."):
                new_state_dict[k] = v

        # ── 4. copy the selected transformer blocks ────────────────────────────
        for s_idx, t_idx in index_map.items():
            t_pref = f"transformer.blocks.{t_idx}."
            s_pref = f"transformer.blocks.{s_idx}."
            for k, v in clean_teacher.items():
                if k.startswith(t_pref):
                    new_state_dict[s_pref + k[len(t_pref):]] = v

        # ── 5. load into the student core ──────────────────────────────────────
        student_core.load_state_dict(new_state_dict, strict=True)

    def get_ema_core(self):
        if self.world_size > 1:
            return self.ema.ema_model.module
        return self.ema.ema_model

    def save(self):
        save_dict = {
            'model': self.student.state_dict(),
            'ema': self.ema.state_dict(),
            'opt': self.opt.state_dict(),
            'scaler': self.scaler.state_dict(),
            'steps': self.total_step_counter
        }
        super().save(save_dict)

    def load(self):
        resume_ckpt = getattr(self.train_cfg, 'resume_ckpt', None)
        if resume_ckpt is None:
            return
        save_dict = super().load(resume_ckpt)
        self.student.load_state_dict(save_dict['model'], strict=False)
        self.ema.load_state_dict(save_dict['ema'])
        self.opt.load_state_dict(save_dict['opt'])
        self.scaler.load_state_dict(save_dict['scaler'])
        self.total_step_counter = save_dict['steps']

    def get_ema_core(self):
        if self.world_size > 1:
            return self.ema.ema_model.module
        return self.ema.ema_model

    def train(self):
        torch.cuda.set_device(self.local_rank)

        self.cfg_scale = self.train_cfg.cfg_scale
        self.subsample = self.train_cfg.subsample

        # Prepare model, ema
        self.student = self.student.to(self.device).train()
        if self.world_size > 1:
            self.student = DDP(self.student)

        #self.teacher = torch.compile(self.teacher, mode='max-autotune', dynamic=False, fullgraph=True)

        self.ema = EMA(
            self.student,
            beta=0.999,
            update_after_step=0,
            update_every=1
        )

        # Optimizer
        if self.train_cfg.opt.lower() == "muon":
            self.opt = init_muon(self.student, rank=self.rank, world_size=self.world_size, **self.train_cfg.opt_kwargs)
        else:
            opt_cls = getattr(torch.optim, self.train_cfg.opt)
            self.opt = opt_cls(self.student.parameters(), **self.train_cfg.opt_kwargs)

        self.scaler = torch.amp.GradScaler()
        self.ctx = torch.amp.autocast(self.device, torch.bfloat16)

        self.accum_steps = self.train_cfg.target_batch_size // self.train_cfg.batch_size // self.world_size
        self.accum_steps = max(1, self.accum_steps)

        decode_fn = make_batched_decode_fn(self.decoder, self.train_cfg.vae_batch_size)
        audio_decode_fn = make_batched_audio_decode_fn(self.audio_decoder, self.train_cfg.vae_batch_size)

        # Dataset setup
        loader = get_loader(self.train_cfg.data_id, self.train_cfg.batch_size, **self.train_cfg.data_kwargs)
        sample_loader = get_loader(self.train_cfg.sample_data_id, self.train_cfg.n_samples, **self.train_cfg.sample_data_kwargs)
        sample_loader = iter(sample_loader)
        sampler = get_sampler_cls(self.train_cfg.sampler_id)(**self.train_cfg.sampler_kwargs)

        # Timer and logging
        timer = Timer()
        timer.reset()
        metrics = LogHelper()
        if self.rank == 0:
            wandb.watch(self.student, log='all')

        self.load()

        local_step = 0
        input_shape = (
            self.model_cfg.channels,
            self.model_cfg.sample_size,
            self.model_cfg.sample_size,
        )
        input_shape_audio = (
            self.model_cfg.audio_channels,
        )

        def get_dummy(z):
            return (
                torch.randn(z.shape[0],z.shape[1],*input_shape, device=z.device, dtype=z.dtype),
                torch.randn(z.shape[0],z.shape[1], *input_shape_audio, device=z.device, dtype=z.dtype)
            )

        @torch.no_grad()
        def sample_with_teacher(mouse, btn, n_steps=20, subsample=1.0):
            """
            Sample using teacher 
            """
            noisy_video, noisy_audio = get_dummy(mouse)

            dt_list = get_sd3_euler(n_steps)
            t = torch.ones(mouse.shape[0],mouse.shape[1], device = mouse.device, dtype = mouse.dtype)

            video_inputs = []
            audio_inputs = []
            mouse_inputs = []
            btn_inputs = []
            ts = []
            video_outputs = []
            audio_outputs = []

            cond_mask = torch.ones(mouse.shape[0], device = mouse.device, dtype = torch.bool)
            uncond_mask = torch.zeros_like(cond_mask)

            for dt in dt_list:
                pred_video_uncond, pred_audio_uncond = self.teacher(noisy_video, noisy_audio, t, mouse, btn, has_controls=uncond_mask)
                pred_video_cond, pred_audio_cond = self.teacher(noisy_video, noisy_audio, t, mouse, btn, has_controls=cond_mask)

                pred_video = pred_video_uncond + self.cfg_scale * (pred_video_cond - pred_video_uncond)
                pred_audio = pred_audio_uncond + self.cfg_scale * (pred_audio_cond - pred_audio_uncond)

                video_inputs.append(noisy_video.clone())
                audio_inputs.append(noisy_audio.clone())
                video_outputs.append(pred_video.clone())
                audio_outputs.append(pred_audio.clone())
                mouse_inputs.append(mouse.clone())
                btn_inputs.append(btn.clone())
                ts.append(t.clone())

                noisy_video = noisy_video - dt * pred_video
                noisy_audio = noisy_audio - dt * pred_audio
                t = t - dt
            
            # Concatenate on batch dim
            video_inputs = torch.cat(video_inputs, dim=0)
            audio_inputs = torch.cat(audio_inputs, dim=0)
            video_outputs = torch.cat(video_outputs, dim=0)
            audio_outputs = torch.cat(audio_outputs, dim=0)
            mouse_inputs = torch.cat(mouse_inputs, dim=0)
            btn_inputs = torch.cat(btn_inputs, dim=0)
            ts = torch.cat(ts, dim=0)

            if subsample < 1.0:
                inds = torch.randperm(video_inputs.shape[0])[:int(video_inputs.shape[0] * subsample)]
                video_inputs = video_inputs[inds]
                audio_inputs = audio_inputs[inds]
                video_outputs = video_outputs[inds]
                audio_outputs = audio_outputs[inds]
                mouse_inputs = mouse_inputs[inds]
                btn_inputs = btn_inputs[inds]
                ts = ts[inds]

            return video_inputs, audio_inputs, video_outputs, audio_outputs, mouse_inputs, btn_inputs, ts

        for (batch_vid, batch_audio, batch_mouse, batch_btn) in loader:
            batch_vid = batch_vid.to(self.device).bfloat16() / self.train_cfg.vae_scale
            batch_audio = batch_audio.to(self.device).bfloat16() / self.train_cfg.audio_vae_scale
            batch_mouse = batch_mouse.to(self.device).bfloat16()
            batch_btn = batch_btn.to(self.device).bfloat16()
            
            with self.ctx:
                (
                    video_inputs,
                    audio_inputs,
                    video_outputs,
                    audio_outputs,
                    mouse_inputs,
                    btn_inputs,
                    ts
                ) = sample_with_teacher(batch_mouse, batch_btn, subsample=self.subsample)

                preds_video, preds_audio = self.student(video_inputs, audio_inputs, ts, mouse_inputs, btn_inputs)
                loss_video = F.mse_loss(preds_video, video_outputs) / self.accum_steps
                loss_audio = F.mse_loss(preds_audio, audio_outputs) / self.accum_steps
                loss = 0.5 * (loss_video + loss_audio)

            metrics.log('loss', loss)
            self.scaler.scale(loss).backward()

            local_step += 1
            if local_step % self.accum_steps == 0:
                # Updates
                self.scaler.unscale_(self.opt)
                torch.nn.utils.clip_grad_norm_(self.student.parameters(), max_norm=10.0)

                self.scaler.step(self.opt)
                self.opt.zero_grad(set_to_none=True)
                self.scaler.update()
                self.ema.update()

                # Logging and sampling
                with torch.no_grad():
                    wandb_dict = metrics.pop()
                    wandb_dict['time'] = timer.hit()
                    wandb_dict['lr'] = self.opt.param_groups[0]['lr']
                    timer.reset()

                    if self.total_step_counter % self.train_cfg.sample_interval == 0:
                        with self.ctx:
                            vid_for_sample, aud_for_sample, mouse_for_sample, btn_for_sample = next(sample_loader)
                            n_samples = self.train_cfg.n_samples
                            samples, audio, sample_mouse, sample_button = sampler(
                                self.get_ema_core(),
                                vid_for_sample.bfloat16().cuda() / self.train_cfg.vae_scale,
                                aud_for_sample.bfloat16().cuda() / self.train_cfg.audio_vae_scale,
                                mouse_for_sample.bfloat16().cuda(),
                                btn_for_sample.bfloat16().cuda(),
                                decode_fn,
                                audio_decode_fn,
                                self.train_cfg.vae_scale,
                                self.train_cfg.audio_vae_scale
                            ) # -> [b,n,c,h,w]
                        if self.rank == 0:
                            wandb_av_out = to_wandb_av(samples, audio, sample_mouse, sample_button)
                            if len(wandb_av_out) == 3:  
                                video, depth_gif, flow_gif = wandb_av_out
                                wandb_dict['samples'] = video
                                wandb_dict['depth_gif'] = depth_gif
                                wandb_dict['flow_gif'] = flow_gif
                            else:
                                video = wandb_av_out
                                wandb_dict['samples'] = video

                    if self.rank == 0:
                        wandb.log(wandb_dict)

                self.total_step_counter += 1
                if self.total_step_counter % self.train_cfg.save_interval == 0:
                    if self.rank == 0:
                        self.save()

                self.barrier()
