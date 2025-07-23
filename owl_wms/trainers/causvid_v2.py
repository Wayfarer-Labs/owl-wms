import torch
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

from .base import BaseTrainer
from ..configs import Config
from ..utils import versatile_load, freeze, unfreeze, Timer
from ..models import get_model_cls
from ..utils.owl_vae_bridge import get_decoder_only, make_batched_decode_fn, make_batched_audio_decode_fn
from ..utils import batch_permute_to_length
from ..nn.kv_cache import KVCache
from ..utils.logging import LogHelper, to_wandb_av
from ..data import get_loader
from ..sampling import get_sampler_cls

from copy import deepcopy
from ema_pytorch import EMA
import einops as eo
from contextlib import nullcontext
import random
import wandb

# === ROLLOUTS ===

class RolloutManager:
    def __init__(self, model_cfg, min_rollout_frames=8, rollout_steps = 1):
        self.model_cfg = model_cfg
        self.min_rollout_frames = min_rollout_frames
        self.rollout_steps = rollout_steps
        self.noise_prev = 0.2
    
    def get_rollouts(
        self, 
        model,
        video_bnchw,
        audio_bnc,
        mouse_bnd,
        btn_bnd
    ):

        with torch.no_grad():
            # Select half of frames from each video
            # Mask is true = we are generating that frame with student
            n_frames = video_bnchw.shape[1]
            half = n_frames // 2
            mask = torch.zeros((video_bnchw.shape[0], n_frames), dtype=torch.bool)
            mask[:, :half] = True
            mask = mask.to(device = video_bnchw.device, dtype = torch.bool)

            mask_video = mask[:,:,None,None,None]
            mask_audio = mask[:,:,None]

            video_eps = torch.randn_like(video_bnchw)
            audio_eps = torch.randn_like(audio_bnc)

            noisy_video_bnchw = torch.where(
                mask_video,
                video_eps,
                video_bnchw * (1. - self.noise_prev) + self.noise_prev * video_eps
            )

            noisy_audio_bnc = torch.where(
                mask_audio,
                audio_eps,
                audio_bnc * (1. - self.noise_prev) + self.noise_prev * audio_eps
            )

            ts = torch.ones_like(video_bnchw[:,:,0,0,0])
            ts = torch.where(
                mask,
                ts,
                ts * self.noise_prev
            )
        
        vid_pred_bnchw, aud_pred_bnc = model(
            noisy_video_bnchw,
            noisy_audio_bnc,
            ts,
            mouse_bnd,
            btn_bnd
        )

        video_bnchw = torch.where(
            mask_video,
            noisy_video_bnchw - vid_pred_bnchw,
            video_bnchw
        )

        audio_bnc = torch.where(
            mask_audio,
            noisy_audio_bnc - aud_pred_bnc,
            audio_bnc
        )

        return video_bnchw, audio_bnc, mouse_bnd, btn_bnd, mask

# === LOSSES ===

def get_critic_loss(
    student, critic,
    video_bnchw, audio_bnc,
    mouse_bnd, btn_bnd,
    rollout_manager
):
    # Get rollout
    with torch.no_grad():
        video_bnchw, audio_bnc, mouse_bnd, btn_bnd, _ = rollout_manager.get_rollouts(
            model = student,
            video_bnchw = video_bnchw,
            audio_bnc = audio_bnc,
            mouse_bnd = mouse_bnd,
            btn_bnd = btn_bnd
        )

        # Get ts ~ U(0.02, 0.98)
        ts_bn = torch.rand(video_bnchw.shape[0], video_bnchw.shape[1]) * 0.96 + 0.02
        ts_bn = ts_bn.to(video_bnchw.device, video_bnchw.dtype)
        ts_bn_exp = ts_bn[:,:,None,None,None]
        ts_bn_exp_audio = ts_bn[:,:,None]

        # Noise student outputs and get target
        z_vid_bnchw = torch.randn_like(video_bnchw)
        z_aud_bnc = torch.randn_like(audio_bnc)

        noisy_vid_bnchw = video_bnchw * (1. - ts_bn_exp) + z_vid_bnchw * ts_bn_exp
        noisy_aud_bnc = audio_bnc * (1. - ts_bn_exp_audio) + z_aud_bnc * ts_bn_exp_audio

        target_vid = (z_vid_bnchw - video_bnchw)
        target_aud = (z_aud_bnc - audio_bnc)

    pred_vid, pred_aud = critic(
        noisy_vid_bnchw,
        noisy_aud_bnc,
        ts_bn,
        mouse_bnd,
        btn_bnd
    )

    vid_loss = F.mse_loss(pred_vid, target_vid)
    aud_loss = F.mse_loss(pred_aud, target_aud)

    return (vid_loss + aud_loss) * 0.5

def get_dmd_loss(
    student, critic, teacher,
    video_bnchw, audio_bnc,
    mouse_bnd, btn_bnd,
    rollout_manager,
    cfg_scale = 1.3
):
    # Get rollout
    video_bnchw, audio_bnc, mouse_bnd, btn_bnd, mask = rollout_manager.get_rollouts(
        model = student,
        video_bnchw = video_bnchw,
        audio_bnc = audio_bnc,
        mouse_bnd = mouse_bnd,
        btn_bnd = btn_bnd
    )
    with torch.no_grad():
        # Get ts ~ U(0.02, 0.98)
        ts_bn = torch.rand(video_bnchw.shape[0], video_bnchw.shape[1]) * 0.96 + 0.02
        ts_bn = ts_bn.to(video_bnchw.device, video_bnchw.dtype)
        ts_bn_exp = ts_bn[:,:,None,None,None]
        ts_bn_exp_audio = ts_bn[:,:,None]

        # Get noise
        z_vid_bnchw = torch.randn_like(video_bnchw)
        z_aud_bnc = torch.randn_like(audio_bnc)
        
        # Noise up the samples
        noisy_vid_bnchw = video_bnchw * (1. - ts_bn_exp) + z_vid_bnchw * ts_bn_exp
        noisy_aud_bnc = audio_bnc * (1. - ts_bn_exp_audio) + z_aud_bnc * ts_bn_exp_audio

        # Masks for cfg
        uncond_mask = torch.zeros(video_bnchw.shape[0], dtype=torch.bool, device=video_bnchw.device)
        cond_mask = torch.ones(video_bnchw.shape[0], dtype=torch.bool, device=video_bnchw.device)
        
        # Get velocities from teacher
        vid_pred_uncond, aud_pred_uncond = teacher(
            noisy_vid_bnchw,
            noisy_aud_bnc,
            ts_bn,
            mouse_bnd,
            btn_bnd,
            has_controls=uncond_mask
        )

        vid_pred_cond, aud_pred_cond = teacher(
            noisy_vid_bnchw,
            noisy_aud_bnc,
            ts_bn,
            mouse_bnd,
            btn_bnd,
            has_controls=cond_mask
        )
        
        v_teacher_vid = vid_pred_uncond + cfg_scale * (vid_pred_cond - vid_pred_uncond)
        v_teacher_aud = aud_pred_uncond + cfg_scale * (aud_pred_cond - aud_pred_uncond)

        # Velocities from critic
        v_critic_vid, v_critic_aud = critic(
            noisy_vid_bnchw,
            noisy_aud_bnc,
            ts_bn,
            mouse_bnd,
            btn_bnd
        )

        # Get predictions mu_real, mu_fake
        mu_teacher_vid = noisy_vid_bnchw - ts_bn_exp * v_teacher_vid
        mu_teacher_aud = noisy_aud_bnc - ts_bn_exp_audio * v_teacher_aud
        mu_critic_vid = noisy_vid_bnchw - ts_bn_exp * v_critic_vid
        mu_critic_aud = noisy_aud_bnc - ts_bn_exp_audio * v_critic_aud
        
        # Get normalizers
        vid_normalizer = torch.abs(video_bnchw - mu_teacher_vid).mean(dim=[1,2,3,4],keepdim=True)
        aud_normalizer = torch.abs(audio_bnc - mu_teacher_aud).mean(dim=[1,2],keepdim=True)

        # Get gradients
        grad_vid = (mu_critic_vid - mu_teacher_vid) / vid_normalizer
        grad_aud = (mu_critic_aud - mu_teacher_aud) / aud_normalizer
        grad_vid = torch.nan_to_num(grad_vid, nan=0.0)
        grad_aud = torch.nan_to_num(grad_aud, nan=0.0)

        # Get targets
        target_vid = (video_bnchw.double() - grad_vid.double()).detach()
        target_aud = (audio_bnc.double() - grad_aud.double()).detach()

        # Get masks
        grad_mask_vid = mask[:,:,None,None,None].repeat(1,1,video_bnchw.shape[2],video_bnchw.shape[3],video_bnchw.shape[4])
        grad_mask_aud = mask[:,:,None].repeat(1,1,audio_bnc.shape[2])

    # Get losses
    vid_loss = 0.5 * F.mse_loss(
        video_bnchw[grad_mask_vid].double(),
        target_vid[grad_mask_vid],
        reduction = 'mean'
    )
    aud_loss = 0.5 * F.mse_loss(
        audio_bnc[grad_mask_aud].double(),
        target_aud[grad_mask_aud],
        reduction = 'mean'
    )
    return (vid_loss + aud_loss) * 0.5

class CausVidTrainer(BaseTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Ensure causal model and no cfg
        self.model_cfg.cfg_prob = 0.0
        self.model_cfg.causal = True

        # === init teacher ===
        teacher_cfg_path = self.train_cfg.teacher_cfg
        teacher_ckpt_path = self.train_cfg.teacher_ckpt
        teacher_cfg = Config.from_yaml(teacher_cfg_path).model
        teacher_ckpt = versatile_load(teacher_ckpt_path)

        self.teacher = get_model_cls(teacher_cfg.model_id)(teacher_cfg)
        try:
            self.teacher.load_state_dict(teacher_ckpt)
        except:
            self.teacher.core.load_state_dict(teacher_ckpt)
    
        # === init student (and fake score fn) ===
        student_ckpt_path = self.train_cfg.student_ckpt
        student_ckpt = versatile_load(student_ckpt_path)

        self.student = get_model_cls(self.model_cfg.model_id)(self.model_cfg)
        try:
            self.student.load_state_dict(student_ckpt)
        except:
            self.student.core.load_state_dict(student_ckpt)

        self.critic = deepcopy(self.student)

        # All models should be cores only
        # (idiosyncracy with model impls)
        self.teacher = self.teacher.core
        self.student = self.student.core
        self.critic = self.critic.core

        # Print model size for logging
        if self.rank == 0:
            n_params = sum(p.numel() for p in self.student.parameters())
            print(f"Model has {n_params:,} parameters")

        # Initialize parameters for organizations sake
        self.ema = None
        self.opt = None
        self.critic_opt = None
        self.scaler = None
        self.critic_scaler = None
        self.total_step_counter = 0
        self.scheduler = None

        # === decoders for sampling ===
        self.decoder = get_decoder_only(
            None,
            self.train_cfg.vae_cfg_path,
            self.train_cfg.vae_ckpt_path
        )

        self.audio_decoder = get_decoder_only(
            None,
            self.train_cfg.audio_vae_cfg_path,
            self.train_cfg.audio_vae_ckpt_path
        )
    
    def save(self):
        save_dict = {
            'model' : self.student.state_dict(),
            'ema' : self.ema.state_dict(),
            'opt' : self.opt.state_dict(),
            'scaler' : self.scaler.state_dict(),
            'critic' : self.critic.state_dict(),
            'critic_opt' : self.critic_opt.state_dict(),
            'critic_scaler' : self.critic_scaler.state_dict(),
        }
        super().save(save_dict)

    def load(self):
        has_ckpt = False
        if hasattr(self.train_cfg, 'resume_ckpt') and self.train_cfg.resume_ckpt is not None:
            save_dict = super().load(self.train_cfg.resume_ckpt)
            has_ckpt = True
        
        if not has_ckpt:
            return
        
        self.student.load_state_dict(save_dict['model'])
        self.ema.load_state_dict(save_dict['ema'])
        self.opt.load_state_dict(save_dict['opt'])
        self.scaler.load_state_dict(save_dict['scaler'])
        self.critic.load_state_dict(save_dict['critic'])
        self.critic_opt.load_state_dict(save_dict['critic_opt'])
        self.critic_scaler.load_state_dict(save_dict['critic_scaler'])
        self.total_step_counter = save_dict['steps']

    def train(self):
        torch.cuda.set_device(self.local_rank)

        # Inference only modules are frozen eval+cuda+bf16
        self.teacher = self.teacher.cuda().eval().bfloat16()
        self.decoder = self.decoder.cuda().eval().bfloat16()
        self.audio_decoder = self.audio_decoder.cuda().eval().bfloat16()

        # Training modules are train+cuda
        self.student = self.student.cuda().train()
        self.critic = self.critic.cuda().train()

        # DDP
        if self.world_size > 1:
            self.student = DDP(self.student, find_unused_parameters=True)
            self.critic = DDP(self.critic, find_unused_parameters=True)

        # Ema model
        self.ema = EMA(
            self.student,
            beta = 0.99,
            update_after_step = 0,
            update_every = 1
        )
        ema_module = lambda: self.ema.ema_model.module if self.world_size > 1 else self.ema.ema_model
        
        # Prepare decode functions for sampling
        frame_decode_fn = make_batched_decode_fn(self.decoder, batch_size=self.train_cfg.vae_batch_size)
        audio_decode_fn = make_batched_audio_decode_fn(self.audio_decoder, batch_size=self.train_cfg.vae_batch_size)

        # Initialize optimizers and scalers and amp context
        self.opt = getattr(torch.optim, self.train_cfg.opt)(self.student.parameters(), **self.train_cfg.opt_kwargs)
        self.critic_opt = getattr(torch.optim, self.train_cfg.opt)(self.critic.parameters(), **self.train_cfg.d_opt_kwargs)
        self.scaler = torch.amp.GradScaler()
        self.critic_scaler = torch.amp.GradScaler()
        ctx = torch.amp.autocast('cuda', torch.bfloat16)

        self.load()

        # Logging helpers
        timer = Timer()
        timer.reset()
        metrics = LogHelper()
        if self.rank == 0:
            wandb.watch(self.student, log = 'all')
        
        # Dataset and sampling prep
        loader = get_loader(self.train_cfg.data_id, self.train_cfg.batch_size, **self.train_cfg.data_kwargs)
        loader = iter(loader)
        sampler = get_sampler_cls(self.train_cfg.sampler_id)(**self.train_cfg.sampler_kwargs)
        rollout_manager = RolloutManager(self.model_cfg, self.train_cfg.min_rollout_frames, self.train_cfg.rollout_steps)

        # Gradient accumulation setup
        accum_steps = self.train_cfg.target_batch_size // self.train_cfg.batch_size // self.world_size
        accum_steps = max(1, accum_steps)

        # optimizer step
        def optimizer_step(model, scaler, optimizer):
            scaler.unscale_(optimizer)
            g_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            optimizer.zero_grad(set_to_none=True)
            scaler.update()
            return g_norm

        # simplify getting batches
        def get_batch():
            vid_bnchw, aud_bnc, mouse_bnd, btn_bnd = next(loader)
            vid_bnchw = vid_bnchw / self.train_cfg.vae_scale
            aud_bnc = aud_bnc / self.train_cfg.audio_vae_scale
            return vid_bnchw.to(self.device), aud_bnc.to(self.device), mouse_bnd.to(self.device), btn_bnd.to(self.device)

        # === training loop ===
        while True:
            freeze(self.student)
            unfreeze(self.critic)

            for _ in range(self.train_cfg.update_ratio):
                for _ in range(accum_steps):
                    vid_bnchw, aud_bnc, mouse_bnd, btn_bnd = get_batch()
                    
                    with ctx:
                        critic_loss = get_critic_loss(
                            student = self.student,
                            critic = self.critic,
                            video_bnchw = vid_bnchw,
                            audio_bnc = aud_bnc,
                            mouse_bnd = mouse_bnd,
                            btn_bnd = btn_bnd,
                            rollout_manager = rollout_manager
                        ) / accum_steps
                        metrics.log('critic_loss', critic_loss)
                        self.critic_scaler.scale(critic_loss).backward()

                optimizer_step(self.critic, self.critic_scaler, self.critic_opt)

            freeze(self.critic)
            unfreeze(self.student)

            for _ in range(accum_steps):
                vid_bnchw, aud_bnc, mouse_bnd, btn_bnd = get_batch()
                
                with ctx:
                    dmd_loss = get_dmd_loss(
                        student = self.student,
                        critic = self.critic,
                        teacher = self.teacher,
                        video_bnchw = vid_bnchw,
                        audio_bnc = aud_bnc,
                        mouse_bnd = mouse_bnd,
                        btn_bnd = btn_bnd,
                        rollout_manager = rollout_manager
                    ) / accum_steps
                    metrics.log('dmd_loss', dmd_loss)
                    self.scaler.scale(dmd_loss).backward()

                g_norm = optimizer_step(self.student, self.scaler, self.opt)
                metrics.log('g_norm', g_norm)

            with torch.no_grad():
                # Logging
                self.ema.update()
                wandb_dict = metrics.pop()
                wandb_dict['time'] = timer.hit()
                timer.reset()

                # Sampling
                if self.total_step_counter % self.train_cfg.sample_interval == 0:
                    with ctx:
                        n_samples = self.train_cfg.n_samples
                        vid_bnchw, aud_bnc, mouse_bnd, btn_bnd = get_batch()

                        vid_bnchw = vid_bnchw[:n_samples]
                        aud_bnc = aud_bnc[:n_samples]
                        mouse_bnd = mouse_bnd[:n_samples]
                        btn_bnd = btn_bnd[:n_samples]

                        sample_video, sample_audio, sample_mouse, sample_button = sampler(
                            ema_module(),
                            vid_bnchw,
                            aud_bnc,
                            mouse_bnd,
                            btn_bnd,
                            frame_decode_fn,
                            audio_decode_fn,
                            self.train_cfg.vae_scale,
                            self.train_cfg.audio_vae_scale
                        )

                        if self.rank == 0:
                            wandb_av_out = to_wandb_av(sample_video, sample_audio, sample_mouse, sample_button)
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