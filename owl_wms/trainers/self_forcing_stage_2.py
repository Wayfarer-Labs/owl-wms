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
from ..utils.logging import LogHelper, to_wandb_av
from ..muon import init_muon
from ..utils.owl_vae_bridge import get_decoder_only, make_batched_decode_fn, make_batched_audio_decode_fn

from ..nn.kv_cache import KVCache
import random
from ..utils import batch_permute_to_length

"""
Notes:
Additional train cfg things:
1. rollout_frames (how many frames to generate))
2. update_ratio (how many times to update the score model)
3. rollout_steps (diffusion steps per rollout)

Model should be core since the wrapper is useless
"""

class DeterministicRNG:
    def __init__(self, seed=0):
        self.seed = seed
        self.call_count = 0

    def __call__(self, low, high):
        return 8
        self.call_count += 1

        # Use a combination of seed and call_count to generate deterministic values
        # Mix the seed with call_count using XOR and multiplication for better distribution
        combined = (self.seed ^ (self.call_count * 1103515245)) % (2**32)
        # Map to range [low, high] inclusive
        range_size = high - low + 1
        return low + (combined % range_size)

RNG_HANDLER = DeterministicRNG()

class RolloutHandler:
    def __init__(self, model_cfg, batch_size, min_rollout_frames, sampling_steps = 3):
        self.batch_size = batch_size
        self.model_cfg = model_cfg

        self.min_rollout_frames = min_rollout_frames
        self.sampling_steps = sampling_steps
        self.window_size = model_cfg.n_frames

    @torch.no_grad()
    def sample_for_critic(
        self,
        model,
        video,
        audio,
        mouse,
        btn
    ):
        kv_cache = KVCache(self.model_cfg)
        kv_cache.reset(self.batch_size)

        context_frames = video.shape[1]
        rollout_frames = RNG_HANDLER(self.min_rollout_frames, context_frames)
        extended_mouse, extended_btn = batch_permute_to_length(mouse, btn, rollout_frames+context_frames)
        target_mouse = extended_mouse[:,rollout_frames:]
        target_btn = extended_btn[:,rollout_frames:]

        context_video = video
        context_audio = audio
        context_mouse = mouse
        context_btn = btn

        b,n,_,_,_ = context_video.shape
        kv_cache.enable_cache_updates()
        ts = torch.zeros_like(context_video[:,:,0,0,0])
        _ = model(context_video, context_audio, ts, context_mouse, context_btn, kv_cache=kv_cache)
        kv_cache.disable_cache_updates()

        # timesteps from 0.25,0.5,0.75,1.0
        for frame_idx in range(rollout_frames):
            ts = torch.ones_like(context_video[:,0:1,0,0,0])
            dt = 1. / self.sampling_steps

            new_frame = torch.randn_like(context_video[:,0:1])
            new_audio = torch.randn_like(context_audio[:,0:1])

            new_mouse = target_mouse[:,frame_idx:frame_idx+1]
            new_btn = target_btn[:,frame_idx:frame_idx+1]

            if kv_cache.n_frames() >= self.window_size:
                kv_cache.truncate(1, front=False)

            for _ in range(self.sampling_steps):
                vid_pred, aud_pred = model(new_frame, new_audio, ts, new_mouse, new_btn, kv_cache=kv_cache)
                new_frame -= dt * vid_pred
                new_audio -= dt * aud_pred
                ts -= dt

            kv_cache.enable_cache_updates()
            _,_ = model(new_frame, new_audio, ts, new_mouse, new_btn, kv_cache=kv_cache)
            kv_cache.disable_cache_updates()

            context_video = torch.cat([context_video, new_frame], dim = 1)
            context_audio = torch.cat([context_audio, new_audio], dim = 1)
    
        context_video = context_video[:,-context_frames:]
        context_audio = context_audio[:,-context_frames:]
        context_mouse = extended_mouse[:,-context_frames:]
        context_btn = extended_btn[:,-context_frames:]

        return context_video, context_audio, context_mouse, context_btn, rollout_frames

    def sample_for_dmd(
        self,
        model,
        video,
        audio,
        mouse,
        btn
    ):
        # ------------------------------------------------------------------ #
        # ❷  Set-up (identical to sample_for_critic)                         #
        # ------------------------------------------------------------------ #
        kv_cache = KVCache(self.model_cfg)
        kv_cache.reset(self.batch_size)

        context_frames = video.shape[1]
        rollout_frames = RNG_HANDLER(self.min_rollout_frames, context_frames)
        extended_mouse, extended_btn = batch_permute_to_length(mouse, btn, rollout_frames+context_frames)
    
        context_video = video
        context_audio = audio
        context_mouse = mouse
        context_btn = btn

        target_mouse = extended_mouse[:, rollout_frames:]
        target_btn = extended_btn[:, rollout_frames:]

        # Seed the cache with the conditioning context
        with torch.no_grad():
            ts = torch.zeros_like(context_video[:, :, 0, 0, 0])
            kv_cache.enable_cache_updates()
            _ = model(
                context_video,
                context_audio,
                ts,
                context_mouse,
                context_btn,
                kv_cache=kv_cache
            )
            kv_cache.disable_cache_updates()

        # ------------------------------------------------------------------ #
        # ❸  Autoregressive rollout                                          #
        # ------------------------------------------------------------------ #
        dt             = 1.0 / self.sampling_steps

        for frame_idx in range(rollout_frames):
            # -------------------------------------------------------------- #
            # Pick the "special" diffusion step s (0-based) that keeps grads #
            # -------------------------------------------------------------- #
            s = random.randint(0, self.sampling_steps - 1)

            ts         = torch.ones_like(context_video[:, 0:1, 0, 0, 0])
            new_frame  = torch.randn_like(context_video[:, 0:1])
            new_audio  = torch.randn_like(context_audio[:, 0:1])
            new_mouse  = target_mouse[:, frame_idx:frame_idx + 1]
            new_btn    = target_btn[:,   frame_idx:frame_idx + 1]

            # Make sure the KV-cache never grows past the sliding window
            if kv_cache.n_frames() >= self.window_size:
                kv_cache.truncate(1, front=False)

            # -------------------------------------------------------------- #
            # Diffusion (self.sampling_steps steps)                          #
            # -------------------------------------------------------------- #
            for step in range(self.sampling_steps):
                if step == s:
                    # ONE step per frame keeps the computation graph
                    vid_pred, aud_pred = model(new_frame,
                                                new_audio,
                                                ts,
                                                new_mouse,
                                                new_btn,
                                                kv_cache=kv_cache)

                    new_frame = new_frame - dt * vid_pred
                    new_audio = new_audio - dt * aud_pred
                    ts = ts - dt

                    break
                else:
                    with torch.no_grad():
                        # All other steps stay in no-grad mode (cheap)
                        vid_pred, aud_pred = model(new_frame,
                                                new_audio,
                                                ts,
                                                new_mouse,
                                                new_btn,
                                                kv_cache=kv_cache)
                    vid_pred = vid_pred.detach()
                    aud_pred = aud_pred.detach()

                    new_frame = new_frame - dt * vid_pred
                    new_audio = new_audio - dt * aud_pred
                    ts        = ts - dt

            with torch.no_grad():
                kv_cache.enable_cache_updates()
                _ = model(new_frame, new_audio, ts, new_mouse, new_btn, kv_cache=kv_cache)
                kv_cache.disable_cache_updates()
            # -------------------------------------------------------------- #
            # Extend the running context that the next frame will condition  #
            # on.                                                            #
            # -------------------------------------------------------------- #
            context_video = torch.cat([context_video, new_frame], dim=1)
            context_audio = torch.cat([context_audio, new_audio], dim=1)

        # ------------------------------------------------------------------ #
        # ❹  Return the fully-generated rollout                              #
        # ------------------------------------------------------------------ #

        context_video = context_video[:,-context_frames:]
        context_audio = context_audio[:,-context_frames:]
        context_mouse = extended_mouse[:,-context_frames:]
        context_btn = extended_btn[:,-context_frames:]

        return context_video, context_audio, context_mouse, context_btn, rollout_frames
            
class SelfForceTrainer(BaseTrainer):
    """
    CausVid Trainer

    :param train_cfg: Configuration for training
    :param logging_cfg: Configuration for logging
    :param model_cfg: Configuration for model
    :param global_rank: Rank across all devices.
    :param local_rank: Rank for current device on this process.
    :param world_size: Overall number of devices
    """
    def __init__(self,*args,**kwargs):  
        super().__init__(*args,**kwargs)

        model_id = self.model_cfg.model_id

        student_cfg = deepcopy(self.model_cfg)
        teacher_cfg = deepcopy(self.model_cfg)

        student_cfg.causal = True
        teacher_cfg.causal = False

        self.model = get_model_cls(model_id)(student_cfg)
        self.score_real = get_model_cls(model_id)(teacher_cfg)

        self.model.load_state_dict(versatile_load(self.train_cfg.student_ckpt))
        self.score_real.load_state_dict(versatile_load(self.train_cfg.teacher_ckpt))
        self.score_fake = deepcopy(self.score_real)
        self.score_fake.cfg_prob = 0.0 # No cfg needed for this one
        freeze(self.score_real)

        self.model = self.model.core

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

    def save(self):
        save_dict = {
            'model' : self.model.state_dict(),
            'ema' : self.ema.state_dict(),
            'opt' : self.opt.state_dict(),
            'scaler' : self.scaler.state_dict(),
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
        if hasattr(self.train_cfg, 'resume_ckpt') and self.train_cfg.resume_ckpt is not None:
            save_dict = super().load(self.train_cfg.resume_ckpt)
            has_ckpt = True
        
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

    def train(self):
        torch.cuda.set_device(self.local_rank)

        # Prepare model and ema
        self.model = self.model.cuda().train()        
        self.decoder = self.decoder.cuda().eval().bfloat16()
        self.audio_decoder = self.audio_decoder.cuda().eval().bfloat16()
        self.score_real = self.score_real.cuda().eval().bfloat16()
        self.score_fake = self.score_fake.cuda().train()

        if self.world_size > 1:
            self.model = DDP(self.model, find_unused_parameters=True)
            self.score_fake = DDP(self.score_fake, find_unused_parameters=True)

        freeze(self.decoder)
        freeze(self.audio_decoder)
        freeze(self.score_real)

        #self.score_real = torch.compile(self.score_real, dynamic = False)

        decode_fn = make_batched_decode_fn(self.decoder, self.train_cfg.vae_batch_size)
        audio_decode_fn = make_batched_audio_decode_fn(self.audio_decoder, self.train_cfg.vae_batch_size)

        self.ema = EMA(
            self.model,
            beta = 0.99,
            update_after_step = 0,
            update_every = 1
        )
        # Hard coded stuff, probably #TODO figure out where to put this?
        self.update_ratio = self.train_cfg.update_ratio
        self.cfg_scale = 1.3

        def get_ema_core():
            if self.world_size > 1:
                return self.ema.ema_model.module
            else:
                return self.ema.ema_model

        # Don't use MUON pls
        self.opt = getattr(torch.optim, self.train_cfg.opt)(self.model.parameters(), **self.train_cfg.opt_kwargs)
        self.s_fake_opt = getattr(torch.optim, self.train_cfg.opt)(self.score_fake.parameters(), **self.train_cfg.d_opt_kwargs)

        if self.train_cfg.scheduler is not None:
            self.scheduler = get_scheduler_cls(self.train_cfg.scheduler)(self.opt, **self.train_cfg.scheduler_kwargs)

        # Scaler
        self.s_fake_scaler = torch.amp.GradScaler()
        self.scaler = torch.amp.GradScaler()
        ctx = torch.amp.autocast('cuda',torch.bfloat16)

        self.load()

        # Timer reset
        timer = Timer()
        timer.reset()
        metrics = LogHelper()
        if self.rank == 0:
            wandb.watch(self.get_module(), log = 'all')
        
        # Dataset setup
        loader = get_loader(self.train_cfg.data_id, self.train_cfg.batch_size, **self.train_cfg.data_kwargs)
        sampler = get_sampler_cls(self.train_cfg.sampler_id)(**self.train_cfg.sampler_kwargs)
        rollouts = RolloutHandler(self.model_cfg, self.train_cfg.batch_size, self.train_cfg.min_rollout_frames, self.train_cfg.rollout_steps)

        # Simplifiying assumptions: data will never stop iter, no grad accum

        def sample_from_gen(vid, audio, mouse, btn, for_dmd=False):
            # Use rollout handler to generate samples
            if for_dmd:
                video_samples, audio_samples, mouse_samples, btn_samples, rollout_frames = rollouts.sample_for_dmd(
                    self.model,
                    vid,
                    audio, 
                    mouse,
                    btn
                )
            else:
                video_samples, audio_samples, mouse_samples, btn_samples, rollout_frames = rollouts.sample_for_critic(
                    self.model,
                    vid,
                    audio, 
                    mouse,
                    btn
                )
            return video_samples, audio_samples, mouse_samples, btn_samples, rollout_frames

        # TODO account for n_context_frames in gradients
        def get_dmd_loss(vid, audio, mouse, btn, rollout_frames):
            s_real_fn = self.score_real.core
            s_fake_fn = self.score_fake.module.core if self.world_size > 1 else self.score_fake.core
            n_context_frames = vid.shape[1] - rollout_frames

            with torch.no_grad():
                b,n,c,h,w = vid.shape
                ts = torch.rand(b,n,device=vid.device, dtype=vid.dtype) * 0.96 + 0.02
                z_vid = torch.randn_like(vid)
                z_audio = torch.randn_like(audio)
                ts_exp = ts[:,:,None,None,None]
                ts_exp_audio = ts[:,:,None]

                lerpd_video = vid * (1. - ts_exp) + z_vid * ts_exp
                lerpd_audio = audio * (1. - ts_exp_audio) + z_audio * ts_exp_audio

                null_mouse = torch.zeros_like(mouse)
                null_btn = torch.zeros_like(btn)

                # Create masks for conditional and unconditional branches
                uncond_mask = torch.zeros(b, dtype=torch.bool, device=vid.device)
                cond_mask = torch.ones(b, dtype=torch.bool, device=vid.device)

                # === GET ALL SCORES/VELOCITIES === #

                # Get unconditional predictions
                vid_pred_uncond, aud_pred_uncond = s_real_fn(lerpd_video, lerpd_audio, ts, null_mouse, null_btn, has_controls=uncond_mask)
                
                # Get conditional predictions
                vid_pred_cond, aud_pred_cond = s_real_fn(lerpd_video, lerpd_audio, ts, mouse, btn, has_controls=cond_mask)

                # Apply CFG
                s_real_vid = vid_pred_uncond + self.cfg_scale * (vid_pred_cond - vid_pred_uncond)
                s_real_aud = aud_pred_uncond + self.cfg_scale * (aud_pred_cond - aud_pred_uncond)

                # Get fake score predictions
                s_fake_vid, s_fake_aud = s_fake_fn(lerpd_video, lerpd_audio, ts, mouse, btn, has_controls=cond_mask)

                # === GET PREDICTIONS === #
                real_pred_video = lerpd_video - ts_exp * s_real_vid
                fake_pred_video = lerpd_video - ts_exp * s_fake_vid
                real_pred_audio = lerpd_audio - ts_exp_audio * s_real_aud
                fake_pred_audio = lerpd_audio - ts_exp_audio * s_fake_aud

                # === GET NORMALIZERS === #
                vid_normalizer = torch.abs(vid - real_pred_video).mean(dim=[1,2,3,4],keepdim=True).clamp(min=1.0e-6)
                aud_normalizer = torch.abs(audio - real_pred_audio).mean(dim=[1,2],keepdim=True).clamp(min=1.0e-6)

                # === GET GRADIENTS === #
                grad_vid = (fake_pred_video - real_pred_video) / vid_normalizer
                grad_aud = (fake_pred_audio - real_pred_audio) / aud_normalizer
                grad_vid = torch.nan_to_num(grad_vid, nan=0.0)
                grad_aud = torch.nan_to_num(grad_aud, nan=0.0)

                # === GET MASK === #

                grad_mask = torch.ones_like(vid[:,:,0,0,0])  # [b,n]
                grad_mask[:,:n_context_frames] = 0  # Zero out context frames
                vid_grad_mask = grad_mask.view(vid.shape[0], vid.shape[1], 1, 1, 1)  # [b,n,1,1,1]
                aud_grad_mask = grad_mask.view(vid.shape[0], vid.shape[1], 1) # [b,n,1]

            # Calculate losses
            vid_loss = 0.5 * F.mse_loss(
                vid.double() * vid_grad_mask, 
                (vid.double() - grad_vid.double()).detach() * vid_grad_mask
            )
            aud_loss = 0.5 * F.mse_loss(
                audio.double() * aud_grad_mask,
                (audio.double() - grad_aud.double()).detach() * aud_grad_mask
            )

            # Return average loss
            return (vid_loss + aud_loss) / 2
        
        def optimizer_step(model, scaler, optimizer):
            # Assumes loss.backward() was already called
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            scaler.step(optimizer)
            optimizer.zero_grad(set_to_none=True)
            scaler.update()
        
        def convert(inputs, device, dtype):
            return [x.to(device=device, dtype=dtype) for x in inputs]

        # Gradient accumulation setup
        accum_steps = self.train_cfg.target_batch_size // self.train_cfg.batch_size // self.world_size
        accum_steps = max(1, accum_steps)
        local_step = 0

        loader = iter(loader)
        while True:
            freeze(self.model)
            unfreeze(self.score_fake)
            
            # Score fake training loop with grad accum
            for _ in range(self.update_ratio):
                for _ in range(accum_steps):
                    batch_vid, batch_audio, batch_mouse, batch_btn = convert(next(loader), 'cuda', torch.bfloat16)
                    batch_vid = batch_vid / self.train_cfg.vae_scale
                    batch_audio = batch_audio / self.train_cfg.audio_vae_scale
     
                    with ctx:
                        with torch.no_grad():
                            video_samples, audio_samples, mouse_samples, btn_samples, _ = sample_from_gen(batch_vid, batch_audio, batch_mouse, batch_btn, for_dmd=False)
                        
                        s_fake_loss = self.score_fake(video_samples, audio_samples, mouse_samples, btn_samples)
                        self.s_fake_scaler.scale(s_fake_loss / accum_steps).backward()

                    metrics.log('s_fake_loss', s_fake_loss/accum_steps)

                # Only do optimization step after accumulation
                optimizer_step(self.score_fake, self.s_fake_scaler, self.s_fake_opt)


            unfreeze(self.model)
            freeze(self.score_fake)
        
            # Generator training with grad accum
            for _ in range(accum_steps):
                batch_vid, batch_audio, batch_mouse, batch_btn = convert(next(loader), 'cuda', torch.bfloat16)
                batch_vid = batch_vid / self.train_cfg.vae_scale
                batch_audio = batch_audio / self.train_cfg.audio_vae_scale

                with ctx:
                    video_samples, audio_samples, mouse_samples, btn_samples, rollout_frames = sample_from_gen(batch_vid, batch_audio, batch_mouse, batch_btn, for_dmd=True)
                    dmd_loss = get_dmd_loss(video_samples, audio_samples, mouse_samples, btn_samples, rollout_frames)
                    self.scaler.scale(dmd_loss / accum_steps).backward()
                
                metrics.log('dmd_loss', dmd_loss/accum_steps)

            # Only do optimization step after accumulation
            optimizer_step(self.model, self.scaler, self.opt)
            
            self.ema.update()

            with torch.no_grad():
                wandb_dict = metrics.pop()
                wandb_dict['time'] = timer.hit()
                timer.reset()

            if self.total_step_counter % self.train_cfg.sample_interval == 0:
                with ctx, torch.no_grad():
                    n_samples = self.train_cfg.n_samples
                    samples, audio, sample_mouse, sample_button = sampler(
                        get_ema_core(),
                        batch_vid[:n_samples],
                        batch_audio[:n_samples],
                        batch_mouse[:n_samples],
                        batch_btn[:n_samples],
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