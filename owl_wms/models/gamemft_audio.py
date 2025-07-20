"""
GameRFT with multi timestep for Mean Flow one step generation
"""

import torch
from torch import nn
import torch.nn.functional as F


from ..nn.embeddings import (
    TimestepEmbedding,
    ControlEmbedding,
    LearnedPosEnc
)
from ..nn.attn import DiT, FinalLayer, UViT
from ..nn.mmattn import MMDIT

# Mean Flow Transformer
class GameMFTAudioCore(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config

        if config.backbone == 'dit':
            backbone_cls = DiT
        elif config.backbone == 'mmdit':
            backbone_cls = MMDIT
        elif config.backbone == 'uvit':
            backbone_cls = UViT
        else:
            raise ValueError(f"Invalid backbone: {config.backbone}")

        self.backbone = config.backbone

        self.transformer = backbone_cls(config)

        if not config.uncond: self.control_embed = ControlEmbedding(config.n_buttons, config.d_model)
        self.t_embed = TimestepEmbedding(config.d_model)
        self.r_embed = TimestepEmbedding(config.d_model)


        self.proj_in = nn.Linear(config.channels, config.d_model, bias = False)
        self.proj_out = FinalLayer(config.sample_size, config.d_model, config.channels)

        self.audio_proj_in = nn.Linear(config.audio_channels, config.d_model, bias=False)
        self.audio_proj_out = FinalLayer(None, config.d_model, config.audio_channels)

        self.uncond = config.uncond

    def forward(self, x, audio, t, mouse, btn, has_controls = None, kv_cache = None, r = None):
        # x is [b,n,c,h,w]
        # audio is [b,n,c]
        # t is [b,n]
        # mouse is [b,n,2]
        # btn is [b,n,n_buttons]

        t_cond = self.t_embed(t)

        if r is None:
            r = torch.zeros_like(t)
        r_cond = self.r_embed(t-r)
        t_cond = t_cond + r_cond

        if not self.uncond:
            ctrl_cond = self.control_embed(mouse, btn) # [b,n,d]
            if has_controls is not None:
                ctrl_cond = torch.where(has_controls[:,None,None], ctrl_cond, torch.zeros_like(ctrl_cond))
            cond = t_cond + ctrl_cond # [b,n,d]
        else:
            cond = t_cond
        
        b,n,c,h,w = x.shape
        x = x.permute(0,1,3,4,2) # bnhwc
        x = x.reshape(b,n*h*w,c) # b(nhw)c

        x = self.proj_in(x) # b(nhw)d
        audio = self.audio_proj_in(audio) # bnd

        if self.backbone == 'dit' or self.backbone == 'uvit':
            audio = audio.unsqueeze(-2) # bn1d
            x = x.reshape(b, n, -1, x.shape[-1]) # bn(hw)d
            x = torch.cat([x, audio], dim = -2) # bn(hw+1)d
            x = x.reshape(b, n * x.shape[2], x.shape[-1]) # b(n(hw+1))d
            x = self.transformer(x, cond, kv_cache)

            # Split into video and audio tokens
            x = x.view(b, n, -1, x.shape[-1]) # bn(hw+1)d
            video, audio = x[...,:-1,:], x[...,-1:,:] # bn(hw)d | bn1d
            video = video.reshape(b, n * video.shape[2], video.shape[-1]) # b(nhw)d

        elif self.backbone == 'mmdit':
            video, audio = self.transformer(x, audio, cond, kv_cache)

        # Project video tokens
        video = self.proj_out(video, cond) 
        video = video.reshape(b, n, h, w, c).permute(0, 1, 4, 2, 3) # bnchw

        # Project audio tokens
        audio = audio.squeeze(-2) # bnd
        audio = self.audio_proj_out(audio, cond)

        return video, audio

class GameMFTAudio(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.core = GameRFTAudioCore(config)

        # Mean Flow has a lot of hyperparameters

        # Defines Sampling for timesteps
        self.ts_mu = - 0.4
        self.ts_sigma = 1.0
        self.ts_ratio = 0.25 # Percent to force r = t

        # CFG specifics
        self.cfg_scale = 1.3 # \omega'
        self.cfg_scale_2 = 1.0 # \omega
        self.kappa = 1.0 - self.cfg_scale_2 / self.cfg_scale # \kappa
        self.cfg_prob = 0.1 
        self.cfg_in = [0.3,0.8] # trigger CFG in this t range

    @torch.no_grad()
    def sample_timesteps(self, b, n, device, dtype):
        mu = self.ts_mu
        sigma = self.ts_sigma
        ratio = self.ts_ratio # percent to force eq

        eq_mask = torch.rand(b,n,device=device,dtype=dtype) < ratio

        t_both = torch.randn(b,n,2,device=device,dtype=dtype)
        t_both = t_both * sigma + mu
        t_both = t_both.sigmoid()

        t1 = t_both[...,0]
        t2 = t_both[...,1]

        lesser = t1 < t2

        r = torch.where(lesser, t1, t2) # r is the smaller number
        t = torch.where(~lesser, t1, t2) # t is the larger number

        r = torch.where(eq_mask, t, r) # make them the same with prob = .25

        return t, r
 
    def handle_cfg(self, has_controls = None, cfg_prob = None):
        if cfg_prob is None:
            cfg_prob = self.cfg_prob
        if cfg_prob <= 0.0 or has_controls is None:
            return has_controls
            
        # Calculate current percentage without controls
        pct_without = 1.0 - has_controls.float().mean()
        
        # Only apply CFG if we need more negatives
        if pct_without < cfg_prob:
            # Calculate how many more we need
            needed = cfg_prob - pct_without
            needed_frac = needed / has_controls.float().mean()
            
            # Only drop controls where has_controls is True
            b = has_controls.shape[0]
            mask = (torch.rand(b, device=has_controls.device) <= needed_frac) & has_controls
            
            # Update has_controls based on mask
            has_controls = has_controls & (~mask)
            
        return has_controls

    def forward(self, x, audio, mouse, btn, return_dict = False, cfg_prob = None, has_controls = None):
        # x is [b,n,c,h,w]
        # audio is [b,n,c]
        # mouse is [b,n,2]
        # btn is [b,n,n_buttons]
        # has_controls: [b,] boolean mask for which batch elements have controls
        b,n,c,h,w = x.shape

        # By default assume all has controls
        # Drop out based on cfg prob
        if has_controls is None:
            has_controls = torch.ones(b, device=x.device, dtype=torch.bool)
        has_controls = self.handle_cfg(has_controls, cfg_prob)

        with torch.no_grad():
            ts, rs = sample_timesteps(b, n, x.device, x.dtype)
            
            # alpha_t = (1. - t)
            # sigma_t = (t)
            # d_alpha_t = -1
            # d_sigma_t = 1

            z_video = torch.randn_like(x)
            z_audio = torch.randn_like(audio)

            noisy_video = x * (1. - ts) + z_video * ts # need view
            v_video_t = z_video - x

            noisy_audio = audio * (1. - ts) + z_audio * ts # need view?
            v_audio_t = z_audio - audio

            ts_diff = (t - r) # need view
            eq_mask = (t == r)
            neq_mask = ~eq_mask

            u_target_video = torch.zeros_like(v_video_t)
            u_target_audio = torch.zeros_like(v_audio_t)

            u_pred_video = torch.zeros_like(v_video_t)
            u_pred_audio = torch.zeros_like(v_audio_t)

        # 1) r == t (no JVP), target is just instant velocity
        # r == t case doesnt worry about cfg
        if eq_mask.any():
            eq_idx = torch.where(eq_mask)[0]
            noisy_video_eq = noisy_video[eq_idx]
            noisy_audio_eq = noisy_audio[eq_idx]
            
            r_eq = r[eq_idx]
            t_eq = t[eq_idx]
            v_video_t_eq = v_video_t[eq_idx]
            v_audio_t_eq = v_audio_t[eq_idx]

            u_eq_video, u_eq_audio = self.core(noisy_video_eq, noisy_audio_eq, t_eq, mouse[eq_idx], btn[eq_idx], has_controls = has_controls[eq_idx], r = r_eq)
            
            u_pred_video[eq_idx] = u_eq_video
            u_pred_audio[eq_idx] = u_eq_audio

            u_target_video[eq_idx] = v_video_t_eq
            u_target_audio[eq_idx] = v_audio_t_eq
        
        # 2) r != t (JVP, standard case)
        if neg_mask.any():
            neq_idx = torch.where(neq_mask)[0]
            noisy_video_neq = noisy_video[neq_idx]
            noisy_audio_neq = noisy_audio[neq_idx]
            
            r_neq = r[neq_idx]
            t_neq = t[neq_idx]
            v_video_t_neq = v_video_t[neq_idx]
            v_audio_t_neq = v_audio_t[neq_idx]
            ts_diff_neq = ts_diff[neq_idx]

            mouse_neq = mouse[neq_idx]
            btn_neq = btn[neq_idx]

            has_controls_neq = has_controls[neq_idx]

            def fn_current_neq(z_vid, z_aud, curr_r, curr_t):
                return self.core(z_vid, z_aud, curr_t, mouse_neq, btn_neq, has_controls = has_controls_neq, r = curr_r)
        
            # Some samples get cfg signal
            cfg_ts_mask = (t_neq >= self.cfg_in[0]) & (t_neq <= self.cfg_in[1]) & (has_controls_neq)

            if cfg_ts_mask.any():
                # Further subdivide into cfg and non-cfg
                cfg_idx_local = torch.where(cfg_ts_mask)[0]
                non_cfg_idx_local = torch.where(~cfg_ts_mask)[0]

                # cfg branch
                if len(cfg_idx_local) > 0:
                    # Select subsets of the neq batch for cfg branch
                    noisy_video_neq_cfg = noisy_video_neq[cfg_idx_local]
                    noisy_audio_neq_cfg = noisy_audio_neq[cfg_idx_local]
                    r_neq_cfg = r_neq[cfg_idx_local]
                    t_neq_cfg = t_neq[cfg_idx_local]
                    v_video_t_neq_cfg = v_video_t_neq[cfg_idx_local]
                    v_audio_t_neq_cfg = v_audio_t_neq[cfg_idx_local]
                    ts_diff_neq_cfg = ts_diff_neq[cfg_idx_local]

                    # Controls
                    mouse_neq_cfg = mouse_neq[cfg_idx_local]
                    btn_neq_cfg = btn_neq[cfg_idx_local]

                    # Since CFG = no conditioning, we need to drop the controls
                    null_mouse = torch.zeros_like(mouse_neq_cfg)
                    null_btn = torch.zeros_like(btn_neq_cfg)
                    has_controls_neq_cfg = has_controls_neq[cfg_idx_local] # Should be all True

                    # And double the inputs so this is all batched
                    noisy_video_neq_cfg_double = torch.cat([noisy_video_neq_cfg, noisy_video_neq_cfg], dim = 0)
                    noisy_audio_neq_cfg_double = torch.cat([noisy_audio_neq_cfg, noisy_audio_neq_cfg], dim = 0)
                    
                    mouse_neq_cfg_double = torch.cat([mouse_neq_cfg, null_mouse], dim = 0)
                    btn_neq_cfg_double = torch.cat([btn_neq_cfg, null_btn], dim = 0)

                    # TODO: Is this rly right? Do we want r instead?
                    # Future note: Yes, it is, cause you want CFG for instant velocity
                    t_neq_cfg_double = torch.cat([t_neq_cfg, t_neq_cfg], dim = 0)
                    t_end_neq_cfg_double = torch.cat([t_neq_cfg, t_neq_cfg], dim = 0)

                    with torch.no_grad():
                        u_neq_video_cfg_double, u_neq_audio_cfg_double = self.core(
                            noisy_video_neq_cfg_double, noisy_audio_neq_cfg_double,
                            t_end_neq_cfg_double, 
                            mouse_neq_cfg_double, btn_neq_cfg_double,
                            has_controls = has_controls_neq_cfg, r = t_neq_cfg_double
                        )

                        u_neq_video_cond, u_neq_video_uncond = u_neq_video_cfg_double.chunk(2)
                        u_neq_audio_cond, u_neq_audio_uncond = u_neq_audio_cfg_double.chunk(2)

                        cfg_v_video_tilde = (
                            self.cfg_scale * v_video_t_neq_cfg +
                            self.kappa * u_neq_video_cond + 
                            (1. - self.cfg_scale - self.cfg_kappa) * u_neq_video_uncond
                        )
                        cfg_v_audio_tilde = (
                            self.cfg_scale * v_audio_t_neq_cfg +
                            self.kappa * u_neq_audio_cond + 
                            (1. - self.cfg_scale - self.cfg_kappa) * u_neq_audio_uncond
                        )

                    def fn_current_cfg(z_vid, z_aud, curr_r, curr_t):
                        return self.core(z_vid, z_aud, curr_t, mouse_neq_cfg, btn_neq_cfg, has_controls = has_controls_neq_cfg, r = curr_r)
                    
                    primals = (noisy_video_neq_cfg, noisy_audio_neq_cfg, r_neq_cfg, t_neq_cfg)
                    tangents = (cfg_v_video_tilde, cfg_v_audio_tilde, torch.zeros_like(r_neq_cfg), torch.ones_like(t_neq_cfg))

                    cfg_u_video_theta, # TODO How tf do you split two outputs from JVP?

                    # Get target, set global idx's
                if len(non_cfg_idx_local) > 0:
                    noisy_video_neq_non_cfg = noisy_video_neq[non_cfg_idx_local]
                    noisy_audio_neq_non_cfg = noisy_audio_neq[non_cfg_idx_local]
                    r_neq_non_cfg = r_neq[non_cfg_idx_local]
                    t_neq_non_cfg = t_neq[non_cfg_idx_local]
                    v_video_t_neq_non_cfg = v_video_t_neq[non_cfg_idx_local]
                    v_audio_t_neq_non_cfg = v_audio_t_neq[non_cfg_idx_local]
                    ts_diff_neq_non_cfg = ts_diff_neq[non_cfg_idx_local]

                    mouse_neq_non_cfg = mouse_neq[non_cfg_idx_local]
                    btn_neq_non_cfg = btn_neq[non_cfg_idx_local]

                    has_controls_neq_non_cfg = has_controls_neq[non_cfg_idx_local]

                    def fn_current_non_cfg(z_vid, z_aud, curr_r, curr_t):
                        return self.core(z_vid, z_aud, curr_t, mouse_neq_non_cfg, btn_neq_non_cfg, has_controls = has_controls_neq_non_cfg, r = curr_r)

                    primals = (noisy_video_neq_non_cfg, noisy_audio_neq_non_cfg, r_neq_non_cfg, t_neq_non_cfg)
                    tangents = (v_video_t_neq_non_cfg, v_audio_t_neq_non_cfg, torch.zeros_like(r_neq_non_cfg), torch.ones_like(t_neq_non_cfg))

                    # TODO JVP STUFF AGAIN

                    # Set targets
            else:
                # No cfg stuff, just normal JVP
                primals = (noisy_video_neq, noisy_audio_neq, r_neq, t_neq)
                tangents = (v_video_t_neq, v_audio_t_neq, torch.zeros_like(r_neq), torch.ones_like(t_neq))

                # TODO JVP STUFF AGAIN
        
        u_target_video = u_target_video.detach()
        u_target_audio = u_target_audio.detach()

        error_video = u_pred_video - u_target_video
        error_audio = u_pred_audio - u_target_audio

        error_norm = torch.norm(error.reshape(error.shape[0], -1), dim = 1)
        loss = error_norm ** 2

        return loss