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
from ..nn.mmattn_v2 import MMDIT2

"""
Notes for future:
- r = timestep at start of step, t = timestep at end of step
- r < t is enforced
- there are 4 branches
    - r == t (no JVP), target is just instant velocity
    - r != t (JVP, standard case)
    - r != t (JVP, CFG case)
    - r != t (JVP, REQT case)

"""

# Mean Flow Transformer
class GameMFTAudioCore(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config

        if config.backbone == 'dit':
            backbone_cls = DiT
        elif config.backbone == 'mmdit':
            backbone_cls = MMDIT2
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

        self.core = GameMFTAudioCore(config)

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

    def handle_reqt_branch(
        self,
        noisy_vid, noisy_aud,
        r, t,
        u_pred_vid, u_pred_aud,
        u_targ_vid, u_targ_aud,
        v_vid, v_aud,
        has_controls, mouse, btn,
    ):
        """
        Refer to documentation on this function for some more details that will apply to all branches.

        The r=t branch is basically just the standard flow objective, since it's instant velocity and v = u

        # Basic inputs
        :param noisy_vid: [b,n,c,h,w] x(t) for video
        :param noisy_aud: [b,n,c] x(t) for audio
        :param r: [b,n] smaller timestep
        :param t: [b,n] larger timestep

        # Stuff we generate for MSE later
        :param u_pred_vid: [b,n,c,h,w] tensor where u predictions will be stored
        :param u_pred_aud: [b,n,c] tensor where u predictions will be stored
        :param u_targ_vid: [b,n,c,h,w] tensor where u targets will be stored
        :param u_targ_aud: [b,n,c] tensor where u targets will be stored

        # Ground truth about velocity
        :param v_vid: [b,n,c,h,w] tensor where invelocity for video will be stored
        :param v_aud: [b,n,c] tensor where v targets will be stored

        # Additional inputs
        :param has_controls: [b,n] boolean mask for which batch elements have controls
        :param mouse: [b,n,2] mouse inputs
        :param btn: [b,n,n_buttons] button inputs

        # Returns:
        :return: u_pred_vid [b,n,c,h,w] with u predictions updated in branch relevant idxs
        :return: u_pred_aud [b,n,c] with u predictions updated in branch relevant idxs
        :return: u_targ_vid [b,n,c,h,w] with u targets updated in branch relevant idxs
        :return: u_targ_aud [b,n,c] with u targets updated in branch relevant idxs
        """
        with torch.no_grad():
            eq_mask = (r == t)
            if not eq_mask.any():
                return u_pred_vid, u_pred_aud, u_targ_vid, u_targ_aud
            idx = torch.where(eq_mask)[0]

            # Get slices of everything that are relevant to this branch
            noisy_vid = noisy_vid[idx]
            noisy_aud = noisy_aud[idx]
            r = r[idx]
            t = t[idx]
            v_vid = v_vid[idx]
            v_aud = v_aud[idx]
            has_controls = has_controls[idx]
            mouse = mouse[idx]
            btn = btn[idx]

        u_vid_out, u_aud_out = self.core(noisy_vid, noisy_aud, t, mouse, btn, has_controls = has_controls, r = r)

        with torch.no_grad():
            u_targ_vid[idx] = v_vid
            u_targ_aud[idx] = v_aud

        u_pred_vid[idx] = u_vid_out
        u_pred_aud[idx] = u_aud_out

        return u_pred_vid, u_pred_aud, u_targ_vid, u_targ_aud

    def handle_neq_and_cfg_branch(
        self,
        noisy_vid, noisy_aud,
        r, t,
        u_pred_vid, u_pred_aud,
        u_targ_vid, u_targ_aud,
        v_vid, v_aud,
        has_controls, mouse, btn,
    ):
        """
        Branch where r != t and also doing cfg
        """
        with torch.no_grad():
            neq_mask = (r != t)

            if not neq_mask.any():
                return u_pred_vid, u_pred_aud, u_targ_vid, u_targ_aud
            
            cfg_mask = has_controls & (t >= self.cfg_in[0]) & (t <= self.cfg_in[1])
            neq_cfg_mask = neq_mask & cfg_mask
            
            if not neq_cfg_mask.any():
                return u_pred_vid, u_pred_aud, u_targ_vid, u_targ_aud

            idx = torch.where(neq_cfg_mask)[0]
            
            noisy_vid = noisy_vid[idx]
            noisy_aud = noisy_aud[idx]
            r = r[idx]
            t = t[idx]
            v_vid = v_vid[idx]
            v_aud = v_aud[idx]
            has_controls = has_controls[idx] # Note this is inherently all true now
            mouse = mouse[idx]
            btn = btn[idx]

            full_cond = torch.ones_like(has_controls)
            null_cond = torch.zeros_like(has_controls)

            ts_diff_exp_vid = (t - r)[:,:,None,None,None]
            ts_diff_exp_aud = (t - r)[:,:,None]

            # Double inputs for CFG
            def double(x):
                return torch.cat([x, x], dim = 0)

            noisy_vid_double = double(noisy_vid)
            noisy_aud_double = double(noisy_aud)
            t_double = double(t)
            mouse_double = double(mouse)
            btn_double = double(btn)
            has_controls = torch.cat([full_cond, null_cond], dim = 0)

            # For CFG, t = r to get instant v with cfg applied

            u_vid_out, u_aud_out = self.core(
                noisy_vid_double, noisy_aud_double,
                t_double, mouse_double, btn_double,
                has_controls = has_controls,
                r = t_double,
            )

            u_vid_out_cond, u_vid_out_uncond = u_vid_out.chunk(2)
            u_aud_out_cond, u_aud_out_uncond = u_aud_out.chunk(2)

            cfg_v_vid_tilde = (
                self.cfg_scale * v_vid +
                self.kappa * u_vid_out_cond +
                (1. - self.cfg_scale - self.kappa) * u_vid_out_uncond
            )

            cfg_v_aud_tilde = (
                self.cfg_scale * v_aud +
                self.kappa * u_aud_out_cond +
                (1. - self.cfg_scale - self.kappa) * u_aud_out_uncond
            )
        
        def fn_1(z_vid, z_aud, curr_r, curr_t):
            return self.core(z_vid, z_aud, curr_t, mouse, btn, has_controls = has_controls, r = curr_r)
        
        primals = (noisy_vid, noisy_aud, r, t)
        tangents = (cfg_v_vid_tilde, cfg_v_aud_tilde, torch.zeros_like(r), torch.ones_like(t))
        (u_outs, dudt_outs) = torch.func.jvp(fn_1, primals, tangents)

        u_pred_vid[idx] = u_outs[0]
        u_pred_aud[idx] = u_outs[1]
        u_targ_vid[idx] = cfg_v_vid_tilde - dudt_outs[0] * ts_diff_exp_vid
        u_targ_aud[idx] = cfg_v_aud_tilde - dudt_outs[1] * ts_diff_exp_aud

        return u_pred_vid, u_pred_aud, u_targ_vid, u_targ_aud

    def handle_neq_and_no_cfg_branch(
        self,
        noisy_vid, noisy_aud,
        r, t,
        u_pred_vid, u_pred_aud,
        u_targ_vid, u_targ_aud,
        v_vid, v_aud,
        has_controls, mouse, btn,
    ):
        with torch.no_grad():
            neq_mask = (r != t)

            if not neq_mask.any():
                return u_pred_vid, u_pred_aud, u_targ_vid, u_targ_aud

            cfg_mask = has_controls & (t >= self.cfg_in[0]) & (t <= self.cfg_in[1])
            neq_no_cfg_mask = neq_mask & ~cfg_mask

            if not neq_no_cfg_mask.any():
                return u_pred_vid, u_pred_aud, u_targ_vid, u_targ_aud

            idx = torch.where(neq_no_cfg_mask)[0]

            noisy_vid = noisy_vid[idx]
            noisy_aud = noisy_aud[idx]
            r = r[idx]
            t = t[idx]
            v_vid = v_vid[idx]
            v_aud = v_aud[idx]
            has_controls = has_controls[idx]
            mouse = mouse[idx]
            btn = btn[idx]
    
            ts_diff_exp_vid = (t - r)[:,:,None,None,None]
            ts_diff_exp_aud = (t - r)[:,:,None]

        def fn_2(z_vid, z_aud, curr_r, curr_t):
            return self.core(z_vid, z_aud, curr_t, mouse, btn, has_controls = has_controls, r = curr_r)
        
        primals = (noisy_vid, noisy_aud, r, t)
        tangents = (v_vid, v_aud, torch.zeros_like(r), torch.ones_like(t))
        (u_outs, dudt_outs) = torch.func.jvp(fn_2, primals, tangents)
        


        u_pred_vid[idx] = u_outs[0]
        u_pred_aud[idx] = u_outs[1]
        u_targ_vid[idx] = v_vid - dudt_outs[0] * ts_diff_exp_vid
        u_targ_aud[idx] = v_aud - dudt_outs[1] * ts_diff_exp_aud

        return u_pred_vid, u_pred_aud, u_targ_vid, u_targ_aud

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
            ts, rs = self.sample_timesteps(b, n, x.device, x.dtype)
            
            # alpha_t = (1. - t)
            # sigma_t = (t)
            # d_alpha_t = -1
            # d_sigma_t = 1

            z_video = torch.randn_like(x)
            z_audio = torch.randn_like(audio)

            ts_exp_vid = ts[:,:,None,None,None]
            ts_exp_aud = ts[:,:,None]

            noisy_vid = x * (1. - ts_exp_vid) + z_video * ts_exp_vid
            v_vid = z_video - x

            noisy_aud = audio * (1. - ts_exp_aud) + z_audio * ts_exp_aud
            v_aud = z_audio - audio

            u_targ_vid = torch.zeros_like(v_vid)
            u_targ_aud = torch.zeros_like(v_aud)

            u_pred_vid = torch.zeros_like(v_vid)
            u_pred_aud = torch.zeros_like(v_aud)

        # Handle all branches
        u_pred_video, u_pred_audio, u_targ_video, u_targ_audio = self.handle_reqt_branch(
            noisy_vid, noisy_aud,
            rs, ts,
            u_pred_vid, u_pred_aud,
            u_targ_vid, u_targ_aud,
            v_vid, v_aud,
            has_controls, mouse, btn,
        )
        u_pred_video, u_pred_audio, u_targ_video, u_targ_audio = self.handle_neq_and_cfg_branch(
            noisy_vid, noisy_aud,
            rs, ts,
            u_pred_vid, u_pred_aud,
            u_targ_vid, u_targ_aud,
            v_vid, v_aud,
            has_controls, mouse, btn,
        )
        u_pred_video, u_pred_audio, u_targ_video, u_targ_audio = self.handle_neq_and_no_cfg_branch(
            noisy_vid, noisy_aud,
            rs, ts,
            u_pred_vid, u_pred_aud,
            u_targ_vid, u_targ_aud,
            v_vid, v_aud,
            has_controls, mouse, btn,
        )
        
        u_targ_vid = u_targ_vid.detach()
        u_targ_aud = u_targ_aud.detach()

        error_video = u_pred_video - u_targ_video
        error_audio = u_pred_audio - u_targ_audio

        # Option 1: Combined error
        error = torch.cat([error_video.flatten(1), error_audio.flatten(1)], dim=1)
        error_norm = torch.norm(error, dim=1)

        # Option 2: Separate norms
        error_video_norm = torch.norm(error_video.reshape(error_video.shape[0], -1), dim=1)
        error_audio_norm = torch.norm(error_audio.reshape(error_audio.shape[0], -1), dim=1)
        loss = error_video_norm ** 2 + error_audio_norm ** 2

        return loss