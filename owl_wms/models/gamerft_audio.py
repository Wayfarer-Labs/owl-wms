"""
GameRFT with Audio
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
from ..nn.mmattn_v2 import MMDIT2

class GameRFTAudioCore(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config

        if config.backbone == 'dit':
            backbone_cls = DiT
        elif config.backbone == 'mmdit':
            backbone_cls = MMDIT
        elif config.backbone == 'mmdit_2':
            backbone_cls = MMDIT2
            config.backbone = 'mmdit'
        elif config.backbone == 'uvit':
            backbone_cls = UViT
        else:
            raise ValueError(f"Invalid backbone: {config.backbone}")

        self.backbone = config.backbone

        self.transformer = backbone_cls(config)

        if not config.uncond: self.control_embed = ControlEmbedding(config.n_buttons, config.d_model)
        self.t_embed = TimestepEmbedding(config.d_model)

        self.proj_in = nn.Linear(config.channels, config.d_model, bias = False)
        self.proj_out = FinalLayer(config.sample_size, config.d_model, config.channels)

        self.audio_proj_in = nn.Linear(config.audio_channels, config.d_model, bias=False)
        self.audio_proj_out = FinalLayer(None, config.d_model, config.audio_channels)

        self.uncond = config.uncond

    def forward(self, x, audio, t, mouse, btn, has_controls = None, kv_cache = None):
        # x is [b,n,c,h,w]
        # audio is [b,n,c]
        # t is [b,n]
        # mouse is [b,n,2]
        # btn is [b,n,n_buttons]

        t_cond = self.t_embed(t)

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

class GameRFTAudio(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.core = GameRFTAudioCore(config)
        self.cfg_prob = config.cfg_prob
    
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

        if has_controls is None:
            has_controls = torch.ones(b, device=x.device, dtype=torch.bool)

        # Apply classifier-free guidance dropout
        has_controls = self.handle_cfg(has_controls, cfg_prob)
        with torch.no_grad():
            ts = torch.randn(b,n,device=x.device,dtype=x.dtype).sigmoid()
            
            # Video noise
            ts_exp = ts[:, :, None, None, None]
            z_video = torch.randn_like(x)
            lerpd_video = x * (1. - ts_exp) + z_video * ts_exp
            target_video = z_video - x

            # Audio noise
            ts_exp_audio = ts.unsqueeze(-1)
            z_audio = torch.randn_like(audio)
            lerpd_audio = audio * (1. - ts_exp_audio) + z_audio * ts_exp_audio
            target_audio = z_audio - audio
            
        pred_video, pred_audio = self.core(lerpd_video, lerpd_audio, ts, mouse, btn, has_controls)
        video_loss = F.mse_loss(pred_video, target_video)
        audio_loss = F.mse_loss(pred_audio, target_audio)
        diff_loss = video_loss + audio_loss

        if not return_dict:
            return diff_loss
        else:
            return {
                'diffusion_loss': diff_loss,
                'video_loss': video_loss,
                'audio_loss': audio_loss,
                'lerpd_video': lerpd_video,
                'lerpd_audio': lerpd_audio,
                'pred_video': pred_video,
                'pred_audio': pred_audio,
                'ts': ts,
                'z_video': z_video,
                'z_audio': z_audio,
                'cfg_mask' : has_controls 
            }

if __name__ == "__main__":
    from ..configs import Config

    cfg = Config.from_yaml("configs/basic.yml").model
    model = GameRFT(cfg).cuda().bfloat16()

    with torch.no_grad():
        x = torch.randn(1, 128, 16, 256, device='cuda', dtype=torch.bfloat16)
        mouse = torch.randn(1, 128, 2, device='cuda', dtype=torch.bfloat16) 
        btn = torch.randn(1, 128, 11, device='cuda', dtype=torch.bfloat16)
        
        loss = model(x, mouse, btn)
        print(f"Loss: {loss.item()}")
