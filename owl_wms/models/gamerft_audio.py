"""
GameRFT with Audio
"""

import torch
from torch import nn
import torch.nn.functional as F
from torch import Tensor
import einops as eo

from owl_wms.nn.embeddings import (
    TimestepEmbedding,
    ControlEmbedding,
    LearnedPosEnc
)
from owl_wms.configs import TransformerConfig
from owl_wms.nn.attn import DiT, FinalLayer
from owl_wms.nn.kv_cache import KVCache

class GameRFTAudioCore(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()

        self.transformer = DiT(config)
        self.control_embed = ControlEmbedding(config.n_buttons, config.d_model)
        self.t_embed = TimestepEmbedding(config.d_model)

        self.proj_in = nn.Linear(config.channels, config.d_model, bias = False)
        self.proj_out = FinalLayer(config.sample_size, config.d_model, config.channels)

        self.audio_proj_in = nn.Linear(config.audio_channels, config.d_model, bias=False)
        self.audio_proj_out = FinalLayer(None, config.d_model, config.audio_channels)

        self.pos_enc = LearnedPosEnc(config.tokens_per_frame * config.n_frames, config.d_model)

    def forward(self, x, audio, t, mouse, btn, kv_cache = None, additive_attn_mask = None) -> tuple[Tensor, Tensor]:
        # x is [b,n,c,h,w]
        # audio is [b,n,c]
        # t is [b,n]
        # mouse is [b,n,2]
        # btn is [b,n,n_buttons]
        ctrl_cond = self.control_embed(mouse, btn)
        t_cond = self.t_embed(t)

        cond = ctrl_cond + t_cond # [b,n,d]
        
        b,n,c,h,w = x.shape
        x = eo.rearrange(x, 'b n c h w -> b (n h w) c')

        x = self.proj_in(x)
        audio = self.audio_proj_in(audio).unsqueeze(-2) # [b,n,1,d]

        x = eo.rearrange(x, 'b (n f) d -> b n f d', n = n)
        x = torch.cat([x, audio], dim = -2)
        x = eo.rearrange(x, 'b n f d -> b (n f) d')

        x = self.pos_enc(x)
        x = self.transformer(x, cond, kv_cache, additive_attn_mask=additive_attn_mask)

        # Split into video and audio tokens
        x = eo.rearrange(x, 'b (n f) d -> b n f d', n=n)
        video, audio = x[...,:-1,:], x[...,-1:,:]

        # Project video tokens
        video = eo.rearrange(video, 'b n f d -> b (n f) d')
        video = self.proj_out(video, cond)
        video = eo.rearrange(video, 'b (n h w) c -> b n c h w', n=n, h=h, w=w)

        # Project audio tokens
        audio = eo.rearrange(audio, 'b n 1 d -> b n d')
        audio = self.audio_proj_out(audio, cond)
        return video, audio

class GameRFTAudio(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config
        self.core = GameRFTAudioCore(config)
        self.cfg_prob = config.cfg_prob
    
    def velocity_fn(self,
                x_t:   torch.Tensor,       # [B, N, C, H, W] noisy video latents
                t:     torch.Tensor,       # [B, N] or scalar
                mouse: torch.Tensor | None = None,
                btn:   torch.Tensor | None = None,
                audio: torch.Tensor | None = None,
                kv_cache: KVCache   | None = None,
                cfg_weight: float          = 0.0) -> tuple[Tensor, Tensor]:
        """
        Return velocities with optional CFG
        """
        B, N, *_   = x_t.shape
        mouse_null = torch.zeros(B, N, 2, device=x_t.device, dtype=x_t.dtype)
        btn_null   = torch.zeros(B, N, self.config.n_buttons, device=x_t.device, dtype=x_t.dtype)

        mouse = mouse if mouse is not None else mouse_null
        btn   = btn   if btn   is not None else btn_null

        # NOTE DO NOT DO THIS - because it messes with the batch dimension of the KV-Cache
        # and causes it to error out. If cfg==0.0 it will be handled just fine with the 
        # existing logic. This is wrong, for now:
        # if cfg_weight == 0.0:
        #     return self.core(x_t, audio, t, mouse, btn, kv_cache)
        
        velocity_video, velocity_audio = self.core(
            x        = torch.cat([x_t,   x_t],   dim=0),
            audio    = torch.cat([audio, audio], dim=0),
            t        = torch.cat([t, t], dim=0),
            mouse    = torch.cat([mouse, mouse_null], dim=0),
            btn      = torch.cat([btn, btn_null], dim=0),
            kv_cache = kv_cache,
        )

        velocity_video_cond, velocity_video_uncond = velocity_video.chunk(2)
        velocity_audio_cond, velocity_audio_uncond = velocity_audio.chunk(2)

        velocity_video = (velocity_video_uncond + cfg_weight) * (velocity_video_cond - velocity_video_uncond)
        velocity_audio = (velocity_audio_uncond + cfg_weight) * (velocity_audio_cond - velocity_audio_uncond)
        
        return velocity_video, velocity_audio


    def forward(self, x, audio, mouse, btn, return_dict = False, cfg_prob = None, additive_attn_mask = None):
        # TODO SAMI - I am a bit confused about the shapes here.
        # in causvid, we take in an entire sequence of 'noise' and generate the next sequence.
        # this sequence is from a sliding window, where the last frame is pure noise and the prior frames
        # are noised clean frames. Therefore, N > 1 because we generate multiple frames at a time.
        # However, with self-forcing, this is different because we generate only 1 frame, however,
        # the conditioning are clean frames that were generated from autoregression. However, I see 
        # that we pass in N = 1 - is this correct? Or should N = num_frames still, except 0:N-1 be fetched from
        # KV Cache?

        # x is [b,n,c,h,w]
        # audio is [b,n,c]
        # mouse is [b,n,2]
        # btn is [b,n,n_buttons]
        b,n,c,h,w = x.shape

        # Apply classifier-free guidance dropout
        if cfg_prob is None:
            cfg_prob = self.cfg_prob
        if cfg_prob > 0.0:
            mask = torch.rand(b, device=x.device) <= self.cfg_prob
            null_mouse = torch.zeros_like(mouse)
            null_btn = torch.zeros_like(btn)
            
            # Where mask is True, replace with zeros
            mouse = torch.where(mask.unsqueeze(-1).unsqueeze(-1), null_mouse, mouse)
            btn = torch.where(mask.unsqueeze(-1).unsqueeze(-1), null_btn, btn)
        
        with torch.no_grad():
            ts = torch.randn(b,n,device=x.device,dtype=x.dtype).sigmoid()
            
            # Video noise
            ts_exp = eo.repeat(ts, 'b n -> b n 1 1 1')
            z_video = torch.randn_like(x)
            lerpd_video = x * (1. - ts_exp) + z_video * ts_exp
            target_video = z_video - x

            # Audio noise
            ts_exp_audio = ts.unsqueeze(-1)
            z_audio = torch.randn_like(audio)
            # calculates rectified flow velocity, distance from noise to clean
            lerpd_audio = audio * (1. - ts_exp_audio) + z_audio * ts_exp_audio
            target_audio = z_audio - audio
        
        # NOTE this is the velocity not the actual video
        pred_video, pred_audio = self.core(lerpd_video, lerpd_audio, ts, mouse, btn, additive_attn_mask)
        # F.mse_loss(denoise(apply_noise(randn, sample)) (randn - sample))
        video_loss = F.mse_loss(pred_video, target_video)
        audio_loss = F.mse_loss(pred_audio, target_audio)
        diff_loss = video_loss + audio_loss

        if not return_dict:  # this trains with Rectified Flow (matching velocities) in AVRFTTrainer
            return diff_loss
        else:
            return {  # this is for CausVid, it fetches lerpd (noise) and pred (velocity?)
                'diffusion_loss': diff_loss,
                'video_loss': video_loss,
                'audio_loss': audio_loss,
                'lerpd_video': lerpd_video,
                'lerpd_audio': lerpd_audio,
                'pred_video': pred_video,
                'pred_audio': pred_audio,
                'ts': ts,
                'z_video': z_video,
                'z_audio': z_audio
            }

if __name__ == "__main__":
    from owl_wms.configs import Config

    cfg = Config.from_yaml("configs/basic.yml").model
    model = GameRFTAudio(cfg).cuda().bfloat16()

    with torch.no_grad():
        x = torch.randn(1, 128, 16, 256, device='cuda', dtype=torch.bfloat16)
        mouse = torch.randn(1, 128, 2, device='cuda', dtype=torch.bfloat16) 
        btn = torch.randn(1, 128, 11, device='cuda', dtype=torch.bfloat16)
        
        loss = model(x, mouse, btn)
        print(f"Loss: {loss.item()}")
