"""
Most basic model.
"""

import torch
from torch import nn
import torch.nn.functional as F

import einops as eo

from owl_wms.nn.embeddings import (
    TimestepEmbedding,
    ControlEmbedding,
    LearnedPosEnc
)
from owl_wms.nn.attn import UViT, FinalLayer

class GameRFTCore(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.transformer = UViT(config)
        self.control_embed = ControlEmbedding(config.n_buttons, config.d_model)
        self.t_embed = TimestepEmbedding(config.d_model)

        self.proj_in = nn.Linear(config.channels, config.d_model, bias = False)
        self.proj_out = FinalLayer(config.sample_size, config.d_model, config.channels)

        self.pos_enc = LearnedPosEnc(config.tokens_per_frame * config.n_frames, config.d_model)

    def forward(self, x, t, mouse, btn):
        # x is [b,n,c,h,w]
        # t is [b,n]
        # mouse is [b,n,2]
        # btn is [b,n,n_buttons]

        ctrl_cond = self.control_embed(mouse, btn)
        t_cond = self.t_embed(t)

        cond = ctrl_cond + t_cond # [b,n,d]
        
        b,n,c,h,w = x.shape
        x = eo.rearrange(x, 'b n c h w -> b (n h w) c')

        x = self.proj_in(x)
        x = self.pos_enc(x)
        x = self.transformer(x, cond)
        x = self.proj_out(x, cond) # -> [b,n*hw,c]
        x = eo.rearrange(x, 'b (n h w) c -> b n c h w', n=n,h=h,w=w)

        return x

class GameRFT(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.core = GameRFTCore(config)
        self.cfg_prob = config.cfg_prob
    
    def forward(self, x, mouse, btn, return_dict = False, cfg_prob = None):
        # x is [b,n,c,h,w]
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
            
            ts_exp = eo.repeat(ts, 'b n -> b n 1 1 1')
            z = torch.randn_like(x)

            lerpd = x * (1. - ts_exp) + z * ts_exp
            target = z - x
        
        pred = self.core(lerpd, ts, mouse, btn)
        diff_loss = F.mse_loss(pred, target)

        if not return_dict:
            return diff_loss
        else:
            return {
                'diffusion_loss' : diff_loss,
                'lerpd' : lerpd, 
                'pred' : pred,
                'ts': ts,
                'z': z
            }

if __name__ == "__main__":
    from owl_wms.configs import Config

    cfg = Config.from_yaml("configs/basic.yml").model
    model = GameRFT(cfg).cuda().bfloat16()

    with torch.no_grad():
        x = torch.randn(1, 128, 16, 256, device='cuda', dtype=torch.bfloat16)
        mouse = torch.randn(1, 128, 2, device='cuda', dtype=torch.bfloat16) 
        btn = torch.randn(1, 128, 11, device='cuda', dtype=torch.bfloat16)
        
        loss = model(x, mouse, btn)
        print(f"Loss: {loss.item()}")