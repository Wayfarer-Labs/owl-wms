"""
Most basic model.
"""

import torch
from torch import nn
import torch.nn.functional as F

import einops as eo

from ..nn.embeddings import (
    TimestepEmbedding,
    ControlEmbedding
)
from ..nn.attn import UViT, ProjOut

class GameRFTCore(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.transformer = UViT(config)
        self.control_embed = ControlEmbedding(config.n_buttons, config.d_model)
        self.t_embed = TimestepEmbedding(config.d_model)

        self.proj_in = nn.Linear(config.channels, config.d_model, bias = False)
        self.proj_out = ProjOut(config.d_model, config.channels)

    def forward(self, x, t, mouse, btn):
        # x is [b,n,m,d]
        # t is [b,n]
        # mouse is [b,n,2]
        # btn is [b,n,n_buttons]

        ctrl_cond = self.control_embed(mouse, btn)
        t_cond = self.t_embed(t)
        cond = ctrl_cond + t_cond # [b,n,d]

        # x is [b,n_frames,n,c]
        b,n,m,c = x.shape
        x = x.view(b,n*m,c) # Flatten into sequence

        x = self.proj_in(x)
        x = self.transformer(x, cond)
        x = self.proj_out(x, cond) # -> [b,n*m,c]
        x = x.view(b,n,m,c)

        return x

class GameRFT(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.core = GameRFTCore(config)
        self.cfg_prob = config.cfg_prob
    
    def forward(self, x, mouse, btn):
        # x is [b,n,m,d]
        # mouse is [b,n,2]
        # btn is [b,n,n_buttons]
        b,n,m,d = x.shape

        # Apply classifier-free guidance dropout
        if self.cfg_prob > 0.:
            mask = torch.rand(b, device=x.device) < self.cfg_prob
            null_mouse = torch.zeros_like(mouse)
            null_btn = torch.zeros_like(btn)
            
            # Where mask is True, replace with zeros
            mouse = torch.where(mask.unsqueeze(-1).unsqueeze(-1), null_mouse, mouse)
            btn = torch.where(mask.unsqueeze(-1).unsqueeze(-1), null_btn, btn)
        
        with torch.no_grad():
            ts = torch.randn(b,n,device=x.device,dtype=x.dtype).sigmoid()
            
            ts_exp = eo.repeat(ts, 'b n -> b n m d',m=m,d=d)
            z = torch.randn_like(x)

            lerpd = x * (1. - ts_exp) + z * ts_exp
            target = z - x
        
        pred = self.core(lerpd, ts, mouse, btn)
        diff_loss = F.mse_loss(pred, target)

        return diff_loss

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