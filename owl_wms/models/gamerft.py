"""
Most basic model.
"""

import torch
from torch import nn
import torch.nn.functional as F

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
        x = self.proj_out(x, cond)

        return x

class GameRFT(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.core = GameRFTCore(config)
    
    def forward(self, x, mouse, btn):
        # x is [b,n,m,d]
        # mouse is [b,n,2]
        # btn is [b,n,n_buttons]

        b,n,m,d = x.shape
        with torch.no_grad():
            ts = torch.randn(b,n,device=x.device,dtype=x.dtype).sigmoid()
            
            ts_exp = eo.repeat(ts, 'b n -> b n m d',m=m,d=d)
            z = torch.randn_like(x)

            lerpd = x * (1. - ts_exp) + z * ts_exp
            target = z - x
        
        pred = self.core(lerpd, ts, mouse, btn)
        diff_loss = F.mse_loss(pred, target)

        return diff_loss

        
        

