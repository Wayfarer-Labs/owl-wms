import torch
from torch import nn
import torch.nn.functional as F

import math

import einops as eo
from owl_wms.nn.mlp import MLPCustom

from rotary_embedding_torch import (
    RotaryEmbedding,
    apply_rotary_emb
)
import einops as eo

class LearnedPosEnc(nn.Module):
    def __init__(self, n_seq, dim):
        super().__init__()

        self.n_seq = n_seq
        self.p = nn.Parameter(torch.randn(n_seq,dim)*0.02)

    def forward(self, x):
        b,n,d = x.shape
        if n < self.n_seq:
            # Only add positional embeddings for the last n tokens
            p = eo.repeat(self.p[-n:], 'n d -> b n d', b=b)
        else:
            p = eo.repeat(self.p, 'n d -> b n d', b=b)
        return x + p

class SinCosEmbed(nn.Module):
    def __init__(self, dim, theta=300, mult=1000):
        super().__init__()
        self.dim = dim
        self.theta = theta
        self.mult = mult

    def forward(self, x):
        # Handle different input types
        if isinstance(x, float):
            x = torch.tensor([x])
        elif not isinstance(x, torch.Tensor):
            x = torch.tensor(x)
        
        # Ensure x is at least 1D
        if x.dim() == 0:
            x = x.unsqueeze(0)
            
        # Handle [b,n] inputs
        reshape_out = False
        if x.dim() == 2:
            b, n = x.shape
            x = x.reshape(b*n)
            reshape_out = True
            
        x = x * self.mult
        
        half_dim = self.dim // 2
        emb = torch.log(torch.tensor(self.theta)) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim) * -emb)
        
        # Match device and dtype of input
        emb = emb.to(device=x.device, dtype=x.dtype)
        
        # Compute sin/cos embeddings
        emb = x.unsqueeze(-1) * emb.unsqueeze(0)
        emb = torch.cat((torch.sin(emb), torch.cos(emb)), dim=-1)
        
        # Reshape back if needed
        if reshape_out:
            emb = emb.reshape(b, n, -1)
            
        return emb

class TimestepEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.sincos = SinCosEmbed(512, theta=300, mult = 1000)
        self.mlp = MLPCustom(512, dim * 4, dim)
    
    def forward(self, x):
        x = self.sincos(x)
        x = self.mlp(x)
        return x

class StepEmbedding(nn.Module):
    def __init__(self, d_out, d_in=512, max_steps=128):
        super().__init__()

        self.mlp = MLPCustom(d_in, dim_middle = 4 * d_out, dim_out=d_out)
        self.max_steps = max_steps
        mult = 1000 / math.log2(max_steps)
        self.sincos = SinCosEmbed(d_in, theta=300, mult=mult)

    def forward(self, steps):
        if not isinstance(steps, torch.Tensor):
            steps = torch.tensor(steps, device=self.mlp.fc_uv.weight.device, dtype=self.mlp.fc_uv.weight.dtype)
        if steps.ndim == 0:
            steps = steps.unsqueeze(0)

        # Map steps to [0, log2(max_steps)]
        t = (math.log2(self.max_steps) - torch.log2(steps.float())).to(steps.dtype)
        embs = self.sincos(t)
        return self.mlp(embs)

class ConditionEmbedding(nn.Module):
    def __init__(self, n_classes, dim):
        super().__init__()
        
        self.embedding = nn.Embedding(n_classes, dim)
        self.mlp = MLPCustom(dim, dim * 4, dim)
    
    def forward(self, x):
        # x is long tensor of [b,]
        x = self.embedding(x)
        x = self.mlp(x)
        return x

class MouseEmbedding(nn.Module):
    def __init__(self, dim_out, dim=512):
        super().__init__()
        
        # For angle embeddings
        self.angle_proj = nn.Linear(2, dim//2, bias=False)
        
        # For magnitude embeddings
        self.magnitude_embed = SinCosEmbed(dim//2)
        
        # Final MLP
        self.mlp = MLPCustom(dim, dim * 4, dim_out)

    def forward(self, x):
        # x is [b,n,2]
        # Convert to polar coordinates
        with torch.no_grad():
            # Apply symlog scaling to x and y coordinates
            x_sign = torch.sign(x)
            x_abs = torch.abs(x)
            x = x_sign * torch.log1p(x_abs)

            angles = torch.atan2(x[..., 1], x[..., 0])  # [b,n]
            magnitudes = torch.norm(x, dim=-1)  # [b,n]
            
            # Embed angles and magnitudes
            angle_emb = torch.stack([
                torch.cos(angles),
                torch.sin(angles)
            ], dim=-1)  # [b,n,2]
            magnitude_emb = self.magnitude_embed(magnitudes)  # [b,n,dim//2]

        angle_emb = self.angle_proj(angle_emb)  # [b,n,dim//2]
        
        # Combine and pass through MLP
        x = torch.cat([angle_emb, magnitude_emb], dim=-1)  # [b,n,dim]
        x = self.mlp(x)
        return x

class ButtonEmbeddding(nn.Module):
    def __init__(self, n_buttons, dim_out, dim=512):
        super().__init__()

        self.proj = MLPCustom(n_buttons, dim*4, dim_out)
    
    def forward(self, x):
        # x is float tensor of 0s and 1s
        x = (x * 2) - 1
        x = self.proj(x)
        return x

class ControlEmbedding(nn.Module):
    def __init__(self, n_buttons, dim_out, dim = 512):
        super().__init__()

        self.mouse = MouseEmbedding(dim_out, dim)
        self.button = ButtonEmbeddding(n_buttons, dim_out, dim)

    def forward(self, mouse, button):
        # mouse : [b,n,2]
        # button : [b,n,n_buttons]

        # out is [b,n,d]

        return self.mouse(mouse) + self.button(button)
