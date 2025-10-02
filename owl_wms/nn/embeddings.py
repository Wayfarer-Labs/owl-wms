import torch
from torch import nn
import math

from .mlp import MLPCustom


class NoiseConditioner(nn.Module):
    """Sigma -> logSNR -> Fourier Features -> Dense"""
    def __init__(self, dim, fourier_dim=512, base=10_000.0):
        super().__init__()
        assert fourier_dim % 2 == 0
        half = fourier_dim // 2
        self.freq = nn.Buffer(torch.logspace(0, -1, steps=half, base=base, dtype=torch.float32), persistent=False)
        self.mlp = MLPCustom(fourier_dim, dim * 4, dim)

    def forward(self, s, eps=torch.finfo(torch.float32).eps):
        assert self.freq.dtype == torch.float32
        orig_dtype, shape = s.dtype, s.shape

        with torch.autocast("cuda", enabled=False):
            s = s.reshape(-1).float()  # fp32 for fourier numerical stability
            s = s * 1000  # expressive rotation range

            # calculate fourier features
            phase = s[:, None] * self.freq[None, :]
            emb = torch.cat((torch.sin(phase), torch.cos(phase)), dim=-1)
            emb = emb * math.sqrt(2)  # Ensure unit variance

            emb = self.mlp(emb)
            return emb.view(*shape, -1).to(orig_dtype)


class SinCosEmbed(nn.Module):
    def __init__(self, dim, theta=300, mult=1000):
        super().__init__()
        self.dim = dim
        self.theta = theta
        self.mult = mult

    @torch.autocast("cuda", enabled=False)
    def forward(self, x):
        orig_dtype = x.dtype
        x = x.float()
        # Handle different input types
        if isinstance(x, float):
            x = torch.tensor([x], dtype=torch.float32)
        elif not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32)

        # Ensure x is at least 1D
        if x.dim() == 0:
            x = x.unsqueeze(0)

        # Handle [b,n] inputs
        reshape_out = False
        if x.dim() == 2:
            b, n = x.shape
            x = x.view(b*n)
            reshape_out = True

        x = x * self.mult

        half_dim = self.dim // 2
        emb = torch.log(torch.tensor(self.theta, dtype=torch.float32)) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim) * -emb)

        # Match device and dtype of input
        emb = emb.to(device=x.device, dtype=x.dtype)

        # Compute sin/cos embeddings
        emb = x.unsqueeze(-1) * emb.unsqueeze(0)
        emb = torch.cat((torch.sin(emb), torch.cos(emb)), dim=-1)

        # Reshape back if needed
        if reshape_out:
            emb = emb.reshape(b, n, -1)

        return emb.to(orig_dtype)


class TimestepEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.sincos = SinCosEmbed(512, theta=300, mult = 1000)
        self.mlp = MLPCustom(512, dim * 4, dim)

    def forward(self, x):
        x = self.sincos(x)
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
            ], dim=-1).to(x.dtype)  # [b,n,2]
            magnitude_emb = self.magnitude_embed(magnitudes).to(x.dtype)  # [b,n,dim//2]

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

    def forward(self, mouse, button, has_controls=None):
        # mouse : [b,n,2]
        # button : [b,n,n_buttons]
        # has_controls : [b,] boolean mask

        # out is [b,n,d]

        return self.mouse(mouse) + self.button(button)
