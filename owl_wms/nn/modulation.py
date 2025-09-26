from torch import nn
import torch.nn.functional as F

from .normalization import rms_norm


class AdaLN(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.fc = nn.Linear(dim, 2 * dim)

    def forward(self, x, cond):
        # cond: [b, n, d], x: [b, n*m, d]
        b, n, d = cond.shape
        _, nm, _ = x.shape
        m = nm // n

        y = F.silu(cond)
        ab = self.fc(y)                    # [b, n, 2d]
        ab = ab.view(b, n, 1, 2*d)         # [b, n, 1, 2d]
        ab = ab.expand(-1, -1, m, -1)      # [b, n, m, 2d]
        ab = ab.reshape(b, nm, 2*d)        # [b, nm, 2d]

        a, b_ = ab.chunk(2, dim=-1)        # [b, nm, d] each
        x = rms_norm(x) * (1 + a) + b_
        return x


class Gate(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.fc_c = nn.Linear(dim, dim)

    def forward(self, x, cond):
        # cond: [b, n, d], x: [b, n*m, d]
        b, n, d = cond.shape
        _, nm, _ = x.shape
        m = nm // n

        y = F.silu(cond)
        c = self.fc_c(y)                  # [b, n, d]
        c = c.view(b, n, 1, d).expand(-1, -1, m, -1).reshape(b, nm, d)

        return c * x


class FinalLayer(nn.Module):
    def __init__(self, d_model, channels, patch_size=1):
        super().__init__()
        self.norm = AdaLN(d_model)
        self.act = nn.SiLU()
        self.proj = nn.Linear(d_model, channels * patch_size * patch_size)

    def forward(self, x, cond):
        x = self.norm(x, cond)
        return self.proj(F.silu(x))
