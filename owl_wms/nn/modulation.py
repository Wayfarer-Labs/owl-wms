from torch import nn
import torch.nn.functional as F

import einops as eo

from .normalization import rms_norm


class AdaLN(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.fc = nn.Linear(dim, 2 * dim, bias=False)

        # AdaLN-Zero
        self.fc.weight.detach().zero_()

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
        self.fc_c = nn.Linear(dim, dim, bias=False)

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


def ada_rmsnorm_nosink(x, scale, bias):
    x4 = eo.rearrange(x, 'b (n m) d -> b n m d', n=scale.size(1))
    y4 = rms_norm(x4) * (1 + scale.unsqueeze(2)) + bias.unsqueeze(2)
    return eo.rearrange(y4, 'b n m d -> b (n m) d')


def ada_gate_nosink(x, gate):
    x4 = eo.rearrange(x, 'b (n m) d -> b n m d', n=gate.size(1))
    return eo.rearrange(x4 * gate.unsqueeze(2), 'b n m d -> b (n m) d')


import torch


def ada_rmsnorm(x, scale, bias):
    # x: [B, S, D], scale/bias: [B, n, D] (repeat across m)
    B, S, D = x.shape
    n = scale.size(1)
    k = S % n                      # number of leading specials
    x_head = x[:, :k]              # (possibly empty)
    x_tail = x[:, k:]              # length is n*m

    if x_tail.numel() == 0:
        return x

    m = x_tail.size(1) // n
    sc = scale.repeat_interleave(m, dim=1)     # [B, n*m, D]
    bs = bias.repeat_interleave(m, dim=1)      # [B, n*m, D]

    y_tail = rms_norm(x_tail) * (1 + sc) + bs  # uses your existing rms_norm
    return torch.cat([x_head, y_tail], dim=1) if k else y_tail


def ada_gate(x, gate):
    """
    x:    [B, S, D]
    gate: [B, n, D]  (shared across the m dimension)
          Also accepts [B, n] or [B, n, 1]; will broadcast to D.
    """
    B, S, D = x.shape
    n = gate.size(1)
    k = S % n                 # number of leading specials (e.g., sink/BOS)

    # Ensure gate is [B, n, D]
    if gate.dim() == 2:
        gate = gate.unsqueeze(-1).expand(-1, -1, D)
    elif gate.size(-1) == 1:
        gate = gate.expand(-1, -1, D)

    # Split sequence
    x_head = x[:, :k]         # specials (left untouched)
    x_tail = x[:, k:]         # length is n * m

    if x_tail.numel() == 0:
        return x

    m = x_tail.size(1) // n
    gate_full = gate.repeat_interleave(m, dim=1)   # [B, n*m, D]

    y_tail = x_tail * gate_full
    return torch.cat([x_head, y_tail], dim=1) if k else y_tail
