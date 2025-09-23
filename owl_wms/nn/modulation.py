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


def cond_adaln(x, scale, bias):
    # scale,bias: [b, n, d], x: [b, n*m, d]
    b, nm, d = *x.shape[:2], x.size(-1)
    n = scale.size(1)
    m = nm // n
    # broadcast [b,n,d] → [b,n*m,d]
    scale = scale.view(b, n, 1, d).expand(-1,-1,m,-1).reshape(b, nm, d)
    bias  = bias .view(b, n, 1, d).expand(-1,-1,m,-1).reshape(b, nm, d)
    x_norm = rms_norm(x)
    return x_norm * (1 + scale) + bias

def cond_gate(x, gate):
    # gate: [b, n, d], x: [b, n*m, d]
    b, nm, d = *x.shape[:2], x.size(-1)
    n = gate.size(1)
    m = nm // n
    gate = gate.view(b, n, 1, d).expand(-1,-1,m,-1).reshape(b, nm, d)
    return gate * x
