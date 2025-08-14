import torch
from torch import nn
import torch.nn.functional as F


def layer_norm(x: torch.Tensor) -> torch.Tensor:
    return F.layer_norm(x, (x.size(-1),)).type_as(x)


def rms_norm(x: torch.Tensor) -> torch.Tensor:
    return F.rms_norm(x, (x.size(-1),))


class RMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()

        # small init to default to no gain
        self.gain = nn.Parameter(torch.randn(dim) * 0.02)
        self.eps = 1e-6

    def forward(self, x):
        # x: [b, h, n, d]
        rms = x.float().pow(2).mean(dim=-1, keepdim=True)  # [b,h,n,1]
        norm = rms.add(self.eps).rsqrt().to(x.dtype)        # stable inverse sqrt
        return x * norm * (1. + self.gain[None, None, None, :])

class L2Norm(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        b,h,n,d = x.shape
        x = F.normalize(x, dim = -1)
        return x


class QKNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()

    def forward(self, q, k):
        return rms_norm(q), rms_norm(k)
