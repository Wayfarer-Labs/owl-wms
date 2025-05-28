import torch
from torch import nn
import torch.nn.functional as F

import einops as eo

class RMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()

        # small init to default to no gain
        self.gain = nn.Parameter(torch.randn(dim) * 0.02)
    
    def forward(self, x):
        b,h,n,d = x.shape
        gain = self.gain[None,None,None,:] # [1,1,1,d]

        gain = (1. + gain)
        rms = (x.float().pow(2).mean(-1,keepdim=True)+1.0e-6).rsqrt().to(x.dtype)

        return x * rms * gain

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

        self.q_norm = RMSNorm(dim)
        self.k_norm = RMSNorm(dim)

    def forward(self, q, k):
        return self.q_norm(q), self.k_norm(k)

def LayerNorm(dim):
    return nn.LayerNorm(dim, elementwise_affine = False)