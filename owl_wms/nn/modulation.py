import torch
from torch import nn
import torch.nn.functional as F

from .normalization import LayerNorm

class AdaLN(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.fc = nn.Linear(dim, 2 * dim)
        self.norm = LayerNorm(dim)

    def forward(self, x, cond):
        # cond is [b,n,d]
        # x is [b,n*m,d] where m is tokens per frame
        b,n,d = cond.shape
        _,nm,_ = x.shape
        m = nm // n

        y = F.silu(cond)
        ab = self.fc(y) # [b,n,d]
        ab = ab[:,:,None,:].repeat(1,1,m,1)
        ab = ab.reshape(b,nm,2*d)
        a,b = ab.chunk(2,dim=-1) # each [b,nm,d]

        x = self.norm(x) * (1. + a) + b
        return x

class Gate(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.fc_c = nn.Linear(dim, dim)

    def forward(self, x, cond):
        # cond is [b,n,d] x is [b,nm,d]
        b,n,d = cond.shape
        _,nm,_ = x.shape
        m = nm//n

        y = F.silu(cond)
        c = self.fc_c(y) # [b,n,d]
        c = c[:,:,None,:].repeat(1,1,m,1)
        c = c.reshape(b,nm,d)

        return c * x