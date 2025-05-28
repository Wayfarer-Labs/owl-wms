import torch
from torch import nn
import torch.nn.functional as F

class SimpleSampler:
    @torch.no_grad()
    def __call__(self, model, dummy_batch, mouse, btn, sampling_steps = 64, decode_fn = None, scale = 1):
        x = torch.randn_like(dummy_batch)
        ts = torch.ones(x.shape[0], device=x.device,dtype=x.dtype)
        dt = 1. / sampling_steps

        for _ in range(sampling_steps):
            pred = model(x, ts, mouse, btn)
            x = x - pred*dt
            ts = ts - dt

        if decode_fn is not None:
            x = x * scale
            x = decode_fn(x)
        return x

if __name__ == "__main__":
    model = lambda x,t,m,b: x

    sampler = Sampler()
    x = sampler(model, torch.randn(4, 3, 64, 64), 
                torch.randn(4, 2), torch.randn(4, 8))
    print(x.shape)