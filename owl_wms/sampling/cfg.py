import torch
from torch import nn
import torch.nn.functional as F

class CFGSampler:
    @torch.no_grad()
    def __call__(self, model, dummy_batch, mouse, btn, sampling_steps = 64, decode_fn = None, scale = 1, cfg_scale = 1.3):
        x = torch.randn_like(dummy_batch)
        ts = torch.ones(x.shape[0], x.shape[1], device=x.device,dtype=x.dtype)
        dt = 1. / sampling_steps

        for _ in range(sampling_steps):
            # Get conditional prediction
            cond_pred = model(x, ts, mouse, btn)
            
            # Get unconditional prediction by zeroing out conditioning
            uncond_pred = model(x, ts, torch.zeros_like(mouse), torch.zeros_like(btn))
            
            # Combine predictions using cfg_scale
            pred = uncond_pred + cfg_scale * (cond_pred - uncond_pred)
            
            x = x - pred*dt
            ts = ts - dt

        if decode_fn is not None:
            x = x * scale 
            x = decode_fn(x)
        return x

class InpaintCFGSampler:
    @torch.no_grad()
    def __call__(self, model, dummy_batch, mouse, btn, sampling_steps = 64, decode_fn = None, scale = 1, cfg_scale = 1.5):
        x = torch.randn_like(dummy_batch)

        ts = torch.ones(x.shape[0], x.shape[1], device=x.device, dtype=x.dtype)
        dt = 1. / sampling_steps
        
        # Calculate midpoint
        mid = x.shape[1] // 2
        x[:,:mid] = dummy_batch[:,:mid]
        
        for _ in range(sampling_steps):
            # Get conditional prediction
            cond_pred = model(x, ts, mouse, btn)
            
            # Get unconditional prediction by zeroing out conditioning
            uncond_pred = model(x, ts, torch.zeros_like(mouse), torch.zeros_like(btn))
            
            # Combine predictions using cfg_scale
            pred = uncond_pred + cfg_scale * (cond_pred - uncond_pred)
            
            # Only update second half
            x[:, mid:] = x[:, mid:] - pred[:, mid:]*dt
            ts[:, mid:] = ts[:, mid:] - dt

        if decode_fn is not None:
            x = x * scale
            x = decode_fn(x)
        return x

if __name__ == "__main__":
    model = lambda x,t,m,b: x

    sampler = CFGSampler()
    x = sampler(model, torch.randn(4, 128, 16, 128), 
                torch.randn(4, 128, 2), torch.randn(4, 128, 11))
    print(x.shape)