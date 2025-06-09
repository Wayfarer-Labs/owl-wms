import torch
from torch import nn
import torch.nn.functional as F

class CFGSampler:
    def __init__(self, n_steps = 20, cfg_scale = 1.3):
        self.n_steps = n_steps
        self.cfg_scale = cfg_scale

    @torch.no_grad()
    def __call__(self, model, dummy_batch, mouse, btn, decode_fn = None, scale = 1):
        sampling_steps = self.n_steps
        cfg_scale = self.cfg_scale
        
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
        return x, mouse, btn

class InpaintCFGSampler(CFGSampler):
    @torch.no_grad()
    def __call__(self, model, dummy_batch, mouse, btn, decode_fn = None, scale = 1):
        sampling_steps = self.n_steps
        cfg_scale = self.cfg_scale
        
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
        return x, mouse, btn


def zlerp(x, alpha):
    z = torch.randn_like(x)
    return x * (1. - alpha) + z * alpha


class WindowCFGSampler:
    @torch.no_grad()
    def __call__(self, model, dummy_batch, mouse, btn,
                 sampling_steps = 64, decode_fn = None, scale = 1, cfg_scale = 1.3,
                 window_length = 60, num_frames = 60, noise_prev = 0.2):
        
        x = torch.randn_like(dummy_batch)
        ts = torch.ones(x.shape[0], x.shape[1], device=x.device,dtype=x.dtype)
        dt = 1. / sampling_steps

        clean_history = dummy_batch.clone()
        
        def step_history():
            new_history = clean_history.clone()[:,-window_length:] # last 60 frames
            b,n,c,h,w = new_history.shape
            # Shift over by 1: 
            #                        [ 0 1 2 3 4 ]
            # reassigned to ->   zlerp([ 1 2 3 4 ]) + [ randn ]
            new_history[:,:-1] = zlerp(new_history[:,1:],noise_prev) # pop off first frame and noise context
            new_history[:,-1] = torch.randn(b,1,c,h,w) # Add noise to last
            return new_history

        for _ in range(num_frames):
            local_history = step_history()
            ts_history = torch.ones(local_history.shape[0], local_history.shape[1], device=x.device,dtype=x.dtype)
            ts_history[:,-1] = noise_prev

            for _ in range(sampling_steps):
                # CFG Branches
                x = local_history.clone()
                ts = ts_history.clone()
                cond_pred = model(x, ts, mouse, btn)
                uncond_pred = model(x, ts, torch.zeros_like(mouse), torch.zeros_like(btn))
                pred = uncond_pred + cfg_scale * (cond_pred - uncond_pred)
                
                x = x - pred*dt
                ts = ts - dt

                local_history[:,-1] = x[:,-1]
                ts_history[:,-1]    = ts[:,-1]
            
            # Frame is entirely cleaned now
            new_frame = local_history[:,-1:]
            clean_history = torch.cat([clean_history, new_frame], dim = 1)

        x = clean_history
        pixels = None

        if decode_fn is not None: 
            pixels = decode_fn(x * scale)
        return x, pixels
    

if __name__ == "__main__":
    model = lambda x,t,m,b: x

    sampler = CFGSampler()
    x, pixels = sampler(model, torch.randn(4, 128, 16, 128), 
                torch.randn(4, 128, 2), torch.randn(4, 128, 11))
    print(x.shape)