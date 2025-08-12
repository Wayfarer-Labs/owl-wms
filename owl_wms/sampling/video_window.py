import torch
from torch import nn
import torch.nn.functional as F

from tqdm import tqdm

from ..nn.kv_cache import KVCache
from .schedulers import get_sd3_euler

def zlerp(x, alpha):
    z = torch.randn_like(x)
    return x * (1. - alpha) + z * alpha

class VideoWindowSampler:
    """
    Window CFG Sampler samples new frames one by one, by inpainting the final frame.
    This is basically diffusion forcing.

    :param n_steps: Number of diffusion steps for each frame (diffusoin steps)
    :param cfg_scale: CFG scale for each frame
    :param window_length: Number of frames to use for each frame generation step
    :param num_frames: Number of new frames to sample
    :param noise_prev: Noise previous frame
    """
    def __init__(self, n_steps = 20, cfg_scale = 1.3, window_length = 60, num_frames = 60, noise_prev = 0.2):
        self.n_steps = n_steps
        self.cfg_scale = cfg_scale
        self.window_length = window_length
        self.num_frames = num_frames
        self.noise_prev = noise_prev

    @torch.no_grad()
    def __call__(self, model, dummy_batch, mouse, btn, decode_fn=None, image_scale=1):
        # dummy_batch is [b,n,c,h,w]
        # mouse is [b,n,2]
        # btn is [b,n,n_button]

        # output will be [b,n+self.num_frames,c,h,w]

        sampling_steps = self.n_steps
        num_frames = self.num_frames

        dt = get_sd3_euler(self.n_steps)

        clean_history = dummy_batch.clone()

        def step_history():
            # Video history
            new_history = clean_history.clone()[:,-self.window_length:]
            b,n,c,h,w = new_history.shape
            new_history[:,:-1] = zlerp(new_history[:,1:],self.noise_prev)
            new_history[:,-1] = torch.randn_like(new_history[:,0])

            return new_history

        for frame_idx in tqdm(range(num_frames)):
            local_history = step_history()
            ts_history = torch.ones(local_history.shape[0], local_history.shape[1], device=local_history.device,dtype=local_history.dtype)
            ts_history[:,:-1] = self.noise_prev

            mouse_step = mouse[:,frame_idx:frame_idx+self.window_length]
            btn_step = btn[:,frame_idx:frame_idx+self.window_length]

            # Create masks for conditional and unconditional branches
            b = local_history.shape[0]
            uncond_mask = torch.zeros(b, dtype=torch.bool, device=local_history.device)
            cond_mask = torch.ones(b, dtype=torch.bool, device=local_history.device)

            for step_idx in range(sampling_steps):
                x = local_history.clone()
                ts = ts_history.clone()

                # Get unconditional predictions
                pred_video_uncond = model(x, ts, mouse_step, btn_step, has_controls=uncond_mask)

                # Get conditional predictions
                pred_video_cond = model(x, ts, mouse_step, btn_step, has_controls=cond_mask)

                # Apply CFG
                pred_video = pred_video_uncond + self.cfg_scale * (pred_video_cond - pred_video_uncond)

                x = x - pred_video*dt[step_idx]
                ts = ts - dt[step_idx]

                local_history[:,-1] = x[:,-1]
                ts_history[:,-1] = ts[:,-1]

            # Frame is entirely cleaned now
            new_frame = local_history[:,-1:]
            clean_history = torch.cat([clean_history, new_frame], dim=1)

        return clean_history

class CausalAVWindowSampler:
    """
    Window CFG Sampler samples new frames one by one, by inpainting the final frame.
    This is basically diffusion forcing.

    :param n_steps: Number of diffusion steps for each frame (diffusoin steps)
    :param cfg_scale: CFG scale for each frame
    :param window_length: Number of frames to use for each frame generation step
    :param num_frames: Number of new frames to sample
    :param noise_prev: Noise previous frame
    """
    def __init__(self, n_steps = 20, cfg_scale = 1.3, window_length = 60, num_frames = 60, noise_prev = 0.2):
        self.n_steps = n_steps
        self.cfg_scale = cfg_scale
        self.window_length = window_length
        self.num_frames = num_frames
        self.noise_prev = noise_prev

    @torch.no_grad()
    def __call__(self, model, dummy_batch, mouse, btn, decode_fn=None, image_scale=1):
        # dummy_batch is [b,n,c,h,w]
        # mouse is [b,n,2]
        # btn is [b,n,n_button]

        # output will be [b,n+self.num_frames,c,h,w]

        sampling_steps = self.n_steps
        num_frames = self.num_frames

        cache_cond = KVCache(model.config)
        cache_uncond = KVCache(model.config)

        dt = get_sd3_euler(self.n_steps)

        clean_history = dummy_batch.clone()

        def step_history():
            # Video history
            new_history = clean_history.clone()[:,-self.window_length:]
            b,n,c,h,w = new_history.shape
            new_history[:,:-1] = zlerp(new_history[:,1:],self.noise_prev)
            new_history[:,-1] = torch.randn_like(new_history[:,0])

            return new_history

        for frame_idx in tqdm(range(num_frames)):
            local_history = step_history()
            ts_history = torch.ones(local_history.shape[0], local_history.shape[1], device=local_history.device,dtype=local_history.dtype)
            ts_history[:,:-1] = self.noise_prev

            mouse_step = mouse[:,frame_idx:frame_idx+self.window_length]
            btn_step = btn[:,frame_idx:frame_idx+self.window_length]

            # Create masks for conditional and unconditional branches
            b = local_history.shape[0]
            uncond_mask = torch.zeros(b, dtype=torch.bool, device=local_history.device)
            cond_mask = torch.ones(b, dtype=torch.bool, device=local_history.device)

            cache_cond.reset(dummy_batch.shape[0])
            cache_uncond.reset(dummy_batch.shape[0])

            cache_cond.enable_cache_updates()
            cache_uncond.enable_cache_updates()

            for step_idx in range(sampling_steps):
                x = local_history.clone()
                ts = ts_history.clone()

                if step_idx > 0:
                    x = x[:,-1:]
                    ts = ts[:,-1:]

                # Get unconditional predictions
                pred_video_uncond = model(x, ts, mouse_step, btn_step, has_controls=uncond_mask, kv_cache=cache_uncond)

                if self.cfg_scale > 0:
                    # Get conditional predictions
                    pred_video_cond = model(x, ts, mouse_step, btn_step, has_controls=cond_mask, kv_cache=cache_cond)
                    # Apply CFG
                    pred_video = pred_video_uncond + self.cfg_scale * (pred_video_cond - pred_video_uncond)
                else:
                    # Skip conditional branch when cfg_scale is 0
                    pred_video = pred_video_uncond

                x = x - pred_video*dt[step_idx]
                ts = ts - dt[step_idx]

                local_history[:,-1] = x[:,-1]
                ts_history[:,-1] = ts[:,-1]

                if step_idx == 0:
                    mouse = mouse[:,-1:]
                    btn = btn[:,-1:]

                    # The final frame doesn't go in cache
                    cache_cond.truncate(1, front = True)
                    cache_uncond.truncate(1, front = True)
                    cache_cond.disable_cache_updates()
                    cache_uncond.disable_cache_updates()

            # Frame is entirely cleaned now
            new_frame = local_history[:,-1:]
            clean_history = torch.cat([clean_history, new_frame], dim=1)

        return clean_history
