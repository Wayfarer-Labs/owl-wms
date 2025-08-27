from typing import Optional
from torch import Tensor

import torch
from tqdm import tqdm

from ..nn.kv_cache import KVCache

from .schedulers import get_sd3_euler


class AVCachingSampler:
    """
    Parameters
    ----------
    :param n_steps: Number of diffusion steps for each frame
    :param cfg_scale: Must be 1.0
    :param num_frames: Number of new frames to sample
    :param noise_prev: Noise previous frame
    """
    def __init__(self, n_steps: int = 16, cfg_scale: float = 1.0, num_frames: int = 60, noise_prev: float = 0.2) -> None:
        if cfg_scale != 1.0:
            raise NotImplementedError("cfg_scale must be 1.0 until updated to handle")
        self.n_steps = n_steps
        self.num_frames = num_frames
        self.noise_prev = noise_prev

    @torch.no_grad()
    def __call__(self, model, x, prompt: Optional[Tensor], controller_input: Optional[Tensor]):
        """Generate `num_frames` new frames and return updated tensors."""
        batch_size, init_len = x.size(0), x.size(1)

        dt = get_sd3_euler(self.n_steps).to(device=x.device, dtype=x.dtype)

        kv_cache = KVCache(model.config)
        kv_cache.reset(batch_size)

        latents = [x]

        # History for the first frame generation step = full clean clip
        prev_x = x
        prev_ctrl = controller_input[:, :init_len] if controller_input is not None else None

        for idx in tqdm(range(self.num_frames), desc="Sampling frames"):
            start = min(init_len + idx, controller_input.size(1) - 1) if controller_input is not None else init_len + idx
            curr_ctrl = controller_input[:, start: start + 1] if controller_input is not None else None

            x = self.denoise_frame(
                model, prompt, kv_cache,
                prev_x, prev_ctrl, curr_ctrl,
                dt=dt,
            )

            latents.append(x)

            # all history kv cached except for newly generated from - set the previous as the new state
            prev_x = x
            prev_ctrl = curr_ctrl

        return torch.cat(latents, dim=1)

    @staticmethod
    def zlerp(x, alpha):
        z = torch.randn_like(x)
        return x * (1. - alpha) + z * alpha

    def denoise_frame(
        self,
        model,
        prompt,
        kv_cache: KVCache,
        prev_video: torch.Tensor,
        prev_ctrl: torch.Tensor,
        curr_ctrl: torch.Tensor,
        dt: torch.Tensor,
    ):
        """Run all denoising steps for new frame"""
        batch_size = prev_video.size(0)

        # Partially re-noise history
        prev_vid = self.zlerp(prev_video, self.noise_prev)
        t_prev = prev_video.new_full((batch_size, prev_vid.size(1)), self.noise_prev)

        # Create new pure-noise frame
        new_vid = torch.randn_like(prev_video[:, :1])
        t_new = t_prev.new_ones(batch_size, 1)

        # update kv cache with previous uncached frames
        kv_cache.enable_cache_updates()
        eps_v = model(
            torch.cat([prev_vid, new_vid], dim=1),
            torch.cat([t_prev, t_new], dim=1),
            prompt,
            torch.cat([prev_ctrl, curr_ctrl], dim=1) if prev_ctrl is not None else None,
            kv_cache=kv_cache,
        )
        kv_cache.disable_cache_updates()
        kv_cache.truncate(1, front=True)  # new "still-being-denoised" frame from kv cache

        # Euler update for step‑0 (affects only the *last* frame)
        new_vid -= eps_v[:, -1:] * dt[0]
        t_new -= dt[0]

        # Remaining diffusion steps with cached history, denoising denoising only new frame
        for step in range(1, self.n_steps):
            eps_vid = model(new_vid, t_new, prompt, curr_ctrl, kv_cache=kv_cache)
            new_vid -= eps_vid * dt[step]
            t_new -= dt[step]

        # Clean frame will be cached automatically in the *next* step‑0
        return new_vid
