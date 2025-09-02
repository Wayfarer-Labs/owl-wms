from typing import Optional
from torch import Tensor
from tensordict import TensorDict

import torch
from tqdm import tqdm

from ..nn.kv_cache import StaticKVCache

from .schedulers import get_sd3_euler


class AVCachingSampler:
    """
    Parameters
    ----------
    :param n_steps: Number of diffusion steps for each frame
    :param cfg_scale: Must be 1.0
    :param noise_prev: Noise previous frame
    """
    def __init__(self, n_steps: int = 16, cfg_scale: float = 1.0, noise_prev: float = 0.2) -> None:
        if cfg_scale != 1.0:
            raise NotImplementedError("cfg_scale must be 1.0 until updated to handle")
        self.n_steps = n_steps
        self.noise_prev = noise_prev

    @torch.inference_mode()
    def __call__(self, model, x, prompt_emb: Optional[TensorDict], controller_input: Optional[Tensor], num_frames=60):
        """Generate `num_frames` new frames and return updated tensors."""
        init_len = x.size(1)

        dt = get_sd3_euler(self.n_steps).to(device=x.device, dtype=x.dtype)

        kv_cache = StaticKVCache(model.config, batch_size=x.size(0), dtype=x.dtype).cuda()

        latents = [x]

        # History for the first frame generation step = full clean clip
        prev_ctrl = controller_input[:, :init_len] if controller_input is not None else None

        for idx in tqdm(range(num_frames), desc="Sampling frames"):
            start = init_len + idx
            curr_ctrl = controller_input[:, start: start + 1] if controller_input is not None else None

            x = self.denoise_frame(
                model, prompt_emb, kv_cache,
                x, prev_ctrl, curr_ctrl,
                dt=dt,
            )

            latents.append(x)
            prev_ctrl = curr_ctrl

        return torch.cat(latents, dim=1)

    @torch.compile
    def denoise_frame(
        self,
        model,
        prompt_emb,
        kv_cache: StaticKVCache,
        prev_video: torch.Tensor,
        prev_ctrl: torch.Tensor,
        curr_ctrl: torch.Tensor,
        dt: torch.Tensor,
    ):
        """Run all denoising steps for new frame"""
        batch_size = prev_video.size(0)

        # Partially re-noise history
        prev_vid = torch.lerp(prev_video, torch.randn_like(prev_video), self.noise_prev)
        t_prev = prev_video.new_full((batch_size, prev_vid.size(1)), self.noise_prev)

        # Create new pure-noise frame
        new_vid = torch.randn_like(prev_video[:, :1])
        t_new = t_prev.new_ones(batch_size, 1)

        for step in range(self.n_steps):
            # step 0: include uncached tokens
            # step >= 1: tokens cached, only include being-denoised tokens
            if step == 0:
                vid = torch.cat([prev_vid, new_vid], dim=1)
                tim = torch.cat([t_prev, t_new], dim=1)
                ctrl = torch.cat([prev_ctrl, curr_ctrl], dim=1) if prev_ctrl is not None else None
            else:
                vid, tim, ctrl = new_vid, t_new, curr_ctrl

            eps = model(vid, tim, prompt_emb, ctrl, kv_cache=kv_cache)
            new_vid -= eps[:, -1:] * dt[step]  # only update the new frame
            t_new -= dt[step]

        # Clean frame will be cached automatically in the *next* step‑0
        return new_vid
