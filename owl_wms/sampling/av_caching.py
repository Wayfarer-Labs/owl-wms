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
    def __call__(self, model, x, mouse: torch.Tensor | None, btn: torch.Tensor | None):
        """Generate `num_frames` new frames and return updated tensors."""
        batch_size, init_len = x.size(0), x.size(1)

        dt = get_sd3_euler(self.n_steps).to(device=x.device, dtype=x.dtype)

        kv_cache = KVCache(model.config)
        kv_cache.reset(batch_size)

        latents = [x]

        # History for the first frame generation step = full clean clip
        prev_x = x
        prev_mouse = mouse[:, :init_len] if mouse is not None else None
        prev_btn = btn[:, :init_len] if btn is not None else None

        for idx in tqdm(range(self.num_frames), desc="Sampling frames"):
            start = min(init_len + idx, mouse.size(1) - 1) if mouse is not None else init_len + idx
            curr_mouse = mouse[:, start: start + 1] if mouse is not None else None
            curr_btn = btn[:, start: start + 1] if btn is not None else None

            x = self.denoise_frame(
                model, kv_cache,
                prev_x, prev_mouse, prev_btn,
                curr_mouse, curr_btn,
                dt=dt,
            )

            latents.append(x)

            # all history kv cached except for newly generated from - set the previous as the new state
            prev_x = x
            prev_mouse, prev_btn = curr_mouse, curr_btn

        return torch.cat(latents, dim=1)

    @staticmethod
    def sigma_noise(x, sigma):
        return x + torch.randn_like(x) * sigma

    def denoise_frame(
        self,
        model,
        kv_cache: KVCache,
        prev_video: torch.Tensor,
        prev_mouse: torch.Tensor,
        prev_btn: torch.Tensor,
        curr_mouse: torch.Tensor,
        curr_btn: torch.Tensor,
        dt: torch.Tensor,
    ):
        """Run all denoising steps for new frame"""
        batch_size = prev_video.size(0)

        # Partially re-noise history
        prev_vid = self.sigma_noise(prev_video, self.noise_prev)
        t_prev = prev_video.new_full((batch_size, prev_vid.size(1)), self.noise_prev)

        # Create new pure-noise frame
        new_vid = torch.randn_like(prev_video[:, :1])
        t_new = t_prev.new_ones(batch_size, 1) * dt.sum()

        # update kv cache with previous uncached frames
        kv_cache.enable_cache_updates()
        eps_v = model(
            torch.cat([prev_vid, new_vid], dim=1),
            torch.cat([t_prev, t_new], dim=1),
            torch.cat([prev_mouse, curr_mouse], dim=1) if prev_mouse is not None else None,
            torch.cat([prev_btn, curr_btn], dim=1) if prev_btn is not None else None,
            kv_cache=kv_cache,
        )
        kv_cache.disable_cache_updates()
        kv_cache.truncate(1, front=True)  # new "still-being-denoised" frame from kv cache

        # Euler update for step‑0 (affects only the *last* frame)
        new_vid -= eps_v[:, -1:] * dt[0]
        t_new -= dt[0]

        # Remaining diffusion steps with cached history, denoising denoising only new frame
        for step in range(1, self.n_steps):
            eps_vid = model(new_vid, t_new, curr_mouse, curr_btn, kv_cache=kv_cache)
            new_vid -= eps_vid * dt[step]
            t_new -= dt[step]

        # Clean frame will be cached automatically in the *next* step‑0
        return new_vid
