from typing import Optional
from torch import Tensor
from tensordict import TensorDict
import einops as eo

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

        kv_cache = StaticKVCache(model.config, batch_size=x.size(0), dtype=x.dtype).to(x.device)

        # History for the first frame generation step = full clean clip
        prev_ctrl = controller_input[:, :init_len] if controller_input is not None else None
        prev_time = torch.arange(init_len, device=x.device, dtype=torch.long)

        latents = [x]
        for idx in tqdm(range(num_frames), desc="Sampling frames"):
            start = init_len + idx
            curr_ctrl = controller_input[:, start: start + 1] if controller_input is not None else None
            curr_time = torch.tensor([start], device=x.device, dtype=torch.long)

            x = self.denoise_frame(
                model, prompt_emb, kv_cache,
                x, prev_ctrl, curr_ctrl,
                prev_time=prev_time, curr_time=curr_time,
                dt=dt,
            )

            latents.append(x)
            prev_ctrl = curr_ctrl
            prev_time = curr_time

        return torch.cat(latents, dim=1)

    def get_pos_ids(self, seq_ts: torch.Tensor, H: int, W: int, B: int, device) -> TensorDict:
        """Return TensorDict with t/y/x positions for [B, F*H*W]. seq_ts: [F] or [B,F] (long)."""
        if seq_ts.ndim == 1:
            seq_ts = seq_ts[None, :].expand(B, -1)  # [B,F]
        F = seq_ts.size(1)
        y = torch.arange(H, device=device)
        x = torch.arange(W, device=device)
        yy, xx = torch.meshgrid(y, x, indexing='ij')
        y_flat = yy.reshape(-1).repeat(F)[None, :].expand(B, -1)
        x_flat = xx.reshape(-1).repeat(F)[None, :].expand(B, -1)
        return TensorDict(
            {
                "t_pos": seq_ts.repeat_interleave(H * W, 1),
                "y_pos": y_flat,
                "x_pos": x_flat,
            },
            batch_size=[B, F * H * W],
        )

    @torch.compile
    def denoise_frame(
        self,
        model,
        prompt_emb,
        kv_cache: StaticKVCache,
        prev_video: torch.Tensor,
        prev_ctrl: torch.Tensor,
        curr_ctrl: torch.Tensor,
        prev_time: torch.Tensor,
        curr_time: torch.Tensor,
        dt: torch.Tensor,
    ):
        """Run all denoising steps for new frame"""
        B = prev_video.size(0)
        H, W = prev_video.size(3), prev_video.size(4)

        # precompute position IDs
        curr_pos_ids = self.get_pos_ids(curr_time, H, W, B, prev_video.device)
        all_pos_ids = self.get_pos_ids(torch.cat([prev_time, curr_time], dim=0), H, W, B, prev_video.device)

        # Partially re-noise history
        prev_vid = torch.lerp(prev_video, torch.randn_like(prev_video), self.noise_prev)
        t_prev = prev_video.new_full((B, prev_vid.size(1)), self.noise_prev)

        # Create new pure-noise frame
        new_vid = torch.randn_like(prev_video[:, :1])
        t_new = t_prev.new_ones(B, 1)

        for step in range(self.n_steps):
            # step 0: include uncached previous frames tokens
            # step >= 1: prev frame cached, only include current frame
            if step == 0:
                vid = torch.cat([prev_vid, new_vid], dim=1)
                tim = torch.cat([t_prev, t_new], dim=1)  # TODO: rename sigma
                ctrl = torch.cat([prev_ctrl, curr_ctrl], dim=1) if prev_ctrl is not None else None
                pos_ids = all_pos_ids
            else:
                vid, tim, ctrl, pos_ids = new_vid, t_new, curr_ctrl, curr_pos_ids

            x_flat = eo.rearrange(vid, 'b n c h w -> b (n h w) c')
            eps = model.flat_forward(x_flat, pos_ids, tim, prompt_emb, ctrl, kv_cache=kv_cache)
            eps = eo.rearrange(eps, 'b (n h w) c -> b n c h w', h=H, w=W)
            new_vid -= eps[:, -1:] * dt[step]  # only update the new frame
            t_new -= dt[step]

        # Clean frame will be cached automatically in the *next* step‑0
        return new_vid
