from typing import Optional, Any
from diffusers import FlowMatchEulerDiscreteScheduler
from tqdm import tqdm
from torch import Tensor
import torch


class StandardProgressiveSampler:
    def __init__(self, n_steps: int = 16, cfg_scale: float = 1.0, sched_step_kw=None) -> None:
        if cfg_scale != 1.0:
            raise NotImplementedError("cfg_scale must be 1.0")
        self.n_steps = n_steps
        self.scheduler = FlowMatchEulerDiscreteScheduler(shift=3.0)
        self.sched_step_kw = sched_step_kw or {}

    @torch.inference_mode()
    def __call__(
        self,
        model: Any,
        x: Tensor,
        prompt_emb: Optional[Any],
        fps: Tensor,
        noise_prev: Tensor,
        mouse: Optional[Tensor] = None,
        button: Optional[Tensor] = None,
        num_frames: int = 120,
    ) -> Tensor:
        B, init_len = x.size(0), x.size(1)
        self.scheduler.set_timesteps(self.n_steps, device=x.device)

        frame_timestamps = model.get_frame_timestamps(fps, init_len + num_frames, x.device)

        latents = [x]

        # initialize per-step history (track tokens at every noise level; exclude final output step)
        _sig = self.scheduler.sigmas.to(x)
        g_hist = torch.randn_like(x) if init_len else x.new_empty((B, 0, *x.shape[2:]))
        hist_levels = torch.stack([torch.lerp(x, g_hist, _sig[s]) for s in range(self.n_steps)], dim=0)

        for idx in tqdm(range(num_frames), desc="Sampling frames"):
            start = init_len + idx
            hist_len = hist_levels.size(2)  # all levels share the same length
            ctx_mouse = mouse[:, start - hist_len: start + 1] if mouse is not None else None
            ctx_btn = button[:, start - hist_len: start + 1] if button is not None else None
            ctx_ts = frame_timestamps[:, start - hist_len: start + 1]

            x, new_levels = self.denoise_frame(model, prompt_emb, hist_levels, ctx_mouse, ctx_btn, ctx_ts)

            latents.append(x)
            hist_levels = torch.cat([hist_levels, new_levels], dim=2)  # cat along sequence dim

        return torch.cat(latents, dim=1)

    @torch.compile
    def denoise_frame(
            self,
            model: Any,
            prompt_emb: Optional[Any],
            hist_levels: Tensor,
            mouse: Optional[Tensor],
            btn: Optional[Tensor],
            frame_ts: Tensor,
    ):
        """Denoise a new frame; return final frame and per-step states as [S, B, 1, C, H, W]."""
        # hist_levels: [S, B, L, C, H, W]
        B, L = hist_levels.size(1), hist_levels.size(2)
        sig = self.scheduler.sigmas.to(hist_levels)
        ds = sig.diff()
        new_frame = torch.randn_like(hist_levels[0, :, :1])  # [B, 1, C, H, W]
        frame_step_cache = new_frame.new_empty((self.n_steps, *new_frame.shape))
        base_levels = torch.arange(L, -1, -1, device=hist_levels.device, dtype=torch.long)  # [L+1]

        for step in range(self.n_steps):
            frame_step_cache[step] = new_frame

            # Context and sigmas at shared levels
            level_idxs = (step + base_levels).clamp_max(self.n_steps - 1)  # Denoise step levels index
            sigma = sig.index_select(0, level_idxs).unsqueeze(0).expand(B, -1).type_as(new_frame)
            ctx = hist_levels.index_select(0, level_idxs[:-1]).diagonal(0, 0, 2).movedim(-1, 1)  # [B, L, C, H, W]
            vid = torch.cat([ctx, new_frame], dim=1)                             # [B, T+1, C, H, W]

            # get velocity -> resolve next step
            v = model(vid, sigma=sigma, frame_timestamp=frame_ts, prompt_emb=prompt_emb, mouse=mouse, button=btn)
            v = v[:, -1:]
            new_frame = (new_frame.float() + ds[step].float() * v.float()).type_as(new_frame)

        return new_frame, frame_step_cache
