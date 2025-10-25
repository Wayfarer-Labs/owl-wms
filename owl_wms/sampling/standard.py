from typing import Optional, Any
from torch import Tensor
import torch
from diffusers import FlowMatchEulerDiscreteScheduler


class StandardSampler:
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

        # snap noise_prev to nearest scheduler sigma (excluding the last)
        noise_prev = torch.as_tensor(noise_prev, device=x.device, dtype=x.dtype)
        _sig = self.scheduler.sigmas.to(x)
        if not torch.isclose(noise_prev, _sig[-1], rtol=1e-5, atol=1e-8):
            noise_prev = _sig[:-1][(_sig[:-1] - noise_prev).abs().argmin()]

        # initialize running noised history once at snapped noise_prev (iid)
        g_hist = torch.randn_like(x) if init_len else x.new_empty((B, 0, *x.shape[2:]))
        hist = torch.lerp(x, g_hist, noise_prev)

        for i in range(num_frames):
            start = init_len + i
            hist_len = hist.size(1)  # match conditioning length to `hist`
            ctx_mouse = mouse[:, start - hist_len : start + 1] if mouse is not None else None
            ctx_btn   = button[:, start - hist_len : start + 1] if button is not None else None
            ctx_ts    = frame_timestamps[:, start - hist_len : start + 1]

            x, hist_new = self.denoise_frame(model, prompt_emb, hist, ctx_mouse, ctx_btn, ctx_ts, noise_prev)

            latents.append(x)
            hist = torch.cat([hist, hist_new], dim=1)

        return torch.cat(latents, dim=1)

    @torch.compile
    def denoise_frame(
            self,
            model: Any,
            prompt_emb: Optional[Any],
            hist: Tensor,
            mouse: Optional[Tensor],
            btn: Optional[Tensor],
            frame_ts: Tensor,
            noise_prev: Tensor
    ):
        """Run all denoising steps for the new frame (no KV cache)."""
        sigma_prev = noise_prev.to(hist).expand(hist.shape[0], hist.shape[1])
        new_vid = torch.randn_like(hist[:, :1])
        hist_new = None
        sig = self.scheduler.sigmas.to(hist)

        for step in range(self.n_steps):
            sig_val = sig[step]
            vid = torch.cat([hist, new_vid], dim=1)
            sigma = torch.cat([sigma_prev, sig_val.expand(hist.size(0), 1)], dim=1)
            v = model(vid, sigma=sigma, frame_timestamp=frame_ts, prompt_emb=prompt_emb, mouse=mouse, button=btn,)
            v = v[:, -1:]  # only the new frame’s eps
            if hist_new is None and torch.isclose(sig_val, noise_prev.to(sig.dtype), rtol=1e-5, atol=1e-8):
                hist_new = new_vid  # capture state exactly at snapped noise_prev

            # FlowMatchEulerDiscreteScheduler step
            dsigma = self.scheduler.sigmas[step + 1] - self.scheduler.sigmas[step]
            new_vid = (new_vid.float() + dsigma.float() * v.float()).type_as(new_vid)

        if torch.isclose(noise_prev.to(sig.dtype), sig[-1], rtol=1e-5, atol=1e-8):
            hist_new = new_vid

        assert hist_new is not None, "noise_prev must lie within the scheduler sigma range."
        return new_vid, hist_new
