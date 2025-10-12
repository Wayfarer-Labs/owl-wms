from typing import Optional
from torch import Tensor
from tensordict import TensorDict

import torch
from tqdm import tqdm

from ..nn.kv_cache import StaticKVCache

from diffusers import FlowMatchEulerDiscreteScheduler


class AVCachingSampler:
    """
    Parameters
    ----------
    :param n_steps: Number of diffusion steps for each frame
    :param cfg_scale: Must be 1.0
    :param noise_prev: Noise previous frame
    """
    def __init__(self, n_steps: int = 16, cfg_scale: float = 1.0, sched_kw=None, sched_step_kw=None) -> None:
        if cfg_scale != 1.0:
            raise NotImplementedError("cfg_scale must be 1.0 until updated to handle")

        self.n_steps = n_steps
        self.scheduler = FlowMatchEulerDiscreteScheduler(shift=3.0)
        self.sched_step_kw = sched_step_kw or {}

        # self.scheduler = FlowMatchHeunDiscreteScheduler(shift=3.0)
        # self.sched_step_kw = sched_step_kw or {}

        # self.scheduler = UniPCMultistepScheduler(
        #     prediction_type="flow_prediction", use_flow_sigmas=True, flow_shift=3.0, timestep_spacing="trailing")
        # )
        # self.sched_step_kw = {}  # must be empty for unipc

        self.scheduler.set_timesteps(n_steps)

    @torch.inference_mode()
    def __call__(
        self,
        model,
        x,
        prompt_emb: Optional[TensorDict],
        controller_input: Optional[Tensor],
        fps: Tensor,
        noise_prev: Tensor,
        num_frames: int = 120,
        noise_distribution: str = "iid"
    ):
        """Generate `num_frames` new frames and return updated tensors."""
        init_len = x.size(1)

        seq_len = init_len + num_frames
        kv_cache = StaticKVCache(model.config, max_seq_len=seq_len, batch_size=x.size(0), dtype=x.dtype).to(x.device)
        frame_timestamps = model.get_frame_timestamps(fps, seq_len, x.device)

        # History for the first frame generation step = full clean clip
        prev_ctrl = controller_input[:, :init_len] if controller_input is not None else None
        prev_ts = frame_timestamps[0, :init_len]

        latents = [x]

        # snap noise_prev to nearest scheduler sigma (excluding the last value) and use it everywhere
        _sigmas = self.scheduler.sigmas.to(x.device, x.dtype)
        _sigmas_wo_last = _sigmas[:-1]
        noise_prev = _sigmas_wo_last[torch.argmin((_sigmas_wo_last - torch.as_tensor(noise_prev, device=x.device, dtype=x.dtype)).abs()).item()]

        # initialize running noised history once at snapped noise_prev
        g_iter = self.iter_gaussians(x[:, :1], noise_distribution=noise_distribution)

        if init_len:
            g_hist = torch.cat([next(g_iter) for _ in range(init_len)], dim=1)
        else:
            g_hist = x.new_empty(size=(x.size(0), 0, *x.shape[2:]))
        hist = torch.lerp(x, g_hist, noise_prev)

        for idx in tqdm(range(num_frames), desc="Sampling frames"):
            start = init_len + idx
            curr_ctrl = controller_input[:, start: start + 1] if controller_input is not None else None
            curr_ts = frame_timestamps[0, start:start + 1]

            x, hist = self.denoise_frame(
                model, prompt_emb, kv_cache,
                hist, prev_ctrl, curr_ctrl,
                prev_ts=prev_ts, curr_ts=curr_ts,
                noise_prev=noise_prev,
                gaussian=next(g_iter),
            )

            latents.append(x)
            prev_ctrl, prev_ts = curr_ctrl, curr_ts

        return torch.cat(latents, dim=1)

    def iter_gaussians(self, x, noise_distribution, alpha: float = 0.7):
        """generator of PYoCo-progressive Gaussians (AR(1))."""
        if noise_distribution == "pyoco_progressive":
            s = (1 + alpha**2) ** -0.5
            rho = alpha * s
            g = torch.randn_like(x, dtype=torch.float32)
            while True:
                yield g.type_as(x)
                g = rho * g + s * torch.randn_like(g, dtype=torch.float32)
        elif noise_distribution == "iid":
            while True:
                yield torch.randn_like(x)
        else:
            raise ValueError()

    @torch.compile
    def fwd(self, model, *args, **kwargs):
        return model(*args, **kwargs)

    def denoise_frame(
        self,
        model,
        prompt_emb,
        kv_cache: StaticKVCache,
        hist: torch.Tensor,
        prev_ctrl: torch.Tensor,
        curr_ctrl: torch.Tensor,
        prev_ts: torch.Tensor,
        curr_ts: torch.Tensor,
        noise_prev: torch.Tensor,
        gaussian: torch.Tensor,
    ):
        """Run all denoising steps for new frame"""
        B = hist.size(0)
        # History is already noised at snapped noise_prev (prepared in __call__)
        sigma_prev = hist.new_full((B, hist.size(1)), torch.as_tensor(noise_prev, device=hist.device, dtype=hist.dtype))

        self.scheduler.set_timesteps(self.n_steps)
        hist_new = None

        new_vid = gaussian
        for step in range(self.n_steps):
            # step 0: include uncached previous frames tokens
            # step >= 1: prev frame cached, only include current frame
            sigma = self.scheduler.sigmas[step].expand(B, 1).to(hist.device, hist.dtype)
            if step == 0:
                vid = torch.cat([hist, new_vid], dim=1)
                sigma = torch.cat([sigma_prev, sigma], dim=1)  # TODO: rename sigma
                ctrl = torch.cat([prev_ctrl, curr_ctrl], dim=1) if prev_ctrl is not None else None
                frame_ts = torch.cat([prev_ts, curr_ts], dim=0)
            else:
                vid, ctrl, frame_ts = new_vid, curr_ctrl, curr_ts
            frame_ts = frame_ts.unsqueeze(0)  # batchsize = 1

            pre = new_vid
            v = self.fwd(
                model,
                vid,
                sigma=sigma,
                frame_timestamp=frame_ts,
                prompt_emb=prompt_emb,
                controller_inputs=ctrl,
                kv_cache=kv_cache
            )[:, -1:]  # only the new frame’s eps

            new_vid = self.scheduler.step(
                model_output=v,
                timestep=self.scheduler.timesteps[step],
                sample=new_vid,
                **self.sched_step_kw
            ).prev_sample

            # capture solver state exactly at snapped noise_prev (no interpolation)
            if hist_new is None and torch.isclose(self.scheduler.sigmas[step].to(noise_prev.device, noise_prev.dtype), noise_prev, rtol=1e-5, atol=1e-8):
                hist_new = pre

        assert hist_new is not None, "noise_prev must lie within the scheduler sigma range."

        return new_vid, hist_new
