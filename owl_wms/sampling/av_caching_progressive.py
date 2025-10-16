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
    """
    def __init__(self, n_steps: int = 16, cfg_scale: float = 1.0, sched_kw=None, sched_step_kw=None) -> None:
        if cfg_scale != 1.0:
            raise NotImplementedError("cfg_scale must be 1.0 until updated to handle")

        self.n_steps = n_steps
        self.scheduler = FlowMatchEulerDiscreteScheduler(shift=3.0)
        self.sched_step_kw = sched_step_kw or {}

        # self.scheduler = FlowMatchHeunDiscreteScheduler(shift=3.0)
        # self.sched_step_kw = sched_step_kw or {}  # must be passed to self.scheduler.step(...)
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
        noise_prev: Tensor = 0.0,  # TODO: remove, unused
        num_frames: int = 120,
        noise_distribution: str = "iid",
        uncached_k: int = 1,
    ):
        """Generate `num_frames` new frames and return updated tensors."""
        init_len = x.size(1)

        seq_len = init_len + num_frames
        kv_cache = StaticKVCache(
            model.config, max_seq_len=seq_len, batch_size=x.size(0), dtype=x.dtype,
            n_uncached_frames=uncached_k
        ).to(x.device)
        frame_timestamps = model.get_frame_timestamps(fps, seq_len, x.device)

        latents = [x]

        # initialize running noised history once at snapped noise_prev
        ctrl = controller_input[:, :init_len] if controller_input is not None else None
        ts = frame_timestamps[0, :init_len].unsqueeze(0)

        # Concatenate once before passing
        seq = torch.cat([x, torch.randn_like(x.new_empty((x.size(0), 1, *x.shape[2:])))], dim=1)

        for idx in tqdm(range(num_frames), desc="Sampling frames"):
            if controller_input is not None:
                ctrl = torch.cat((ctrl, controller_input[:, init_len + idx:init_len + idx + 1]), dim=1)
            ts = torch.cat((ts, frame_timestamps[0, init_len + idx:init_len + idx + 1].unsqueeze(0)), dim=1)

            seq, ctrl, ts = self.denoise_frame(
                model, prompt_emb, kv_cache, seq=seq, ctrl=ctrl, ts=ts, uncached_k=uncached_k

            )
            latents.append(seq[:, -1:])
            seq = torch.cat([seq[:, -uncached_k:], torch.randn_like(x.new_empty((x.size(0), 1, *x.shape[2:])))], dim=1)

        return torch.cat(latents, dim=1)

    @torch.compile
    def fwd(self, model, *args, **kwargs):
        return model(*args, **kwargs)

    def denoise_frame(
        self,
        model,
        prompt_emb,
        kv_cache: StaticKVCache,
        seq: torch.Tensor,
        ctrl: Optional[torch.Tensor],
        ts: torch.Tensor,
        uncached_k: int,
    ):
        """Run all denoising steps for new frame (seq = cat(hist, gaussian))."""
        self.scheduler.set_timesteps(self.n_steps)

        B = seq.size(0)

        # history sigmas at snapped sigma (formerly noise_prev)
        sigma_hist = seq.new_zeros((B, seq.size(1) - 1))

        for step in range(self.n_steps):
            # history sigmas = 0, last frame follows scheduler
            sigma_hist = seq.new_zeros((B, seq.size(1) - 1))
            sigma_last = self.scheduler.sigmas[step].expand(B, 1).to(seq.device, seq.dtype)
            sigma = torch.cat([sigma_hist, sigma_last], dim=1)

            v = self.fwd(
                model,
                seq,
                sigma=sigma,
                frame_timestamp=ts,
                prompt_emb=prompt_emb,
                controller_inputs=ctrl,
                kv_cache=kv_cache
            )[:, -1:]  # only the new frame’s eps

            seq[:, -1:] = self.scheduler.step(
                model_output=v,
                timestep=self.scheduler.timesteps[step],
                sample=seq[:, -1:],
            ).prev_sample

            # after step 0, drop history and continue with last `uncached_k` frames
            seq = seq[:, -uncached_k:]
            ts = ts[:, -uncached_k:]
            ctrl = ctrl[:, -uncached_k:] if ctrl is not None else None

        return seq, ctrl, ts
