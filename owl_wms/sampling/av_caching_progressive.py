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
        self.scheduler.set_timesteps(self.n_steps)
        self.sigmas = self.scheduler.sigmas

        # self.scheduler = FlowMatchHeunDiscreteScheduler(shift=3.0)
        # self.sched_step_kw = sched_step_kw or {}  # must be passed to self.scheduler.step(...)
        # self.scheduler = UniPCMultistepScheduler(
        #     prediction_type="flow_prediction", use_flow_sigmas=True, flow_shift=3.0, timestep_spacing="trailing")
        # )
        # self.sched_step_kw = {}  # must be empty for unipc

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
        # uncached_k: int = 1,  # now self.n_steps
    ):
        """Generate `num_frames` new frames and return updated tensors."""
        self.sigmas = self.sigmas.to(device=x.device, dtype=x.dtype)
        uncached_k = self.n_steps

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
        noise = torch.randn_like(x, dtype=torch.float32)

        for idx in tqdm(range(num_frames), desc="Sampling frames"):

            new_noise = torch.randn_like(noise.new_empty((x.size(0), 1, *x.shape[2:])))
            noise = torch.cat([noise, new_noise], dim=1)
            x = torch.cat([x, new_noise.type_as(x)], dim=1)

            if controller_input is not None:
                ctrl = torch.cat((ctrl, controller_input[:, init_len + idx:init_len + idx + 1]), dim=1)
            ts = torch.cat((ts, frame_timestamps[0, init_len + idx:init_len + idx + 1].unsqueeze(0)), dim=1)

            x, ctrl, ts, noise = self.denoise_frame(model, prompt_emb, kv_cache, x, ctrl, ts, noise, uncached_k, noise_prev)

            latents.append(x[:, -1:])

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
        noise: torch.Tensor,
        uncached_k: int,
        noise_prev: float,
    ):
        """Run all denoising steps for new frame (seq = cat(hist, gaussian))."""
        B = seq.size(0)

        for step in range(self.n_steps):
            L = seq.size(1)
            H = L - 1
            d = torch.arange(H, 0, -1, device=seq.device)                      # distances H..1 (empty if H==0)
            idx = (step + d).clamp(max=self.sigmas.numel() - 1)                         # per-history indices

            sigma = torch.zeros(B, L, device=seq.device, dtype=torch.float32)
            sigma[:, :-1] = self.sigmas[idx].view(1, H).expand(B, H).clamp_min_(noise_prev)
            sigma[:, -1] = self.sigmas[step]  # current frame

            seq_in = seq.clone()
            seq_in[:, :-1] = torch.lerp(
                seq[:, :-1].float(),
                noise[:, :-1].float(),
                self.sigmas[idx].view(1, H, *([1] * (seq.ndim - 2))).float()
            ).type_as(seq)

            v = self.fwd(
                model,
                seq_in,
                sigma=sigma,
                frame_timestamp=ts,
                prompt_emb=prompt_emb,
                controller_inputs=ctrl,
                kv_cache=kv_cache
            )[:, -1:]  # only the new frame’s eps

            dsigma = self.sigmas[step + 1] - self.sigmas[step]
            seq[:, -1:] = (seq[:, -1:] + dsigma * v).type_as(seq)

            # after step 0, drop history and continue with last `uncached_k` frames
            noise = torch.randn_like(noise[:, -uncached_k:])  # HACK: new noise, might be better
            seq = seq[:, -uncached_k:]
            ts = ts[:, -uncached_k:]
            ctrl = ctrl[:, -uncached_k:] if ctrl is not None else None

        return seq, ctrl, ts, noise
