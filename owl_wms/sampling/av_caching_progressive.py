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
        self.sigmas = self.sigmas.to(device=x.device)
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

        # denoising steps for last `uncached_k` frames
        prev_rollouts = x.new_full((uncached_k, self.n_steps, x.size(0), *x.shape[2:]), torch.nan)
        hist = x[:, -min(uncached_k, x.size(1)):].transpose(0, 1).float()
        prev_rollouts[:hist.size(0)] = torch.lerp(
            hist.unsqueeze(1),                              # clean
            torch.randn_like(hist).unsqueeze(1),           # noise
            self.sigmas[1:self.n_steps + 1]                # shape: (n_steps,)
                .reshape(1, self.n_steps, *([1] * (hist.ndim - 1)))
        ).type_as(x)

        for idx in tqdm(range(num_frames), desc="Sampling frames"):
            new_noise = torch.randn_like(x.new_empty((x.size(0), 1, *x.shape[2:])))
            x = torch.cat([x, new_noise], dim=1)

            if controller_input is not None:
                ctrl = torch.cat((ctrl, controller_input[:, init_len + idx:init_len + idx + 1]), dim=1)
            ts = torch.cat((ts, frame_timestamps[0, init_len + idx:init_len + idx + 1].unsqueeze(0)), dim=1)

            x, ctrl, ts, new_rollout = self.denoise_frame(
                model, prompt_emb, kv_cache, x, ctrl, ts, prev_rollouts, uncached_k, noise_prev
            )

            # slide window of history frames
            prev_rollouts = torch.roll(prev_rollouts, shifts=-1, dims=0)
            prev_rollouts[-1] = new_rollout

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
        prev_rollouts: torch.Tensor,
        uncached_k: int,
        noise_prev: float,
    ):
        """Run all denoising steps for new frame (seq = cat(hist, gaussian))."""
        B = seq.size(0)
        new_rollout = seq.new_empty((self.n_steps, B, *seq.shape[2:]))

        for step in range(self.n_steps):
            L, H = seq.size(1), seq.size(1) - 1
            dist = torch.arange(L - 1, -1, -1, device=seq.device)
            idx_all = (step + dist).clamp_max(self.n_steps)
            # to ignore clean step: .clamp_max(self.n_steps - 1)
            sigma = self.sigmas[idx_all][None].expand(B, -1)
            assert (sigma[:, -1] == self.sigmas[step]).all()
            seq_in = seq.clone()
            # Align prev_rollouts rows with absolute seq positions [-K .. -2]
            K = uncached_k
            R = min(H, K - 1)  # history frames only
            if R > 0:
                oldest = max(0, L - K)  # absolute index for position -K (or 0 if fewer than K frames)
                for offset in range(R):  # maps rows 0..R-1 -> positions [-K .. -2]
                    pos = oldest + offset
                    s = int(idx_all[pos])  # scheduler index in [0..n_steps]
                    if s > 0:              # cache stores σ[1..] in slots [0..]
                        seq_in[:, pos] = prev_rollouts[offset, s - 1]

            v = self.fwd(
                model,
                seq_in,
                sigma=sigma,
                frame_timestamp=ts,
                prompt_emb=prompt_emb,
                controller_inputs=ctrl,
                kv_cache=kv_cache
            )[:, -1:]  # only the new frame’s eps

            dsigma = (self.sigmas[step + 1] - self.sigmas[step])
            seq[:, -1:] = (seq[:, -1:] + dsigma * v).type_as(seq)
            new_rollout[step] = seq[:, -1:].squeeze(1)

            seq = seq[:, -uncached_k:]
            ts = ts[:, -uncached_k:]
            if ctrl is not None:
                ctrl = ctrl[:, -uncached_k:]

        return seq, ctrl, ts, new_rollout
