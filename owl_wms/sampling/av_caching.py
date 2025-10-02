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

        self.scheduler = FlowMatchEulerDiscreteScheduler(shift=3.0)
        self.sched_step_kw = sched_step_kw or {}

        # self.scheduler = FlowMatchHeunDiscreteScheduler(shift=3.0)
        # self.sched_step_kw = sched_step_kw or {}

        # self.scheduler = UniPCMultistepScheduler(
        #     prediction_type="flow_prediction", use_flow_sigmas=True, flow_shift=3.0, timestep_spacing="trailing")
        # )
        # self.sched_step_kw = {}  # must be empty for unipc

        self.scheduler.set_timesteps(n_steps)
        self.sigmas = self.scheduler.sigmas
        self.timesteps = self.scheduler.timesteps


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
    ):
        """Generate `num_frames` new frames and return updated tensors."""
        init_len = x.size(1)

        self.sigmas = self.sigmas.to(x.device, x.dtype)
        self.timesteps = self.timesteps.to(x.device, x.dtype)

        seq_len = init_len + num_frames
        kv_cache = StaticKVCache(model.config, max_seq_len=seq_len, batch_size=x.size(0), dtype=x.dtype).to(x.device)
        frame_timestamps = model.get_frame_timestamps(fps, seq_len, x.device)

        # History for the first frame generation step = full clean clip
        prev_ctrl = controller_input[:, :init_len] if controller_input is not None else None
        prev_ts = frame_timestamps[0, :init_len]

        latents = [x]
        for idx in tqdm(range(num_frames), desc="Sampling frames"):
            start = init_len + idx
            curr_ctrl = controller_input[:, start: start + 1] if controller_input is not None else None
            curr_ts = frame_timestamps[0, start:start + 1]

            x = self.denoise_frame(
                model, prompt_emb, kv_cache,
                x, prev_ctrl, curr_ctrl,
                prev_ts=prev_ts, curr_ts=curr_ts,
                noise_prev=noise_prev
            )

            latents.append(x)
            prev_ctrl, prev_ts = curr_ctrl, curr_ts

        return torch.cat(latents, dim=1)

    @torch.compile
    def fwd(self, model, *args, **kwargs):
        return model(*args, **kwargs)[:, -1:]

    def denoise_frame(
        self,
        model,
        prompt_emb,
        kv_cache: StaticKVCache,
        prev_video: torch.Tensor,
        prev_ctrl: torch.Tensor,
        curr_ctrl: torch.Tensor,
        prev_ts: torch.Tensor,
        curr_ts: torch.Tensor,
        noise_prev: torch.Tensor,
    ):
        """Run all denoising steps for new frame"""
        B = prev_video.size(0)

        # Partially re-noise history
        prev_vid = torch.lerp(prev_video, torch.randn_like(prev_video), noise_prev)
        sigma_prev = prev_video.new_full((B, prev_vid.size(1)), noise_prev)

        # Create new pure-noise frame
        new_vid = torch.randn_like(prev_video[:, :1])

        for step, (t, s) in enumerate(zip(self.timesteps[:-1], self.sigmas[:-1])):
            # step 0: include uncached previous frames tokens
            # step >= 1: prev frame cached, only include current frame
            sigma = s.expand(B, 1)
            if step == 0:
                vid = torch.cat([prev_vid, new_vid], dim=1)
                sigma = torch.cat([sigma_prev, sigma], dim=1)  # TODO: rename sigma
                ctrl = torch.cat([prev_ctrl, curr_ctrl], dim=1) if prev_ctrl is not None else None
                frame_ts = torch.cat([prev_ts, curr_ts], dim=0)
            else:
                vid, ctrl, frame_ts = new_vid, curr_ctrl, curr_ts
            frame_ts = frame_ts.unsqueeze(0)  # batchsize = 1

            eps = self.fwd(
                model,
                vid,
                sigma=sigma,
                frame_timestamp=frame_ts,
                prompt_emb=prompt_emb,
                controller_inputs=ctrl,
                kv_cache=kv_cache
            )[:, -1:]  # only the new frame’s eps

            new_vid = self.scheduler.step(
                model_output=eps,
                timestep=t,
                sample=new_vid,
                **self.sched_step_kw
            ).prev_sample

        return new_vid
