import torch
from torch import Tensor
from torch.nn import Module
from functools import cache

from owl_wms.nn.kv_cache import KVCache
from owl_wms.configs import Config as RunConfig
from owl_wms.utils.flow_match_scheduler import FlowMatchScheduler

_SIGMA_TABLE = FlowMatchScheduler(num_inference_steps=1000, num_train_timesteps=1000).sigmas.cpu()

@cache
def sigma(t: int) -> Tensor: return _SIGMA_TABLE[int(t)].to(torch.float32)

def alpha(t: int) -> Tensor: return (1 - sigma(t).square()).sqrt()


class SelfForcingSampler:
    def __init__(self,
            model: Module,
            config: RunConfig,
            batch_size: int,
            latent_shape: tuple[int, int, int, int],
            t_schedule: list[int] = [1000, 750, 500, 250],
            context_len: int = 48,
            frame_gradient_cutoff: int = 20,
            training: bool = False,
            autocast: torch.dtype = torch.bfloat16
        ):
        self.training = training
        self.autocast = autocast

        # -- models, hardware
        self.model = model
        self.config = config
        self.device = next(model.parameters()).device

        # -- sampling
        self.t_schedule = t_schedule
        self.batch_size = batch_size
        self.context_len = context_len
        self.latent_shape = latent_shape
        
        # -- gradient optimisation
        self.frame_gradient_cutoff = frame_gradient_cutoff
        self.kv_cache = KVCache(self.config.model).to(self.device)
        self.kv_cache.reset(self.config.train.batch_size)

        # -- validation
        assert self.frame_gradient_cutoff < self.context_len
    
    @torch.no_grad()
    def _warmup_kv(self, latent_primers: list[dict[str, Tensor]]):
        """Fill rolling KV cache without tracking grads."""
        self.kv_cache.enable_cache_updates()
        for f in latent_primers:
            _ = self.model(
                x       = f["latent"].to(self.device),
                t       = 0,
                kv_cache= self.kv_cache,
                mouse   = f["mouse"].to(self.device),
                btn     = f["button"].to(self.device),
            )
        self.kv_cache.disable_cache_updates()

    def autoregressive_rollout(self,
                               btn: Tensor,
                               mouse: Tensor,
                               latent_conditioning: list[dict[str, Tensor]],
                               num_frames: int) -> list[Tensor]:
        B             = self.batch_size
        C, F, H, W    = self.latent_shape
        Tschedule     = self.t_schedule
        device        = self.device
        N             = num_frames
        start_grad_at = N - self.frame_gradient_cutoff
        tokens_per_frame = self.config.model.tokens_per_frame

        if latent_conditioning:
            self._warmup_kv(latent_conditioning)

        clean_latents = []

        for i in range(N):
            grad_frame = i >= start_grad_at  # last L₁ frames
            s_idx = torch.randint(0, len(Tschedule), ())

            # sample x_t at the *largest* timestep
            x_t = torch.randn(B, C, F, H, W, device=device)

            for step_idx, t in enumerate(reversed(Tschedule)):
                # enable grad **only** on the chosen (reversed) step
                keep_grad = grad_frame and (step_idx == s_idx) and self.training
                x_t.requires_grad_(keep_grad)
                with torch.autocast(device_type=device.type, dtype=self.autocast):
                    self.kv_cache.enable_cache_updates()
                    x_0 = self.model(
                        x       = x_t,
                        t       = t,
                        kv_cache= self.kv_cache,
                        mouse   = mouse,
                        btn     = btn,
                    )
                    self.kv_cache.disable_cache_updates()
                    cache_overflow = len(self.kv_cache) - (self.context_len * tokens_per_frame)
                    if cache_overflow > 0:
                        drop = -(-cache_overflow // tokens_per_frame)  # ceil division
                        self.kv_cache.truncate(drop)

                # move to the previous timestep unless we hit t=0
                if t != 0:
                    eps   = torch.randn_like(x_0)
                    _alpha, _sigma = alpha(t), sigma(t)
                    x_t   = (_alpha * x_0) + (_sigma * eps)
                else:
                    break                            # reached fully-denoised

            # detach unless this frame carries grads
            clean_latents.append(x_0 if keep_grad else x_0.detach())

        return clean_latents
