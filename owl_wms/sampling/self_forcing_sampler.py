import torch
import random
from torch import Tensor
from torch.nn import Module
from functools import cache

from owl_wms.nn.kv_cache import KVCache
from owl_wms.configs import TransformerConfig as ModelConfig, TrainingConfig
from owl_wms.utils.flow_match_scheduler import FlowMatchScheduler

_SIGMA_TABLE = FlowMatchScheduler(num_inference_steps=1000, num_train_timesteps=1000).sigmas.cpu()


# NOTE t is one element tensor, or int
@cache
def sigma(t: int | Tensor) -> Tensor: return _SIGMA_TABLE[int(t)-1].to(torch.float32) # list is 0-indexed but t_schedule is 1-indexed?

def alpha(t: int | Tensor) -> Tensor: return (1 - sigma(t).square()).sqrt()


class SelfForcingSampler:
    def __init__(self,
            model: Module,
            model_config: ModelConfig,
            batch_size: int,
            latent_shape: tuple[int, int, int, int],
            t_schedule: list[int] = [1000, 750, 500, 250],
            context_len: int = 48,
            frame_gradient_cutoff: int = 8,
            num_gen_frames: int = 64,
            training: bool = False,
            autocast: torch.dtype = torch.bfloat16
        ):
        self.training = training
        self.autocast = autocast

        # -- models, hardware
        self.model = model
        self.model_config = model_config
        self.device = next(model.parameters()).device
        self.tokens_per_frame = self.model_config.tokens_per_frame

        # -- sampling
        self.t_schedule = t_schedule
        self.batch_size = batch_size
        self.context_len = context_len
        self.latent_shape = latent_shape
        self.num_gen_frames = num_gen_frames
    
        
        # -- gradient optimisation
        self.kv_cache = KVCache(self.model_config).to(self.device)
        self.kv_cache.reset(self.batch_size)
        self.frame_gradient_cutoff = frame_gradient_cutoff
        self.start_grad_at = max(0, self.num_gen_frames - self.frame_gradient_cutoff)
        if self.start_grad_at <= 0:
            print(f'WARNING: {self.num_gen_frames=} <= {self.frame_gradient_cutoff=}')

        # -- validation
        assert self.frame_gradient_cutoff < self.context_len
    
    @torch.no_grad()
    def _warmup_kv(self, latent_primers: list[dict[str, Tensor]]):
        """Fill rolling KV cache without tracking grads."""
        self.kv_cache.enable_cache_updates()
        t = torch.zeros((self.batch_size, 1), device=self.device)
        for f in latent_primers:
            _ = self.model.core(
                x       = f["latent"],
                t       = t,
                kv_cache= self.kv_cache,
                mouse   = f["mouse"],
                btn     = f["btn"],
                audio   = f["audio"],
            )
        self.kv_cache.disable_cache_updates()

    def autoregressive_rollout(self,
                               btn: Tensor,
                               mouse: Tensor,
                               audio: Tensor,
                               latent_conditioning: list[dict[str, Tensor]]) -> tuple[Tensor, Tensor]:
        assert btn.shape[1] == mouse.shape[1] == audio.shape[1] == self.num_gen_frames, \
            f'btn, mouse, and audio must have the same number of frames: \
                {self.num_gen_frames=} {btn.shape[1]=} {mouse.shape[1]=} {audio.shape[1]=}'
        
        B             = self.batch_size
        C, H, W       = self.latent_shape       # dims of latent of upstream autoencoder
        t_schedule    = self.t_schedule         # few-step distillation schedule
        device        = self.device
        N             = self.num_gen_frames     # number of frames to generate that are outside the context
        start_grad_at = self.start_grad_at      # frame_idx past which we start keeping track of grads (vanishing error accumulation horizon)        

        if latent_conditioning:
            self._warmup_kv(latent_conditioning)

        tokens_per_context = self.context_len * self.tokens_per_frame
        clean_latents_video = []
        clean_latents_audio = []

        for i in range(N):
            grad_frame  = i >= start_grad_at                     # last L₁ frames
            s_idx       = random.randint(0, len(t_schedule)-1)  # step index of the chosen (reversed) step
            x_t         = torch.randn(B, 1, C, H, W, device=device) # sample x_t at the *largest* timestep

            for step_idx, t in enumerate(reversed(t_schedule)):
                # enable grad **only** on the chosen (reversed) step
                keep_grad = grad_frame and (step_idx == s_idx) and self.training
                x_t.detach_()                   # always detach x_t, so it starts as a leaf node
                x_t.requires_grad_(keep_grad)   # but reattach if keep_grad
                with torch.autocast(device_type=device.type, dtype=self.autocast):
                    self.kv_cache.enable_cache_updates()
                    # NOTE: ignore audio as _ for now? ask shab to double check :( 
                    x_0, audio_0 = self.model.core(
                        x        = x_t,
                        t        = t * torch.ones((self.batch_size, 1), device=self.device),
                        kv_cache = self.kv_cache,
                        mouse    = mouse [:, i:i+1],
                        btn      = btn   [:, i:i+1],
                        audio    = audio [:, i:i+1],
                    )
                    self.kv_cache.disable_cache_updates()
                    cache_overflow = len(self.kv_cache) - tokens_per_context
                    if cache_overflow > 0:
                        drop = -(-cache_overflow // self.tokens_per_frame)  # ceil division
                        self.kv_cache.truncate(drop)

                # move to the previous timestep unless we hit t=0
                if t != 0:
                    eps             = torch.randn_like(x_0)
                    _alpha, _sigma  = alpha(t), sigma(t)
                    x_t             = (_alpha * x_0) + (_sigma * eps)
                    audio_t         = (_alpha * audio_0) + (_sigma * eps)
                else:
                    break                            # reached fully-denoised

            # detach unless this frame carries grads
            clean_latents_video.append(x_0 if keep_grad else x_0.detach())
            clean_latents_audio.append(audio_0 if keep_grad else audio_0.detach())

        return torch.cat(clean_latents_video, dim=1), torch.cat(clean_latents_audio, dim=1)
