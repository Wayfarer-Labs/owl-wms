import torch
import random
from torch import Tensor
from typing import Optional

from owl_wms.models.gamerft_audio       import GameRFTAudio
from owl_wms.nn.kv_cache                import KVCache
from owl_wms.configs                    import TransformerConfig as ModelConfig
from owl_wms.utils.flow_match_scheduler import FlowMatchScheduler
from contextlib import nullcontext

def fwd_rectified_flow(x: Tensor, t: Tensor, noise: Optional[Tensor] = None) -> tuple[Tensor, Tensor]:
    if x.ndim == 5: assert tuple(t.shape) == (x.shape[0], x.shape[1], 1, 1, 1)
    if x.ndim == 3: assert tuple(t.shape) == (x.shape[0], x.shape[1], 1)

    noise = noise or torch.randn_like(x)
    noise = noise.requires_grad_(False)
    return (1.-t).mul(x) + (t).mul(noise), noise

def _truncate_if_overflow(kv_cache: KVCache) -> None:
    # number of frames that the cache has overflown by
    # important to note than len(kv_cache) is the number of key tokens cached in one layer,
    # which we use to derive the number of frames.
    cache_overflow = (len(kv_cache) - kv_cache.max_length) // kv_cache.tokens_per_frame
    if cache_overflow > 0: kv_cache.truncate(cache_overflow)

def delta_t(t: float, n_steps: int) -> float:
    return 1. / n_steps

def _repeat_on_dim(x: Tensor, dim: int, n: int, *, repeat_as_null: bool = False) -> Tensor:
    repeat_as = x if not repeat_as_null else torch.zeros_like(x)
    repeats   = [repeat_as] * n
    return torch.cat([x, *repeats], dim=dim)

class SelfForcingSampler:
    def __init__(self,
            model: GameRFTAudio,
            model_config: ModelConfig,
            batch_size: int,
            latent_shape: tuple[int, int, int],
            denoising_steps: int = 20,
            context_len: int = 48,
            frame_gradient_cutoff: int = 8,
            num_gen_frames: int = 64,
            training: bool = False,
            autocast: torch.dtype = torch.bfloat16,
            cfg_scale: float = 1.3,
        ):
        self.training = training
        self.autocast = autocast

        # -- models, hardware
        self.model: GameRFTAudio = model
        self.model_config = model_config
        self.device = next(model.parameters()).device
        self.tokens_per_frame = self.model_config.tokens_per_frame

        # -- sampling
        self.denoising_steps = denoising_steps
        self.batch_size = batch_size
        self.context_len = context_len
        self.latent_shape = latent_shape
        self.num_gen_frames = num_gen_frames
        self.cfg_scale = cfg_scale
        # -- gradient optimisation
        self.kv_cache = KVCache(self.model_config, rank=self.device.index).to(device=self.device.type, rank=self.device.index)
        self.kv_cache.reset(self.batch_size * 2) # NOTE our batch-size doubles for classifier-free guidance cause we do (cat uncond cond) on the batchdim
        self.frame_gradient_cutoff = frame_gradient_cutoff
        self.start_grad_at = max(0, self.num_gen_frames - self.frame_gradient_cutoff)
        if self.start_grad_at <= 0:
            print(f'WARNING: {self.num_gen_frames=} <= {self.frame_gradient_cutoff=}')

        # -- validation
        assert self.frame_gradient_cutoff < self.context_len

    @torch.no_grad()
    def _warmup_kv(self, clip_bnchw: Tensor, audio: Tensor, mouse: Tensor, btn: Tensor):
        """Fill rolling KV cache without tracking grads."""
        self.kv_cache.reset(self.batch_size * 2) # NOTE doubles for cfg 
        self.kv_cache.enable_cache_updates()
        num_frames = clip_bnchw.shape[1]

        for frame_idx in range(num_frames):
            # -- no need to denoise multiple times since the clips are 
            # clean (groundtruth)
            _ = self.model.velocity_fn(
                x_t      = clip_bnchw  [:, frame_idx:frame_idx+1],
                audio    = audio       [:, frame_idx:frame_idx+1],
                t        = torch.zeros((self.batch_size, 1), device=self.device),
                mouse    = mouse       [:, frame_idx:frame_idx+1],
                btn      = btn         [:, frame_idx:frame_idx+1],
                kv_cache = self.kv_cache,
                cfg_weight = self.cfg_scale,
            )
        self.kv_cache.disable_cache_updates()
        _truncate_if_overflow(self.kv_cache)


    def autoregressive_rollout(self,
                               btn: Tensor,
                               mouse: Tensor,
                               audio: Tensor) -> dict[str, Tensor]:
        assert btn.shape[1] == mouse.shape[1] == self.num_gen_frames, \
            f'btn, mouse (history data) must have the same number of frames: \
                {self.num_gen_frames=} {btn.shape[1]=} {mouse.shape[1]=} {audio.shape[1]=}'
        
        B               = self.batch_size
        C, H, W         = self.latent_shape   # dims of latent of upstream autoencoder
        A               = audio.shape[-1]     # dims of audio
        denoising_steps = self.denoising_steps
        device          = self.device
        N               = self.num_gen_frames # number of frames to generate that are outside the context
        F               = getattr(self, 'frames_per_chunk', 1) # number of frames to process in one chunk. didnt implement this yet lol i dont wanna scroll up
        start_grad_at   = self.start_grad_at  # frame_idx past which we start keeping track of grads (vanishing error accumulation horizon)        

        clean_latents_video, clean_latents_audio = [], []  # N frames, one for each N
        selected_timesteps                       = []      # N values, one for each t sampled from t_schedule
        # -- choose random 5 consecutive steps from our schedule to keep gradients for
        start = random.randint(1, denoising_steps + 1 - 5)
        s_t   = set(i for i in range(start, start + 5))

        for i in range(N):
            # -- ignore gradients for frames that are too far backwards, as calculated by frame_gradient_cutoff
            # in self-forcing's repo this never happens as per their config.
            grad_frame  = i >= start_grad_at
            x_t         = torch.randn(B, F, C, H, W, device=device) # sample x_t at the *largest* timestep
            audio_t     = torch.randn(B, F, A,       device=device)
            timestep    = 1.0

            self.kv_cache.disable_cache_updates() # do not update cache while we are denoising
            for idx in range(1, denoising_steps + 1):
                dt = delta_t(idx, denoising_steps)      # for now this is just 1./denoising_steps but we can replace w euler later

                keep_grad = grad_frame and (idx in s_t) and self.training
                grad_ctxt = nullcontext() if keep_grad else torch.no_grad()
                # -- rectified flow denoising step with no_grad if we are not on the selected s_t
                with torch.autocast(device_type=device.type, dtype=self.autocast), grad_ctxt:
                    velocity_x_t, velocity_audio_t = self.model.velocity_fn(
                        x_t        = x_t,
                        audio      = audio_t,
                        t          = timestep * torch.ones((self.batch_size, F), device=self.device),
                        mouse      = mouse [:, i:i+1],
                        btn        = btn   [:, i:i+1],
                        kv_cache   = self.kv_cache,
                        cfg_weight = self.cfg_scale)

                    x_t     = x_t     - (dt * velocity_x_t)
                    audio_t = audio_t - (dt * velocity_audio_t)
                
                # https://arxiv.org/pdf/2506.08009 last paragraph of Section 3.2
                # "use the denoised output of the s-th step as the final output"
                # -- Unfortunately, we don't do flow matching so we don't have a clean
                # estimate until idx == N - otherwise, we'd break.
                # -- We don't break until we end up with a clean sample so that we 
                # can populate the KV Cache with a clean frame.
                if min(s_t) >= idx:
                    clean_latents_video += [x_t]
                    clean_latents_audio += [audio_t]
                
                timestep -= dt

            # -- update kv cache with clean frames (line 13, Algorithm 1)
            self.kv_cache.enable_cache_updates()
            with torch.no_grad():
                _ = self.model.velocity_fn(
                    x_t                = x_t,
                    audio              = audio_t,
                    t                  = torch.zeros((self.batch_size, F), device=self.device),
                    mouse              = mouse [:, i:i+1],
                    btn                = btn   [:, i:i+1],
                    kv_cache           = self.kv_cache,
                    cfg_weight         = self.cfg_scale)

            # -- technically, it is always keep_grad by the time we get here, because we break
            # right after our sampled timestep s
            selected_timesteps  += [min(s_t)]
            
        return {
            'clean_latents_video':  torch.cat(clean_latents_video, dim=1),
            'clean_latents_audio':  torch.cat(clean_latents_audio, dim=1),
            'selected_timesteps':   torch.tensor(selected_timesteps, device=device).repeat(B, 1),
        }
