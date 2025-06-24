import torch
import random
from torch import Tensor
from typing import Optional

from owl_wms.models.gamerft_audio       import GameRFTAudio
from owl_wms.nn.kv_cache                import KVCache
from owl_wms.configs                    import TransformerConfig as ModelConfig
from owl_wms.utils.flow_match_scheduler import FlowMatchScheduler


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

class SelfForcingSampler:
    def __init__(self,
            model: GameRFTAudio,
            model_config: ModelConfig,
            batch_size: int,
            latent_shape: tuple[int, int, int],
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
        self.model: GameRFTAudio = model
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
        self.kv_cache = KVCache(self.model_config, rank=self.device.index).to(device=self.device.type, rank=self.device.index)
        self.kv_cache.reset(self.batch_size)
        self.frame_gradient_cutoff = frame_gradient_cutoff
        self.start_grad_at = max(0, self.num_gen_frames - self.frame_gradient_cutoff)
        if self.start_grad_at <= 0:
            print(f'WARNING: {self.num_gen_frames=} <= {self.frame_gradient_cutoff=}')

        # -- validation
        assert self.frame_gradient_cutoff < self.context_len

    @torch.no_grad()
    def _warmup_kv(self, clip_bnchw: Tensor, audio: Tensor, mouse: Tensor, btn: Tensor):
        """Fill rolling KV cache without tracking grads."""
        self.kv_cache.reset(self.batch_size)
        self.kv_cache.enable_cache_updates()
        num_frames = clip_bnchw.shape[1]

        for frame_idx in range(num_frames):
            _ = self.model.core(
                x        = clip_bnchw  [:, frame_idx:frame_idx+1],
                audio    = audio       [:, frame_idx:frame_idx+1],
                t        = torch.zeros((self.batch_size, 1), device=self.device),
                mouse    = mouse       [:, frame_idx:frame_idx+1],
                btn      = btn         [:, frame_idx:frame_idx+1],
                kv_cache = self.kv_cache,
            )
        self.kv_cache.disable_cache_updates()
        _truncate_if_overflow(self.kv_cache)

    def autoregressive_rollout(self,
                               btn: Tensor,
                               mouse: Tensor,
                               audio: Tensor) -> dict[str, Tensor]:
        assert btn.shape[1] == mouse.shape[1] == audio.shape[1] == self.num_gen_frames, \
            f'btn, mouse, and audio must have the same number of frames: \
                {self.num_gen_frames=} {btn.shape[1]=} {mouse.shape[1]=} {audio.shape[1]=}'
        
        B             = self.batch_size
        C, H, W       = self.latent_shape   # dims of latent of upstream autoencoder
        A             = audio.shape[-1]     # dims of audio
        t_schedule    = self.t_schedule     # few-step distillation schedule
        device        = self.device
        N             = self.num_gen_frames # number of frames to generate that are outside the context
        F             = getattr(self, 'frames_per_chunk', 1) # number of frames to process in one chunk. didnt implement this yet lol i dont wanna scroll up
        start_grad_at = self.start_grad_at  # frame_idx past which we start keeping track of grads (vanishing error accumulation horizon)        

        # TODO is KV-Cache warmed up with a randomly generated t? No, it is generated with t=0.
        # See https://github.com/guandeh17/Self-Forcing/blob/a93f2f80ce60f4b022b0340d0026fca24d4f72a2/pipeline/self_forcing_training.py#L114-L128
        # And it doesn't matter, see below:
        # TODO is the initial conditioning noised? otherwise, we aren't denoising anything?
        #      ANSWER - SAMI: I think it doesn't matter if our denoising is valid, because the noise would be the Query and the latents
        #                     would be the Key and Value. As long as those get computed, we will get cache benefits,
        #                     and not whether we generate a valid output.
        # TODO: This should not be a separate step, I believe that this is done auto-regressively automatically.
        
        clean_latents_video, clean_latents_audio = [], []  # N frames, one for each N
        selected_timesteps                       = []      # N values, one for each t sampled from t_schedule
        s_t         = random.choice(t_schedule)            # chosen step to keep grads on for - TODO It seems s is sampled a "train_step" level meaning at batch level, otherwise kv cache frames wont be fully denoised

        for i in range(N):
            # -- ignore gradients for frames that are too far backwards, as calculated by frame_gradient_cutoff
            # in self-forcing's repo this never happens as per their config.
            grad_frame  = i >= start_grad_at                        # last L₁ frames
            x_t         = torch.randn(B, F, C, H, W, device=device) # sample x_t at the *largest* timestep
            audio_t     = torch.randn(B, F, A,       device=device)
            
            # -- prevent the final generation from attending to first frame.
            # since the first frame has unique statistical properties (it has no temporal compression),
            # generation degrades once it is no longer in context, which harms long video generation

            for t in reversed(t_schedule):
                dt = delta_t(t, len(t_schedule)) # for now this is just 1./t but we can replace w euler later

                keep_grad = grad_frame and (t == s_t) and self.training
                grad_ctxt = torch.enable_grad() if keep_grad else torch.no_grad()
                
                with torch.autocast(device_type=device.type, dtype=self.autocast):
                    self.kv_cache.disable_cache_updates() # -- do not update cache while we are denoising?
                    velocity_x_t, velocity_audio_t = self.model.core(
                        x                  = x_t,
                        audio              = audio_t,
                        t                  = t * torch.ones((self.batch_size, 1), device=self.device),
                        mouse              = mouse [:, i:i+1],
                        btn                = btn   [:, i:i+1],
                        kv_cache           = self.kv_cache
                    )
                    x_t     = x_t       - (dt * velocity_x_t)
                    audio_t = audio_t   - (dt * velocity_audio_t)

                # https://arxiv.org/pdf/2506.08009 last paragraph of Section 3.2
                # use the denoised output of the s-th step as the final output
                if t == 0 or t == s_t:
                    # update kv-cache by re-running with cleaned frames (line 13 in Algorithm 1)
                    self.kv_cache.enable_cache_updates()
                    with torch.no_grad():
                        _ = self.model.core(
                            x                  = x_t,
                            audio              = audio_t,
                            t                  = torch.zeros((self.batch_size, 1), device=self.device),
                            mouse              = mouse [:, i:i+1],
                            btn                = btn   [:, i:i+1],
                            kv_cache           = self.kv_cache,
                        )
                    _truncate_if_overflow(self.kv_cache)
                    break

            # -- technically, it is always keep_grad by the time we get here, because we break
            # right after our sampled timestep s
            clean_latents_video += [x_t]
            clean_latents_audio += [audio_t]
            selected_timesteps  += [t]  # technically always equal to s_t
            
        return {
            'clean_latents_video':  torch.cat(clean_latents_video, dim=1),
            'clean_latents_audio':  torch.cat(clean_latents_audio, dim=1),
            'selected_timesteps':   torch.tensor(selected_timesteps, device=device).repeat(B, 1),
        }
