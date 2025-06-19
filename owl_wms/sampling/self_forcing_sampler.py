import torch
import einops
import random
from torch import Tensor
from torch.nn import Module
from functools import cache

from owl_wms.models.gamerft_audio import GameRFTAudio
from owl_wms.nn.kv_cache import KVCache
from owl_wms.configs import TransformerConfig as ModelConfig, TrainingConfig
from owl_wms.utils.flow_match_scheduler import FlowMatchScheduler

_SIGMA_TABLE = None # NOTE Set at run-time so it inherits the right device

# NOTE t is one element tensor, or int
@cache
def sigma(t: int | Tensor) -> Tensor: 
    if isinstance(t, Tensor): return _SIGMA_TABLE[(t - 1).long()].to(torch.float32)
    else:                     return _SIGMA_TABLE[int(t) - 1].to(torch.float32)

def alpha(t: int | Tensor) -> Tensor: 
    return (1 - sigma(t).square()).sqrt()


# NOTE In sampling, t_bn is a tensor of shape [B, 1]
# NOTE In training, t_bn is a tensor of shape [B, N]
# NOTE In both cases, columns are identical across batch elements
# NOTE In both cases, x can be audio ([B, N, D]) or video ([B,N,C,H,W])

def q_sample(x: Tensor, t_bn: Tensor) -> Tensor:
    # x: [B, N, ...], t_bn: [B, 1] (sampling) or [B, N] (training)
    
    # Handle both sampling and training cases
    if t_bn.shape[1] == 1:  # Sampling: [B, 1] -> broadcast to [B, N]
        t_bn = t_bn.expand(-1, x.shape[1])  # [B, N]
    
    # Since all batches identical, use first row
    t_single = t_bn[0]        # [N]
    
    alphas = alpha(t_single)  # [N] 
    sigmas = sigma(t_single)  # [N]
    
    # Handle common cases
    if x.ndim == 3:  # Audio: [B, N, D]
        alphas = einops.repeat(alphas, 'n -> 1 n 1')
        sigmas = einops.repeat(sigmas, 'n -> 1 n 1')
    elif x.ndim == 5:  # Images: [B, N, C, H, W]  
        alphas = einops.repeat(alphas, 'n -> 1 n 1 1 1')
        sigmas = einops.repeat(sigmas, 'n -> 1 n 1 1 1')
    else:
        raise ValueError(f"Unsupported tensor shape: {x.shape}")
    
    eps = torch.randn_like(x)
    return (alphas * x) + (sigmas * eps), eps


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

        global _SIGMA_TABLE

        self.training = training
        self.autocast = autocast

        # -- models, hardware
        self.model: GameRFTAudio = model
        self.model_config = model_config
        self.device = next(model.parameters()).device
        self.tokens_per_frame = self.model_config.tokens_per_frame
        _SIGMA_TABLE = FlowMatchScheduler(num_inference_steps=1000, num_train_timesteps=1000).sigmas.to(self.device)

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
                               latent_conditioning: list[dict[str, Tensor]]) -> tuple[Tensor, Tensor, Tensor]:
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
        
        clean_latents_video, clean_latents_audio = [], []  # N frames, one for each N
        scores_video,        scores_audio        = [], []  # N frames, one for each (t == s_t)
        selected_timesteps                       = []      # N values, one for each t sampled from t_schedule

        for i in range(N):
            grad_frame  = i >= start_grad_at                        # last L₁ frames
            s_t         = random.choice(t_schedule)      # chosen step to keep grads on for
            x_t         = torch.randn(B, 1, C, H, W, device=device) # sample x_t at the *largest* timestep

            for t in reversed(t_schedule):
                # enable grad **only** on the chosen timestep
                keep_grad = grad_frame and (t == s_t) and self.training
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
                
                # -- ignore gradients for frames that are too far backwards, as calculated by frame_gradient_cutoff
                x_0     = x_0     if grad_frame else x_0.detach()
                audio_0 = audio_0 if grad_frame else audio_0.detach()

                # -- only keep track of the scores for the chosen timestep for few-step distillation
                scores_video       += [x_0]     if (t == s_t) else []
                scores_audio       += [audio_0] if (t == s_t) else []
                selected_timesteps += [t]       if (t == s_t) else []  # NOTE selected_timestamps is a list of randomly sampled N values

                # move to the previous timestep unless we hit t=0
                if t != 0:
                    x_t,     _ = q_sample(x_0,     t * torch.ones((B, 1), device=x_t.device))
                    audio_t, _ = q_sample(audio_0, t * torch.ones((B, 1), device=audio_0.device))
                else: break # reached fully-denoised

            # -- we never use these for gradients in self-forcing. this is because, to get the teacher's score,
            # we take the scores_video/scores_audio, re-noise them, and then get the score from the teacher on
            # the re-noised frames. these are only needed for inference.
            clean_latents_video += [x_0.detach()]
            clean_latents_audio += [audio_0.detach()]

        if self.training:
            return (torch.cat(scores_video,          dim=1),
                    torch.cat(scores_audio,          dim=1),
                    torch.tensor(selected_timesteps, device=device).repeat(B, 1))

        return (torch.cat(clean_latents_video,   dim=1),
                torch.cat(clean_latents_audio,   dim=1),
                torch.tensor(selected_timesteps, device=device).repeat(B, 1))
