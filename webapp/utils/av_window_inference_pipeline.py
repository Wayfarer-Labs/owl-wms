import os
import time
import torch
from torch.nn import Module
from owl_wms.models import get_model_cls
from owl_wms.utils.owl_vae_bridge import get_decoder_only
from owl_wms.configs import Config as RunConfig
from owl_wms.models.gamerft_audio import GameRFTCore

def zlerp(x, alpha):
    return x * (1. - alpha) + alpha * torch.randn_like(x)


def print_duration(func):
    """Decorator that logs the input and output of a function."""
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"{func.__name__} took {execution_time:.3f} seconds to execute, which would yield FPS of {1/execution_time:.3f}")
        return result
    return wrapper


class AV_WindowInferencePipeline:
    def __init__(self,
                 config: RunConfig,
                 video_latent_history: torch.Tensor,
                 audio_latent_history: torch.Tensor,
                 mouse_history: torch.Tensor,
                 button_history: torch.Tensor,
                 ckpt_path: str                     = "av_dfot_35k_ema_200m.pt",
                 alpha: float                       = 0.2,
                 cfg_scale: float                   = 1.3,
                 sampling_steps: int                = 4,
                 audio_f: int                       = 735,
                 return_only_generated: bool        = True,
                 compile: bool                      = True,
                 with_audio: bool                   = True,
                 device: str                        = 'cuda'):
        
        self.return_only_generated = return_only_generated
        self.config = config
        self.device = device
        self.with_audio = with_audio

        self.model: GameRFTCore = get_model_cls(self.config.model.model_id)(self.config.model).core
        state_dict = torch  .load(ckpt_path, map_location="cpu")
        self.model          .load_state_dict(state_dict)
        self.model          .eval().bfloat16()
        self.model          .to(self.device)

        self.frame_decoder: Module = get_decoder_only(
            None,
            self.config.train.vae_cfg_path,
            self.config.train.vae_ckpt_path
        )
        self.frame_decoder.eval().bfloat16()
        self.frame_decoder.to(self.device)

        if self.with_audio:
            self.audio_decoder: Module = get_decoder_only(
                None,
                self.config.train.audio_vae_cfg_path,
                self.config.train.audio_vae_ckpt_path
            )
            self.audio_decoder.eval().bfloat16()
            self.audio_decoder.to(self.device)

        self.frame_scale    = self.config.train.vae_scale
        if self.with_audio:
            self.audio_scale    = self.config.train.audio_vae_scale

        self.history_buffer = (video_latent_history / self.frame_scale).to(self.device).bfloat16()
        if self.with_audio:
            self.audio_buffer   = (audio_latent_history / self.audio_scale).to(self.device).bfloat16()
        self.mouse_buffer   = mouse_history.to(self.device).bfloat16()
        self.button_buffer  = button_history.to(self.device).bfloat16()

        self.alpha          = alpha
        self.cfg_scale      = cfg_scale
        self.sampling_steps = sampling_steps
        self.audio_f        = audio_f

        if compile:
            print(f'Compiling models...')
            torch.compile(self.model)
            torch.compile(self.frame_decoder)
            if self.with_audio:
                torch.compile(self.audio_decoder)

    @print_duration
    @torch.no_grad()
    def __call__(self,
            user_input_mouse: torch.Tensor, # b,1,2
            user_input_button: torch.Tensor # b,1,11
        ) -> tuple[torch.Tensor, torch.Tensor]:

        noised_history      = zlerp(self.history_buffer[:,1:], self.alpha)
        if self.with_audio:
            noised_audio        = zlerp(self.audio_buffer[:,1:], self.alpha)

        noised_history      = torch.cat([noised_history, torch.randn_like(noised_history[:,0:1])], dim = 1)
        if self.with_audio:
            noised_audio        = torch.cat([noised_audio, torch.randn_like(noised_audio[:,0:1])], dim = 1)

        self.mouse_buffer   = torch.cat([self.mouse_buffer[:,1:],user_input_mouse],dim=1)
        self.button_buffer  = torch.cat([self.button_buffer[:,1:],user_input_button],dim=1)

        dt = 1. / self.sampling_steps

        x = noised_history
        if self.with_audio:
            a = noised_audio
        ts = torch.ones_like(noised_history[:,:,0,0,0])
        ts[:,:-1] = self.alpha

        mouse_batch = self.mouse_buffer
        btn_batch = self.button_buffer

        # Time the denoising steps
        denoising_start = time.time()
        for step in range(self.sampling_steps):
            step_start = time.time()
            x_batch = x
            if self.with_audio:
                a_batch = a
            ts_batch = ts

            if self.with_audio:
                video_rollout, audio_rollout = self.model(x_batch,a_batch,ts_batch,mouse_batch,btn_batch)
            else:
                video_rollout = self.model(x_batch,ts_batch,mouse_batch,btn_batch)

            cond_pred_video = video_rollout
            if self.with_audio:
                cond_pred_audio = audio_rollout

            pred_video = cond_pred_video
            if self.with_audio:
                pred_audio = cond_pred_audio

            x[:,-1] = x[:,-1] - dt * pred_video[:,-1]
            if self.with_audio:
                a[:,-1] = a[:,-1] - dt * pred_audio[:,-1]
            ts[:,-1] = ts[:,-1] - dt
            
            step_time = time.time() - step_start    
            print(f"Denoising step {step} took {step_time:.3f} seconds")
        
        total_denoising_time = time.time() - denoising_start
        print(f"Total denoising time ({self.sampling_steps} steps) took {total_denoising_time:.3f} seconds")
        
        new_frame = x[:,-1:] # [1,1,c,h,w]
        if self.with_audio:
            new_audio = a[:,-1:] # [1,1,c]

        self.history_buffer = torch.cat([self.history_buffer[:,1:], new_frame], dim=1)
        if self.with_audio:
            self.audio_buffer = torch.cat([self.audio_buffer[:,1:], new_audio], dim=1)

        # Time frame decoding
        frame_start = time.time()
        frame = self.frame_decoder(new_frame[0] * self.frame_scale).squeeze() # [c,h,w]
        frame_time = time.time() - frame_start
        print(f"Frame decoding took {frame_time:.3f} seconds")

        # Time audio decoding
        audio = torch.zeros(self.audio_f, 2)
        if self.with_audio:
            audio_start = time.time()
            audio = self.audio_decoder(
                self.audio_buffer[:,-self.audio_buffer.shape[1]:].permute(0,2,1)  # need this as [b,c,t] for some reason
                * self.audio_scale
            ).squeeze()[-self.audio_f:].T # [735,2]
            audio_time = time.time() - audio_start
            print(f"Audio decoding took {audio_time:.3f} seconds")

        return frame, audio
