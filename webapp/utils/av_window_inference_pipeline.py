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
                 sampling_steps: int                = 10,
                 audio_f: int                       = 735,
                 return_only_generated: bool        = True,
                 compile: bool                      = True,
                 device: str                        = 'cuda'):
        
        self.return_only_generated = return_only_generated
        self.config = config
        self.device = device
        
        self.model: GameRFTCore = get_model_cls(self.config.model.model_id)(self.config.model).core
        state_dict = torch  .load(ckpt_path, map_location="cpu")
        self.model          .load_state_dict(state_dict)
        self.model          .eval()
        self.model          .to(self.device)

        self.frame_decoder: Module = get_decoder_only(
            None,
            self.config.train.vae_cfg_path,
            self.config.train.vae_ckpt_path
        )
        self.frame_decoder.eval()
        self.frame_decoder.to(self.device)

        self.audio_decoder: Module = get_decoder_only(
            None,
            self.config.train.audio_vae_cfg_path,
            self.config.train.audio_vae_ckpt_path
        )
        self.audio_decoder.eval()
        self.audio_decoder.to(self.device)

        self.frame_scale    = self.config.train.vae_scale
        self.audio_scale    = self.config.train.audio_vae_scale

        self.history_buffer = (video_latent_history / self.frame_scale).to(self.device)
        self.audio_buffer   = (audio_latent_history / self.audio_scale).to(self.device)
        self.mouse_buffer   = mouse_history.to(self.device)
        self.button_buffer  = button_history.to(self.device)

        self.alpha          = alpha
        self.cfg_scale      = cfg_scale
        self.sampling_steps = sampling_steps
        self.audio_f        = audio_f

        if compile:
            print(f'Compiling models...')
            torch.compile(self.model)
            torch.compile(self.frame_decoder)
            torch.compile(self.audio_decoder)

    @print_duration
    @torch.no_grad()
    def __call__(self,
            user_input_mouse: torch.Tensor, # b,1,2
            user_input_button: torch.Tensor # b,1,11
        ) -> tuple[torch.Tensor, torch.Tensor]:

        noised_history      = zlerp(self.history_buffer[:,1:], self.alpha)
        noised_audio        = zlerp(self.audio_buffer[:,1:], self.alpha)

        noised_history      = torch.cat([noised_history, torch.randn_like(noised_history[:,0:1])], dim = 1)
        noised_audio        = torch.cat([noised_audio, torch.randn_like(noised_audio[:,0:1])], dim = 1)

        self.mouse_buffer   = torch.cat([self.mouse_buffer[:,1:],user_input_mouse],dim=1)
        self.button_buffer  = torch.cat([self.button_buffer[:,1:],user_input_button],dim=1)

        dt = 1. / self.sampling_steps

        x = noised_history
        a = noised_audio
        ts = torch.ones_like(noised_history[:,:,0,0,0])
        ts[:,:-1] = self.alpha

        # mouse_batch = torch.cat([self.mouse_buffer, torch.zeros_like(user_input_mouse)], dim=0) 
        # btn_batch = torch.cat([self.button_buffer, torch.zeros_like(user_input_button)], dim=0)
        # TODO Who knows bruh idk. I think this is to get cfg - uncond is just no actions (zeros)
        mouse_batch = torch.cat([self.mouse_buffer, torch.zeros_like(self.mouse_buffer)], dim=0) 
        btn_batch = torch.cat([self.button_buffer,  torch.zeros_like(self.button_buffer)], dim=0)

        for _ in range(self.sampling_steps):
            x_batch = torch.cat([x, x], dim=0)
            a_batch = torch.cat([a, a], dim=0)
            ts_batch = torch.cat([ts, ts], dim=0)

            video_rollout, audio_rollout = self.model(x_batch,a_batch,ts_batch,mouse_batch,btn_batch)

            cond_pred_video, uncond_pred_video = video_rollout.chunk(2)
            cond_pred_audio, uncond_pred_audio = audio_rollout.chunk(2)

            pred_video = uncond_pred_video + self.cfg_scale * (cond_pred_video - uncond_pred_video)
            pred_audio = uncond_pred_audio + self.cfg_scale * (cond_pred_audio - uncond_pred_audio)

            x[:,-1] = x[:,-1] - dt * pred_video[:,-1]
            a[:,-1] = a[:,-1] - dt * pred_audio[:,-1]
            ts[:,-1] = ts[:,-1] - dt
        
        new_frame = x[:,-1:] # [1,1,c,h,w]
        new_audio = a[:,-1:] # [1,1,c]

        self.history_buffer = torch.cat([self.history_buffer[:,1:], new_frame], dim=1)
        self.audio_buffer = torch.cat([self.audio_buffer[:,1:], new_audio], dim=1)

        frame = self.frame_decoder(new_frame[0]      * self.frame_scale).squeeze() # [c,h,w]
        audio = self.audio_decoder(
            self.audio_buffer.permute(0,2,1)  # need this as [b,c,t] for some reason
            * self.audio_scale
        ).squeeze()[-self.audio_f:].T # [735,2]

        return frame, audio
