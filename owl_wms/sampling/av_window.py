import torch
from torch import nn
import torch.nn.functional as F

from tqdm import tqdm

from ..utils import batch_permute_to_length

def zlerp(x, alpha):
    z = torch.randn_like(x)
    return x * (1. - alpha) + z * alpha

class AVWindowSampler:
    """
    Window CFG Sampler samples new frames one by one, by inpainting the final frame.
    This is basically diffusion forcing.

    :param n_steps: Number of diffusion steps for each frame (diffusoin steps)
    :param cfg_scale: CFG scale for each frame
    :param window_length: Number of frames to use for each frame generation step
    :param num_frames: Number of new frames to sample
    :param noise_prev: Noise previous frame
    :param only_return_generated: Whether to only return the generated frames
    """
    def __init__(self, n_steps = 20, cfg_scale = 1.3, window_length = 60, num_frames = 60, noise_prev = 0.2, only_return_generated = False):
        self.n_steps = n_steps
        self.cfg_scale = cfg_scale
        self.window_length = window_length
        self.num_frames = num_frames
        self.noise_prev = noise_prev
        self.only_return_generated = only_return_generated

    @torch.no_grad()
    def __call__(self, model, dummy_batch, audio, mouse, btn, decode_fn = None, audio_decode_fn = None, image_scale = 1, audio_scale = 1):
        # dummy_batch is [b,n,c,h,w]
        # audio is [b,n,c] and should be treated same as video (it'e being generated)
        # mouse is [b,n,2]
        # btn is [b,n,n_button]

        # output will be [b,n+self.num_frames,c,h,w]
        
        sampling_steps = self.n_steps
        num_frames = self.num_frames

        dt = 1. / sampling_steps

        clean_history = dummy_batch.clone()
        clean_audio_history = audio.clone()
        
        extended_mouse, extended_btn = batch_permute_to_length(mouse, btn, num_frames + self.window_length)

        def step_history():
            # Video history
            new_history = clean_history.clone()[:,-self.window_length:] 
            b,n,c,h,w = new_history.shape
            new_history[:,:-1] = zlerp(new_history[:,1:],self.noise_prev)
            new_history[:,-1] = torch.randn_like(new_history[:,0])

            # Audio history 
            new_audio = clean_audio_history.clone()[:,-self.window_length:]
            new_audio[:,:-1] = zlerp(new_audio[:,1:],self.noise_prev)
            new_audio[:,-1] = torch.randn_like(new_audio[:,0])

            return new_history, new_audio

        for frame_idx in tqdm(range(num_frames)):
            local_history, local_audio = step_history()
            ts_history = torch.ones(local_history.shape[0], local_history.shape[1], device=local_history.device,dtype=local_history.dtype)
            ts_history[:,:-1] = self.noise_prev

            mouse = extended_mouse[:,frame_idx:frame_idx+self.window_length]
            btn = extended_btn[:,frame_idx:frame_idx+self.window_length]

            mouse_batch = torch.cat([mouse, torch.zeros_like(mouse)], dim=0) 
            btn_batch = torch.cat([btn, torch.zeros_like(btn)], dim=0)
            for _ in range(sampling_steps):
                # CFG Branches
                x = local_history.clone()
                a = local_audio.clone()
                ts = ts_history.clone()

                x_batch = torch.cat([x, x], dim=0)
                a_batch = torch.cat([a, a], dim=0)
                ts_batch = torch.cat([ts, ts], dim=0)
                
                pred_video_batch, pred_audio_batch = model(x_batch, a_batch, ts_batch, mouse_batch, btn_batch)
                
                # Split predictions back into conditional and unconditional
                cond_pred_video, uncond_pred_video = pred_video_batch.chunk(2)
                cond_pred_audio, uncond_pred_audio = pred_audio_batch.chunk(2)

                pred_video = uncond_pred_video + self.cfg_scale * (cond_pred_video - uncond_pred_video)
                pred_audio = uncond_pred_audio + self.cfg_scale * (cond_pred_audio - uncond_pred_audio)
                
                x = x - pred_video*dt
                a = a - pred_audio*dt
                ts = ts - dt

                local_history[:,-1] = x[:,-1]
                local_audio[:,-1] = a[:,-1]
                ts_history[:,-1] = ts[:,-1]
            
            # Frame is entirely cleaned now
            new_frame = local_history[:,-1:]
            new_audio = local_audio[:,-1:]
            clean_history = torch.cat([clean_history, new_frame], dim=1)
            clean_audio_history = torch.cat([clean_audio_history, new_audio], dim=1)

        x = clean_history
        audio = clean_audio_history
        if self.only_return_generated:
            x = x[:,-num_frames:]
            audio = audio[:,-num_frames:]
            extended_mouse = extended_mouse[:,-num_frames:]
            extended_btn = extended_btn[:,-num_frames:]

        if decode_fn is not None:
            x = x * image_scale
            x = decode_fn(x)

        if audio_decode_fn is not None:
            audio = audio * audio_scale
            audio = audio_decode_fn(audio)
    
        return x, audio, extended_mouse, extended_btn


class Inference_AV_WindowSampler:
    """
    Window CFG Sampler samples new frames one by one, by inpainting the final frame.
    This is basically diffusion forcing.

    :param n_steps: Number of diffusion steps for each frame (diffusoin steps)
    :param cfg_scale: CFG scale for each frame
    :param window_length: Number of frames to use for each frame generation step
    :param num_frames: Number of new frames to sample
    :param noise_prev: Noise previous frame
    :param only_return_generated: Whether to only return the generated frames
    """
    def __init__(self, n_steps = 20, cfg_scale = 1.3, window_length = 60, num_frames = 60, noise_prev = 0.2, only_return_generated = False):
        self.n_steps = n_steps
        self.cfg_scale = cfg_scale
        self.window_length = window_length
        self.num_frames = num_frames
        self.noise_prev = noise_prev
        self.only_return_generated = only_return_generated

    @torch.no_grad()
    def __call__(self, model, dummy_batch, audio, mouse, btn, decode_fn = None, audio_decode_fn = None, image_scale = 1, audio_scale = 1):
        # dummy_batch is [b,n,c,h,w]
        # audio is [b,n,c] and should be treated same as video (it'e being generated)
        # mouse is [b,n,2]
        # btn is [b,n,n_button]

        # output will be [b,n+self.num_frames,c,h,w]
        
        sampling_steps = self.n_steps
        num_frames = self.num_frames

        dt = 1. / sampling_steps

        clean_history = dummy_batch.clone()
        clean_audio_history = audio.clone()
        
        assert mouse.shape[1] == num_frames + self.window_length
        assert btn.shape[1]   == num_frames + self.window_length

        extended_mouse, extended_btn = mouse, btn

        def step_history():
            # Video history
            new_history = clean_history.clone()[:,-self.window_length:] 
            b,n,c,h,w = new_history.shape
            new_history[:,:-1] = zlerp(new_history[:,1:],self.noise_prev)
            new_history[:,-1] = torch.randn_like(new_history[:,0])

            # Audio history 
            new_audio = clean_audio_history.clone()[:,-self.window_length:]
            new_audio[:,:-1] = zlerp(new_audio[:,1:],self.noise_prev)
            new_audio[:,-1] = torch.randn_like(new_audio[:,0])

            return new_history, new_audio

        for frame_idx in tqdm(range(num_frames)):
            local_history, local_audio = step_history()
            ts_history = torch.ones(local_history.shape[0], local_history.shape[1], device=local_history.device,dtype=local_history.dtype)
            ts_history[:,:-1] = self.noise_prev

            mouse = extended_mouse[:,frame_idx:frame_idx+self.window_length]
            btn = extended_btn[:,frame_idx:frame_idx+self.window_length]

            mouse_batch = torch.cat([mouse, torch.zeros_like(mouse)], dim=0) 
            btn_batch = torch.cat([btn, torch.zeros_like(btn)], dim=0)
            for _ in range(sampling_steps):
                # CFG Branches
                x = local_history.clone()
                a = local_audio.clone()
                ts = ts_history.clone()

                x_batch = torch.cat([x, x], dim=0)
                a_batch = torch.cat([a, a], dim=0)
                ts_batch = torch.cat([ts, ts], dim=0)
                
                pred_video_batch, pred_audio_batch = model(x_batch, a_batch, ts_batch, mouse_batch, btn_batch)
                
                # Split predictions back into conditional and unconditional
                cond_pred_video, uncond_pred_video = pred_video_batch.chunk(2)
                cond_pred_audio, uncond_pred_audio = pred_audio_batch.chunk(2)

                pred_video = uncond_pred_video + self.cfg_scale * (cond_pred_video - uncond_pred_video)
                pred_audio = uncond_pred_audio + self.cfg_scale * (cond_pred_audio - uncond_pred_audio)
                
                x = x - pred_video*dt
                a = a - pred_audio*dt
                ts = ts - dt

                local_history[:,-1] = x[:,-1]
                local_audio[:,-1] = a[:,-1]
                ts_history[:,-1] = ts[:,-1]
            
            # Frame is entirely cleaned now
            new_frame = local_history[:,-1:]
            new_audio = local_audio[:,-1:]
            clean_history = torch.cat([clean_history, new_frame], dim=1)
            clean_audio_history = torch.cat([clean_audio_history, new_audio], dim=1)

        x = clean_history
        audio = clean_audio_history
        if self.only_return_generated:
            x = x[:,-num_frames:]
            audio = audio[:,-num_frames:]
            extended_mouse = extended_mouse[:,-num_frames:]
            extended_btn = extended_btn[:,-num_frames:]

        pixel_latents = x
        pixels = None

        audio_latents = audio
        audio_wav = None

        if decode_fn is not None:
            pixels = decode_fn(pixel_latents * image_scale)

        if audio_decode_fn is not None:
            audio_wav = audio_decode_fn(audio_latents * audio_scale)
    
        return (
            pixel_latents, audio_latents,  # NOTE Need this for history
            pixels, audio_wav,             # NOTE Need this for rendering
            extended_mouse, extended_btn,
            clean_history, clean_audio_history
        )


def test_av_window_sampler():
    sampler = AVWindowSampler()
    model = lambda x, ts, mouse, btn: x
    dummy_batch = torch.randn(1, 32, 128, 4, 4)
    audio = torch.randn(1, 32, 128)
    mouse = torch.zeros(1, 32, 2)
    btn = torch.zeros(1, 32, 11)

if __name__ == "__main__":
    test_av_window_sampler()
