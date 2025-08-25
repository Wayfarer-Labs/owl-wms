from owl_wms.models import get_model_cls
from owl_wms.utils.owl_vae_bridge import get_decoder_only
from owl_wms.configs import Config
from owl_wms.data import get_loader

import torch.nn.functional as F
import torch

import random
import torch
from webapp.utils.configs import StreamingConfig
from accelerate import init_empty_weights
import os
import time

def zlerp(x, alpha):
    return x * (1. - alpha) + alpha * torch.randn_like(x)

@torch.no_grad()
def to_bgr_uint8(frame, target_size=(1080,1920)):
    # frame is [rgb,h,w] in [-1,1]
    frame = frame.flip(0)
    frame = frame.permute(1,2,0)
    frame = (frame + 1) * 127.5
    frame = frame.clamp(0, 255).to(device='cpu',dtype=torch.uint32,memory_format=torch.contiguous_format,non_blocking=True)
    return frame

class CausvidPipeline:
    def __init__(self, cfg_path="configs/dit_v4_dmd.yml", ckpt_path="vid_dit_v4_dmd_7k.pt"):
        cfg = Config.from_yaml(cfg_path)
        model_cfg = cfg.model
        train_cfg = cfg.train
        
        self.model = get_model_cls(model_cfg.model_id)(model_cfg).core
        self.model.load_state_dict(torch.load(ckpt_path, map_location='cpu'))
        self.model = self.model.cuda().bfloat16().eval()

        self.frame_decoder = get_decoder_only(
            None,
            train_cfg.vae_cfg_path,
            train_cfg.vae_ckpt_path
        )
        self.frame_decoder = self.frame_decoder.cuda().bfloat16().eval()

        # Store scales as instance variables
        self.frame_scale = train_cfg.vae_scale
        self.audio_scale = train_cfg.audio_vae_scale
        self.image_scale = train_cfg.vae_scale  # Assuming this is the same as frame_scale

        self.history_buffer = None
        self.audio_buffer = None
        self.mouse_buffer = None
        self.button_buffer = None

        self.alpha = 0.2

        self.model = torch.compile(self.model, mode = 'max-autotune', dynamic = False, fullgraph = True)
        self.frame_decoder = torch.compile(self.frame_decoder, mode = 'max-autotune', dynamic = False, fullgraph = True)
        #self.audio_decoder = torch.compile(audio_decoder, mode = 'max-autotune', dynamic = False, fullgraph = True)

        self.audio_f = 735
        
        # Initialize buffers and cache their initial state for restart
        self.stream_config = StreamingConfig(None)
        self.device = 'cuda'
        
        self.init_buffers()

        self._initial_history_buffer = self.history_buffer.clone()
        self._initial_audio_buffer = self.audio_buffer.clone()
        self._initial_mouse_buffer = self.mouse_buffer.clone()
        self._initial_button_buffer = self.button_buffer.clone()

        self.sampling_steps = 1

        self.min_samps = 1
        self.max_samps = 20

    def init_buffers(self):
        # Load cached tensors from data_cache directory
        cache_dir = "data_cache"
        cache_idx = random.randint(0, 99)  # Use first cached sample
        
        self.history_buffer = torch.load(os.path.join(cache_dir, f"history_buffer_{cache_idx}.pt"))
        self.audio_buffer = torch.load(os.path.join(cache_dir, f"audio_buffer_{cache_idx}.pt")) 
        self.mouse_buffer = torch.load(os.path.join(cache_dir, f"mouse_buffer_{cache_idx}.pt"))
        self.button_buffer = torch.load(os.path.join(cache_dir, f"button_buffer_{cache_idx}.pt"))

        # Scale buffers (already on cuda and in bfloat16 from cache)
        self.history_buffer = self.history_buffer / self.frame_scale
        self.audio_buffer = self.audio_buffer / self.audio_scale

    def restart_from_buffer(self):
        """Restore buffers to their initial state."""
        self.history_buffer = self._initial_history_buffer.clone()
        self.audio_buffer = self._initial_audio_buffer.clone()
        self.mouse_buffer = self._initial_mouse_buffer.clone()
        self.button_buffer = self._initial_button_buffer.clone()
    
    def up_sampling_steps(self):
        self.sampling_steps = min(self.sampling_steps + 1, self.max_samps)
    
    def down_sampling_steps(self):
        self.sampling_steps = max(self.sampling_steps - 1, self.min_samps)

    @torch.no_grad()
    def __call__(self, new_mouse, new_btn):
        """
        new_mouse is [2,] bfloat16 tensor (assume cuda for both)
        new_btn is [11,] bool tensor indexing into [W,A,S,D,LSHIFT,SPACE,R,F,E,LMB,RMB] (i.e. true if key is currently pressed, false otherwise)

        return frame as [c,h,w] tensor in [-1,1]
        """
        new_mouse = new_mouse.bfloat16()
        new_btn = new_btn.bfloat16()

        # [2,] float and [11,] bool over [W,A,S,D,LSHIFT,SPACE,R,F,E,LMB,RMB]
        noised_history = zlerp(self.history_buffer[:,1:], self.alpha)
        noised_audio = zlerp(self.audio_buffer[:,1:], self.alpha)

        noised_history = torch.cat([noised_history, torch.randn_like(noised_history[:,0:1])], dim = 1)
        noised_audio = torch.cat([noised_audio, torch.randn_like(noised_audio[:,0:1])], dim = 1)

        new_mouse = new_mouse[None,None,:]
        new_btn = new_btn[None,None,:]

        self.mouse_buffer = torch.cat([self.mouse_buffer[:,1:],new_mouse],dim=1)
        self.button_buffer = torch.cat([self.button_buffer[:,1:],new_btn],dim=1)

        x = noised_history
        a = noised_audio
        ts = torch.ones_like(noised_history[:,:,0,0,0])
        ts[:,:-1] = self.alpha
        dt = 1.0 / self.sampling_steps

        start_time = time.time()
        for _ in range(self.sampling_steps):
            pred_vid_batch, pred_aud_batch = self.model(x, a, ts, self.mouse_buffer, self.button_buffer)
            x[:,-1] = x[:,-1] - dt*pred_vid_batch[:,-1]
            a[:,-1] = a[:,-1] - dt*pred_aud_batch[:,-1]
            ts[:,-1] -= dt
        end_time = time.time()
        
        new_frame = x[:,-1:] # [1,1,c,h,w]
        new_audio = a[:,-1:] # [1,1,c] - Fixed: was 'audio' instead of 'a'

        self.history_buffer = torch.cat([self.history_buffer[:,1:], new_frame], dim=1)
        self.audio_buffer = torch.cat([self.audio_buffer[:,1:], new_audio], dim=1)

        x_to_dec = new_frame[0] * self.image_scale
        a_to_dec = self.audio_buffer * self.audio_scale

        frame = self.frame_decoder(x_to_dec).squeeze() # [c,h,w]
        #audio = self.audio_decoder(a_to_dec).squeeze()[-self.audio_f:] # [735,2]

        frame = to_bgr_uint8(frame)
        return frame, end_time - start_time
    

if __name__ == "__main__":
    pipe = CausvidPipeline()

    # INSERT_YOUR_CODE
    # Simple test: initialize buffers, print their shapes, run a forward pass and print output frame shape

    # Print buffer shapes
    print("history_buffer shape:", pipe.history_buffer.shape if pipe.history_buffer is not None else None)
    print("audio_buffer shape:", pipe.audio_buffer.shape if pipe.audio_buffer is not None else None)
    print("mouse_buffer shape:", pipe.mouse_buffer.shape if pipe.mouse_buffer is not None else None)
    print("button_buffer shape:", pipe.button_buffer.shape if pipe.button_buffer is not None else None)

    # Prepare dummy mouse and button input (matching last dimension of mouse/button buffer)
    mouse_shape = pipe.mouse_buffer.shape[-1] if pipe.mouse_buffer is not None else 2
    button_shape = pipe.button_buffer.shape[-1] if pipe.button_buffer is not None else 11
    import torch

    with torch.no_grad():
        dummy_mouse = torch.zeros(2).bfloat16().cuda()
        dummy_button = torch.zeros(11).bool().cuda()

        # Run a single forward pass
        frame = pipe(dummy_mouse, dummy_button)
        print("Generated frame shape:", frame.shape)
    