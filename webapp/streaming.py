import math
import time
import torch
import asyncio
import numpy as np
from torch import nn
from functools import cache

from webapp.utils.samplers                  import create_sampler
from webapp.utils.configs                   import SamplingConfig, StreamingConfig
from owl_wms.configs                        import TrainingConfig, TransformerConfig as ModelConfig
from webapp.utils.visualize_overlay_actions import _draw_video as _draw_action_overlays

class FrameBuffer:
    """
    Manages frame streaming at precise timing, to adhere to a max FPS.
    We need this because sometimes we could generate frames faster than the max FPS,
    but, intuitively, the frame outputs will be the same 'set time' apart in the world state that we are approximating.
    """

    def __init__(self, streaming_config: StreamingConfig):
        self.streaming_config   = streaming_config
        self.video_frame_queue  = asyncio.Queue(maxsize=streaming_config.frames_per_batch * 2)  # Buffer 2 batches
        self.buttons_queue      = asyncio.Queue(maxsize=streaming_config.frames_per_batch * 2)  # Buffer 2 batches
        self.mouse_queue        = asyncio.Queue(maxsize=streaming_config.frames_per_batch * 2)  # Buffer 2 batches
        self.last_frame_time    = 0.0
        
    async def queue_frames(self, video_frames: torch.Tensor, mouse: torch.Tensor, button: torch.Tensor):
        # video_frames shape: [frames_per_batch, channels, height, width]
        # overlay_frames shape: [frames_per_batch, channels, overlay_height, width]
        num_frames = video_frames.shape[0]
        
        for i in range(num_frames):
            await self.video_frame_queue.put(video_frames[i])
            await self.buttons_queue.put(button[i])
            await self.mouse_queue.put(mouse[i])
    
    async def get_next_frames(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get next video and overlay frames for streaming at capped FPS."""
        now = time.time()
        time_since_last = now - self.last_frame_time
        time_to_wait    = max(0, self.streaming_config.frame_interval - time_since_last)

        if time_to_wait > 0:
            await asyncio.sleep(time_to_wait)
        
        video_frame = await self.video_frame_queue.get()
        button      = await self.buttons_queue.get()
        mouse       = await self.mouse_queue.get()
        self.last_frame_time = time.time()
        return video_frame, button, mouse

class StreamingFrameGenerator:
    """Wraps WindowCFGSampler to generate frames."""
    
    def __init__(self, encoder: nn.Module, decoder: nn.Module,
                 streaming_config: StreamingConfig,
                 model_config: ModelConfig,
                 train_config: TrainingConfig,
                 sampling_config: SamplingConfig,
                 debug: bool = False):
        self.streaming_config = streaming_config
        self.model_config     = model_config
        self.train_config     = train_config
        self.sampling_config  = sampling_config
        self.debug            = debug

        self.encoder = encoder
        self.decoder = decoder
        # 
        # Create WindowCFGSampler for frame generation
        self.sample_window_fn = create_sampler('window', encoder, decoder,
                                             batch_size=1,
                                             sampling_steps=self.sampling_config.sampling_steps,
                                             vae_scale=self.sampling_config.vae_scale,
                                             cfg_scale=self.sampling_config.cfg_scale,
                                             window_length=self.sampling_config.window_length,
                                             num_frames=self.sampling_config.num_frames,
                                             noise_prev=self.sampling_config.noise_prev)
        # Initialize frame history as empty tensor
        self.latent_history: torch.Tensor = torch.tensor([], device=self.streaming_config.device)
        self.mouse_history:  torch.Tensor = torch.tensor([], device=self.streaming_config.device)
        self.button_history: torch.Tensor = torch.tensor([], device=self.streaming_config.device)

    def add_to_history(self, frame_batch: torch.Tensor, mouse_batch: torch.Tensor, button_batch: torch.Tensor):
        if self.latent_history.equal(torch.tensor([], device=self.streaming_config.device)):
            self.latent_history = frame_batch
            self.mouse_history  = mouse_batch
            self.button_history = button_batch
            return

        self.latent_history  = torch.cat([self.latent_history, frame_batch],    dim=0)
        self.mouse_history   = torch.cat([self.mouse_history, mouse_batch],     dim=0)
        self.button_history  = torch.cat([self.button_history, button_batch],   dim=0)
        # cap this at around 60 frames
        if self.latent_history.shape[0] > self.streaming_config.window_length:
            self.latent_history  = self.latent_history[-self.streaming_config.window_length:]
            self.mouse_history   = self.mouse_history [-self.streaming_config.window_length:]
            self.button_history  = self.button_history[-self.streaming_config.window_length:]

    def get_latent_history_batch(self) -> torch.Tensor:
        if self.latent_history.equal(torch.tensor([], device=self.streaming_config.device)):
            return self.dummy_batch
        
        return self.latent_history.unsqueeze(0)

    @property
    @cache
    def dummy_batch(self) -> torch.Tensor:
        """Dummy autoencoder latents for the sampler to initialize shapes."""
        tokens_h = tokens_w = int(math.sqrt(self.model_config.tokens_per_frame))
        dummy_frames = torch.randn(
            1, self.streaming_config.window_length,
            self.model_config.channels, tokens_h, tokens_w,
            device=self.streaming_config.device, dtype=torch.bfloat16)
        return dummy_frames
    

    def overlay_actions(self,
                        video: torch.Tensor,
                        mouse: torch.Tensor, button: torch.Tensor,
                        action_margin_px_height: int = 150) -> torch.Tensor:
        num_frames, channels, height, width = video.shape
        action_height = height + action_margin_px_height
        action_width  = width
        action_video  = torch.zeros((num_frames, action_height, action_width, channels), # [n h w c]
                                    device=self.streaming_config.device, dtype=torch.bfloat16)
        # Copy video into top portion of action video
        action_video[:, :height, :, :]    = video.permute(0, 2, 3, 1) # [n c h w] -> [n h w c]
        action_video_np: list[np.ndarray] = _draw_action_overlays(action_video, button, mouse)
        action_video                      = [torch.from_numpy(frame) for frame in action_video_np]
        action_video                      = torch.stack(action_video).permute(0, 3, 1, 2) # [n h w c] -> [n c h w]
        return action_video

    def create_overlay_only(self,
                           video: torch.Tensor,
                           mouse: torch.Tensor, button: torch.Tensor,
                           action_margin_px_height: int = 150) -> torch.Tensor:
        num_frames, channels, height, width = video.shape
        # Create overlay-only frames with just the action margin height
        action_video = torch.zeros((num_frames, action_margin_px_height, width, channels), # [n h w c]
                                   device=self.streaming_config.device, dtype=torch.bfloat16)
        action_video_np: list[np.ndarray] = _draw_action_overlays(action_video, button, mouse)
        action_video = [torch.from_numpy(frame) for frame in action_video_np]
        action_video = torch.stack(action_video).permute(0, 3, 1, 2) # [n h w c] -> [n c h w]
        return action_video

    async def generate_frames(self, mouse: torch.Tensor, button: torch.Tensor,
                                    action_margin_px_height: int = 150) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Generate window_length frames, return separate video and overlay frames for streaming.
        
        Args:
            mouse:  [window_length, 2] 
            button: [window_length, n_buttons]
            
        Returns:
            tuple: (video_frames, overlay_frames)
            video_frames: [frames_per_batch, 3, 256, 256] - pure video frames
            overlay_frames: [frames_per_batch, 3, action_margin_px_height, 256] - action overlay frames
        """
        if self.debug:
            num_frames  = mouse.shape[0]
            # Create gradient from white to black to white across columns
            col_indices = torch.arange(256, device=self.streaming_config.device)
            # Create gradient that goes from 1 to 0 to 1
            gradient = torch.where(
                col_indices < 128,
                1.0 - (col_indices / 127.0),  # First half: 1 to 0
                (col_indices - 128) / 127.0   # Second half: 0 to 1
            ).view(1, 1, 1, -1)  # [1, 1, 1, 256]
            full_frames = gradient.expand(num_frames, 3, 256, 256).to(torch.bfloat16)
            # to between 0 and 1
            full_frames = (full_frames - full_frames.min()) / (full_frames.max() - full_frames.min())
            # between 0 and 255
            full_frames = (full_frames * 255).to(torch.uint8)
        else:
            with torch.no_grad(), torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                latents, full_frames = self.sample_window_fn(dummy_batch=self.get_latent_history_batch(),
                                                             mouse=mouse.float().unsqueeze(0),
                                                             btn=button.float().unsqueeze(0))  # [1, window_length, 3, 256, 256]
                # remove batch dimension, then take only the frames we generated, since the WindowCFGSampler appends the history (which is of window_length=60)
                latents     = latents       [0, -self.sampling_config.num_frames:]
                full_frames = full_frames   [0, -self.sampling_config.num_frames:]
                # then, convert the frames to a pixel-range of [0-255] from [-1,1]
                full_frames = (full_frames + 1) / 2
                full_frames = (full_frames * 255).to(torch.uint8)
                self.add_to_history(latents, mouse, button)

        # Create overlay frames separately
        overlay_frames = self.create_overlay_only(full_frames, mouse, button, action_margin_px_height)
        
        return full_frames, overlay_frames  # [window_length, 3, 256, 256], [window_length, 3, action_margin_px_height, 256]

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        del self.encoder, self.decoder ; torch.cuda.empty_cache()
