import math
import time
import torch
import asyncio
from torch          import nn

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
        self.streaming_config = streaming_config
        self.frame_queue = asyncio.Queue(maxsize=streaming_config.frames_per_batch * 2)  # Buffer 2 batches
        self.last_frame_time = 0.0
        
    async def add_frame_batch(self, frame_batch: torch.Tensor):
        # frame_batch shape: [1, frames_per_batch, channels, height, width]
        batch_size, num_frames = frame_batch.shape[:2]
        
        for i in range(num_frames):
            frame = frame_batch[0, i]  # take first batch cause 1 user only, then [channels, height, width]
            await self.frame_queue.put(frame)
    
    async def get_next_frame(self) -> torch.Tensor:
        """Get next frame for streaming at capped FPS."""
        now = time.time()
        time_since_last = now - self.last_frame_time
        time_to_wait = max(0, self.streaming_config.frame_interval - time_since_last)
        
        if time_to_wait > 0:
            await asyncio.sleep(time_to_wait)
        
        frame = await self.frame_queue.get()
        self.last_frame_time = time.time()
        return frame

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
        # Create WindowCFGSampler for 8-frame generation
        self.sample_window_fn = create_sampler('window', encoder, decoder,
                                             batch_size=1,
                                             sampling_steps=self.sampling_config.sampling_steps,
                                             cfg_scale=self.sampling_config.cfg_scale,
                                             scale=self.train_config.vae_scale)
        # Initialize frame history as empty tensor
        self.frame_history: torch.Tensor = torch.tensor([], device=self.streaming_config.device)
        self.mouse_history: torch.Tensor = torch.tensor([], device=self.streaming_config.device)
        self.button_history: torch.Tensor = torch.tensor([], device=self.streaming_config.device)

    def add_to_history(self, frame_batch: torch.Tensor, mouse_batch: torch.Tensor, button_batch: torch.Tensor):
        if self.frame_history.equal(torch.tensor([], device=self.streaming_config.device)):
            self.frame_history  = frame_batch
            self.mouse_history  = mouse_batch
            self.button_history = button_batch
            return

        self.frame_history  = torch.cat([self.frame_history, frame_batch], dim=1)
        self.mouse_history  = torch.cat([self.mouse_history, mouse_batch], dim=1)
        self.button_history = torch.cat([self.button_history, button_batch], dim=1)
        # cap this at around 60 frames
        if self.frame_history.shape[1] > self.streaming_config.window_length:
            self.frame_history  = self.frame_history[:, -self.streaming_config.window_length:]
            self.mouse_history  = self.mouse_history[:, -self.streaming_config.window_length:]
            self.button_history = self.button_history[:, -self.streaming_config.window_length:]

    @property
    def dummy_batch(self) -> torch.Tensor:
        """Dummy autoencoder latents for the sampler to initialize shapes."""
        tokens_h = tokens_w = int(math.sqrt(self.model_config.tokens_per_frame))
        dummy_frames = torch.randn(
            1, self.streaming_config.window_length,
            self.model_config.channels, tokens_h, tokens_w,
            device=self.streaming_config.device, dtype=torch.bfloat16)
        return dummy_frames

    def debug_generate_frame_batch(self, mouse_batch: torch.Tensor, button_batch: torch.Tensor) -> torch.Tensor:
        # NOTE Debug mode, just draw overlays on a black video.
        num_frames = mouse_batch.shape[1]
        video = torch.zeros(num_frames, 256, 256, 3,
                            device=self.streaming_config.device,
                            dtype=torch.bfloat16)
        frames = _draw_action_overlays(video, button_batch[0, ::], mouse_batch[0, ::])
        if frames == []:
            return video

        frames = [torch.from_numpy(frame) for frame in frames]
        return torch.stack(frames).permute(0, 3, 1, 2).unsqueeze(0) # [1, num_frames, c, h, w]

    async def generate_frame_batch(self, mouse_batch: torch.Tensor, button_batch: torch.Tensor) -> torch.Tensor:
        """
        Generate window_length frames, return first frames_per_batch for streaming.
        
        Args:
            mouse_batch:  [1, window_length, 2] 
            button_batch: [1, window_length, n_buttons]
            
        Returns:
            frame_batch: [1, frames_per_batch, 3, 256, 256] - only streaming frames
        """
        if self.debug:
            return self.debug_generate_frame_batch(mouse_batch, button_batch)

        with torch.no_grad(), torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            latents, full_frames = self.sample_window_fn(dummy_batch=self.dummy_batch,
                                                         mouse=mouse_batch, btn=button_batch)  # [1, window_length, 3, 256, 256]
            self.add_to_history(latents[0, ::], mouse_batch[0, ::], button_batch[0, ::]) # ignore batch dimension (1 user per model)

        return full_frames  # [1, window_length, 3, 256, 256]

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        del self.encoder, self.decoder ; torch.cuda.empty_cache()
