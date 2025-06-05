import math
import time
import torch
import asyncio
from dataclasses    import dataclass
from torch          import nn

from owl_wms.sampling.cfg           import WindowCFGSampler
from owl_wms.utils.owl_vae_bridge   import make_batched_decode_fn
from owl_wms.configs                import TrainingConfig, WindowSamplingConfig, TransformerConfig as ModelConfig


@dataclass
class StreamingConfig:
    fps: int = 20
    frames_per_batch: int = 8
    window_length: int = 60
    device: str = 'cuda'
    n_buttons: int = 11
    mouse_range: tuple[float, float] = (-1.0, 1.0)
    
    @property
    def frame_interval(self) -> float:
        return 1.0 / self.fps

    @property
    def batch_duration(self) -> float:
        return self.frames_per_batch / self.fps

class FrameBuffer:
    """Manages frame streaming at precise timing."""
    
    def __init__(self, streaming_config: StreamingConfig):
        self.streaming_config = streaming_config
        self.frame_queue = asyncio.Queue(maxsize=streaming_config.frames_per_batch * 2)  # Buffer 2 batches
        self.last_frame_time = 0.0
        
    async def add_frame_batch(self, frame_batch: torch.Tensor):
        """Add a batch of frames to the streaming queue."""
        # frame_batch shape: [1, frames_per_batch, channels, height, width]
        batch_size, num_frames = frame_batch.shape[:2]
        
        for i in range(num_frames):
            frame = frame_batch[0, i]  # take first batch cause 1 user only, then [channels, height, width]
            await self.frame_queue.put(frame)
    
    async def get_next_frame(self) -> torch.Tensor:
        """Get next frame for streaming at precise timing."""
        # Calculate when to release the next frame
        now = time.time()
        time_since_last = now - self.last_frame_time
        time_to_wait = max(0, self.streaming_config.frame_interval - time_since_last)
        
        if time_to_wait > 0:
            await asyncio.sleep(time_to_wait)
        
        frame = await self.frame_queue.get()
        self.last_frame_time = time.time()
        return frame

class StreamingFrameGenerator:
    """Wraps WindowCFGSampler for real-time 8-frame batch generation."""
    
    def __init__(self, encoder: nn.Module, decoder: nn.Module,
                 streaming_config: StreamingConfig,
                 model_config: ModelConfig,
                 train_config: TrainingConfig,
                 sampling_config: WindowSamplingConfig):
        self.streaming_config = streaming_config
        self.model_config     = model_config
        self.train_config     = train_config
        self.sampling_config  = sampling_config

        self.encoder = encoder
        self.decoder = decoder
        
        # Create WindowCFGSampler for 8-frame generation
        self.window_sampler = WindowCFGSampler(
            window_length=self.streaming_config.window_length,
            num_frames=self.streaming_config.frames_per_batch,  # 8 frames per batch
            noise_prev=self.sampling_config.noise_prev,
            cfg_scale=self.sampling_config.cfg_scale
        )
        
        # Create batched decode function
        self.decode_fn = make_batched_decode_fn(decoder, batch_size=8)
        
        # Initialize frame history (60 frames of dummy data)
        self.frame_history = self._initialize_frame_history()
    
    def _initialize_frame_history(self) -> torch.Tensor:
        """Initialize with dummy frame history for cold start."""
        # Generate random latent frames matching model's expected input
        # Shape: [1, window_length, channels, height, width]
        tokens_h = tokens_w = int(math.sqrt(self.model_config.tokens_per_frame))
        dummy_frames = torch.randn(
            1, self.streaming_config.window_length,
            self.model_config.channels, tokens_h, tokens_w,
            device=self.streaming_config.device, dtype=torch.float32
        )
        return dummy_frames
    
    async def generate_frame_batch(self, mouse_batch: torch.Tensor, button_batch: torch.Tensor) -> torch.Tensor:
        """
        Generate 8 frames using WindowCFGSampler.
        
        Args:
            mouse_batch: [1, 8, 2] 
            button_batch: [1, 8, n_buttons]
            
        Returns:
            frame_batch: [1, 8, 3, 256, 256] - decoded RGB frames
        """
        # Use current frame history as dummy_batch for the window sampler
        dummy_batch = self.frame_history  # [1, 60, channels, h, w]
        
        # Generate new frames
        with torch.no_grad():
            new_frames = self.window_sampler(
                model=self.encoder,
                dummy_batch=dummy_batch,
                mouse=mouse_batch,
                btn=button_batch,
                decode_fn=self.decode_fn,
                scale=self.train_config.vae_scale
            )
        
        # Update frame history by sliding window
        # Remove oldest 8 frames, add newest 8 frames
        new_history = torch.cat([
            self.frame_history[:, self.streaming_config.frames_per_batch:],     # Remove first 8
            new_frames                                                          # Add new 8 frames
        ], dim=1)
        self.frame_history = new_history
        return new_frames

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        del self.encoder, self.decoder ; torch.cuda.empty_cache()

