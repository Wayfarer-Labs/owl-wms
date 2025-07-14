import time
import torch
import asyncio
from torch import nn

from webapp.utils.configs                           import StreamingConfig, SamplingConfig
from webapp.utils.av_window_inference_pipeline      import AV_WindowInferencePipeline
from owl_wms.configs                                import Config as RunConfig


class FrameBuffer:
    """
    Manages frame streaming at precise timing, to adhere to a max FPS.
    """

    def __init__(self, streaming_config: StreamingConfig):
        self.streaming_config   = streaming_config
        self.video_frame_queue  = asyncio.Queue(maxsize=streaming_config.frames_per_batch * 2)
        self.audio_frame_queue  = asyncio.Queue(maxsize=streaming_config.frames_per_batch * 2)  # Add audio queue
        self.buttons_queue      = asyncio.Queue(maxsize=streaming_config.frames_per_batch * 2)
        self.mouse_queue        = asyncio.Queue(maxsize=streaming_config.frames_per_batch * 2)
        self.last_frame_time    = 0.0
        
    async def queue_frames(self, 
                          video_frames: torch.Tensor,  # [t,c,h,w]
                          audio_frames: torch.Tensor,  # [t,?,2]
                          mouse: torch.Tensor, 
                          button: torch.Tensor):
        # video_frames shape: [frames_per_batch, channels, height, width]
        # audio_frames shape: [frames_per_batch, 2]
        num_frames = video_frames.shape[0]
        
        for i in range(num_frames):
            await self.video_frame_queue.put(video_frames[i])
            await self.audio_frame_queue.put(audio_frames[i])  # Queue audio frames
            await self.buttons_queue.put(button[i])
            await self.mouse_queue.put(mouse[i])
    
    async def get_next_frames(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get next video, audio, and input frames for streaming at capped FPS."""
        now = time.time()
        time_since_last = now - self.last_frame_time
        time_to_wait    = max(0, self.streaming_config.frame_interval - time_since_last)

        if time_to_wait > 0:
            await asyncio.sleep(time_to_wait)
        
        video_frame = await self.video_frame_queue.get()
        audio_frame = await self.audio_frame_queue.get()
        button      = await self.buttons_queue.get()
        mouse       = await self.mouse_queue.get()
        self.last_frame_time = time.time()
        return video_frame, audio_frame, button, mouse


class StreamingFrameGenerator:
    """Wraps WindowCFGSampler to generate frames."""
    
    def __init__(self,
                 streaming_config:  StreamingConfig,
                 run_config:        RunConfig,
                 debug:             bool = False):
        
        self.run_config       = run_config
        self.streaming_config = streaming_config

        self.debug            = debug


        self.av_window_inference_pipeline = AV_WindowInferencePipeline(
            config                  = self.run_config,
            ckpt_path               = self.streaming_config.model_checkpoint_path,
            video_latent_history    = self.streaming_config.video_latent_history,
            audio_latent_history    = self.streaming_config.audio_latent_history,
            mouse_history           = self.streaming_config.mouse_history,
            button_history          = self.streaming_config.button_history,
            return_only_generated   = True,
            compile                 = True,
            with_audio              = self.streaming_config.with_audio
        )


    async def generate_frames(self, mouse: torch.Tensor, button: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Generate window_length frames, return separate video and overlay frames for streaming.
        
        Args:
            mouse:  [window_length, 2] , user input mouse
            button: [window_length, n_buttons] , user input button
            
        Returns:
            tuple: (video_frames, audio_frames)
            video_frames: [frames_per_batch, 3, 256, 256] - pure video frames
            audio_frames: [frames_per_batch, 2] - audio frames
        """
        mouse = mouse.to(self.streaming_config.device)
        button = button.to(self.streaming_config.device)

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
            audio_frames = torch.randn(num_frames, 2, device=self.streaming_config.device, dtype=torch.bfloat16)
            # to between 0 and 1
            full_frames = (full_frames - full_frames.min()) / (full_frames.max() - full_frames.min())
            # between 0 and 255
            full_frames = (full_frames * 255).to(torch.uint8)
        else:
            with torch.no_grad():
                full_frames, audio_frames = self.av_window_inference_pipeline(
                    user_input_mouse=mouse.bfloat16().unsqueeze(0),  # NOTE Need batch dimension
                    user_input_button=button.bfloat16().unsqueeze(0) # NOTE Need batch dimension
                )  # [3, 256, 256], [f, 2]

                # convert the frames to a pixel-range of [0-255] from [-1,1]
                full_frames  = (full_frames  + 1) / 2 # NOTE for some reason this is slightly off of [-1, 1]
                full_frames  = (full_frames  * 255).to(torch.uint8) # [3, 256, 256]
                full_frames  = torch.clip(full_frames, 0, 255) # bandaid for the [-1,1]

        return full_frames.unsqueeze(0), audio_frames.unsqueeze(0)  # [t, 3, 256, 256], [t, f, 2] where t = 1 for frame-by-frame rollouts


    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        torch.cuda.empty_cache()
