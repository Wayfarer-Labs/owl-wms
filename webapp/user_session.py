import cv2
import time
import json
import torch
import base64
import asyncio
import termcolor
import numpy as np

from fastapi                    import WebSocket
from webapp.action_converter    import ActionCollector
from webapp.streaming           import StreamingFrameGenerator, FrameBuffer


class UserGameSession:
    """
    Orchestrates receiving actions from the UI, generating frames, and displaying them.
    """
    def __init__(self, frame_generator: StreamingFrameGenerator):
        self.frame_generator    = frame_generator
        self.action_collector   = ActionCollector(frame_generator.streaming_config)
        self.frame_buffer       = FrameBuffer(frame_generator.streaming_config)
        
    async def run_session(self, websocket: WebSocket):
        with self.frame_generator:
            print(termcolor.colored(f"Starting streaming session at {self.frame_generator.streaming_config.fps} FPS", "green"))
            print(termcolor.colored(f"Generating                    {self.frame_generator.streaming_config.frames_per_batch} frames per batch", "green"))
            print(termcolor.colored(f"Batch duration:               {self.frame_generator.streaming_config.batch_duration:.3f}s", "green"))
            
            async with asyncio.TaskGroup() as tg:
                tg.create_task(self._action_input_loop      (websocket))
                tg.create_task(self._frame_generation_loop  ())
                tg.create_task(self._frame_display_loop     (websocket))
    
    async def _action_input_loop(self, websocket: WebSocket):
        while True:
            try:
                message = await websocket.receive_text()
                action_data = json.loads(message)
                await self.action_collector.add_websocket_action(action_data)
            except Exception as e:
                # Check if this is a WebSocket disconnect
                if "websocket.close" in str(e) or "response already completed" in str(e) or "WebSocket" in str(e):
                    print(termcolor.colored("🔌 WebSocket disconnected - stopping action input", "yellow"))
                    break
                else:
                    print(f"Error processing action: {e}")
                    break  # Exit loop on any other error

    async def _frame_generation_loop(self):
        """Generate frame batches continuously."""
        print(termcolor.colored("Frame generation loop started", "green"))
        while True:
            try:
                # Collect multiple frames worth of actions, e.g. X frames, that have happened between the last frame generation and now.
                mouse, button = await self.action_collector.collect_actions()
                # Generate Y frames from X actions by taking the X[-1]'th action. Typically, X >> Y, because they are sampled at uncapped FPS from the UI,
                #  whereas Y frames are sampled from the model one at a time.
                video_frames, overlay_frames = await self.frame_generator.generate_frames(mouse, button)
                # Queue frames for streaming at a capped FPS. If model predictions speed up or slow down, it won't cause any dilation of frames being displayed.
                # However, if the model predictions are too slow, the frames will be displayed at a lower FPS than the capped FPS.
                await self.frame_buffer.queue_frames(video_frames, mouse, button)
            except Exception as e:
                import traceback
                print(termcolor.colored(f"Error in frame generation: {e} :\n {traceback.format_exc()}", "red"))
                await asyncio.sleep(0.05)  # Brief pause before retry
    
    async def _frame_display_loop(self, websocket: WebSocket):
        while True:
            try:
                # Check if WebSocket is still connected before processing
                if websocket.client_state.name != 'CONNECTED':
                    print(termcolor.colored("🔌 WebSocket no longer connected - stopping frame stream", "yellow"))
                    break
                
                video_frame, button, mouse = await self.frame_buffer.get_next_frames()
                await self._send_frames_to_client(websocket, video_frame, button, mouse)
            except Exception as e:
                # Check if this is a WebSocket disconnect
                if ("websocket.close" in str(e) or 
                    "response already completed" in str(e) or
                    "Cannot call \"send\" once a close message has been sent" in str(e) or
                    "RuntimeError" in str(e)):
                    print(termcolor.colored("🔌 WebSocket disconnected - stopping frame stream", "yellow"))
                    break
                else:
                    import traceback
                    print(termcolor.colored(f"Error in frame streaming: {e} :\n {traceback.format_exc()}", "red"))
                    await asyncio.sleep(self.frame_generator.streaming_config.frame_interval)
    
    async def _send_frames_to_client(self,
                                     websocket: WebSocket,
                                     video_frame: torch.Tensor,
                                     button: torch.Tensor,
                                     mouse: torch.Tensor):
        try:
            # Check WebSocket state before sending
            if websocket.client_state.name != 'CONNECTED':
                raise RuntimeError("WebSocket is not connected")
            
            # TODO Do this more intelligently. I'm sure there's better tech to stream video to a UI.
            # Convert video frame to base64 JPEG
            video_frame_np = video_frame.float().cpu().numpy().transpose(1, 2, 0)  # CHW -> HWC
            # Normalize frame data to 0-255 range
            if video_frame_np.max() <= 1.0:
                video_frame_np = (video_frame_np * 255).clip(0, 255).astype(np.uint8)
            else:
                video_frame_np = video_frame_np.clip(0, 255).astype(np.uint8)
            
            _, video_buffer = cv2.imencode('.jpg', video_frame_np)
            video_base64 = base64.b64encode(video_buffer).decode('utf-8')
            
            
            await websocket.send_json({
                "type":         "frame",
                "video_data":   video_base64,
                "button_data":  self.action_collector.converter.buttons_to_dict(button),
                "mouse_data":   self.action_collector.converter.mouse_to_dict(mouse),
                "timestamp":    time.time()
            })
        except Exception as e:
            # Re-raise to be caught by the display loop
            raise e
    # TODO Audio
