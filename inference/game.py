import pygame as pg
import numpy as np
import torch
from .causvid_pipeline import CausvidPipeline

import time
import os
os.environ['SDL_AUDIODRIVER'] = 'dummy'

class DummyPipeline:
    def __init__(self, *args, **kwargs):
        self.sampling_steps = 1
    def __call__(self,*args,**kwargs):
        time.sleep(0.003)
        x = torch.randn(3, 360, 640, device = 'cuda', dtype=torch.bfloat16)
        x = x.to(device='cpu',dtype=torch.uint32,memory_format=torch.contiguous_format,non_blocking=False)

class Game:
    def __init__(self, width=None, height=None, fps=60, mouse_scale=0.01, use_fast_display=True):
        pg.init()
        # If width/height not specified, use current display size (fullscreen)
        display_info = pg.display.Info()
        self.width = width if width is not None else display_info.current_w
        self.height = height if height is not None else display_info.current_h
        self.fps = fps
        self.mouse_scale = mouse_scale
        self.use_fast_display = use_fast_display
        
        # Use FULLSCREEN mode if width/height not specified
        # Remove OpenGL for WSL compatibility
        if width is None and height is None:
            self.screen = pg.display.set_mode((self.width, self.height), pg.FULLSCREEN | pg.HWSURFACE | pg.DOUBLEBUF)
        else:
            self.screen = pg.display.set_mode((self.width, self.height), pg.HWSURFACE | pg.DOUBLEBUF)
        pg.display.set_caption("Causvid Game")
        #self.clock = pg.time.Clock()
        self.font = pg.font.SysFont("Arial", 24)
        
        self.pipeline = CausvidPipeline()
        #self.pipeline = DummyPipeline()

        self.last_mouse_pos = None
        self.running = True

        # For FPS/latency display
        self.last_latency = 0.0
        self.last_fps = 0.0

        # Pre-allocate surfaces for better performance
        self.frame_surface = None
        self.scaled_surface = None
        self.last_frame_shape = None
        
        # Fast display buffer (not needed for current implementation)
        # if self.use_fast_display:
        #     self.display_buffer = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        
        # Set up initial mouse/button state
        self.n_mouse_axes = 2
        self.n_buttons = 11
        self.button_map = [
            pg.K_w, pg.K_a, pg.K_s, pg.K_d,
            pg.K_LSHIFT, pg.K_SPACE, pg.K_r, pg.K_f, pg.K_e,
            pg.BUTTON_LEFT, pg.BUTTON_RIGHT
        ]
        self.button_state = [False] * self.n_buttons

    def get_mouse_delta(self):
        mouse_pos = pg.mouse.get_pos()
        if self.last_mouse_pos is None:
            self.last_mouse_pos = mouse_pos
            return [0.0, 0.0]
        dx = mouse_pos[0] - self.last_mouse_pos[0]
        dy = mouse_pos[1] - self.last_mouse_pos[1]
        self.last_mouse_pos = mouse_pos
        # Scale and clamp to [-1, 1]
        dx = max(-1.0, min(1.0, dx * self.mouse_scale))
        dy = max(-1.0, min(1.0, dy * self.mouse_scale))
        return [dx, dy]

    def get_button_state(self):
        keys = pg.key.get_pressed()
        mouse_buttons = pg.mouse.get_pressed(num_buttons=5)
        state = []
        # Keyboard buttons
        for i in range(9):
            state.append(bool(keys[self.button_map[i]]))
        # Mouse buttons (left/right)
        state.append(bool(mouse_buttons[0]))  # left
        state.append(bool(mouse_buttons[2]))  # right
        return state

    def simple_display(self, frame):
        """Simple display method for WSL compatibility"""
        # Convert frame to numpy efficiently
        frame_np = frame.detach().cpu().float().numpy()
        
        # Optimize the conversion: do all operations in one go
        if frame_np.shape[0] == 3:
            # Optimized conversion for RGB
            frame_np = ((frame_np + 1) * 127.5).clip(0, 255).astype("uint8")
            frame_np = frame_np.transpose(1, 2, 0)
        elif frame_np.shape[0] == 1:
            # Optimized conversion for grayscale
            frame_np = ((frame_np + 1) * 127.5).clip(0, 255).astype("uint8")
            frame_np = np.repeat(frame_np, 3, axis=0).transpose(1, 2, 0)
        
        # Create surface directly (no scaling for simplicity)
        surf = pg.surfarray.make_surface(frame_np.swapaxes(0, 1))
        return surf

    def ultra_fast_display(self, frame):
        """Ultra-fast display using direct buffer manipulation"""
        # Convert frame to numpy efficiently
        frame_np = frame.detach().cpu().float().numpy()
        
        # Optimize the conversion: do all operations in one go
        if frame_np.shape[0] == 3:
            # Optimized conversion for RGB
            frame_np = ((frame_np + 1) * 127.5).clip(0, 255).astype("uint8")
            frame_np = frame_np.transpose(1, 2, 0)
        elif frame_np.shape[0] == 1:
            # Optimized conversion for grayscale
            frame_np = ((frame_np + 1) * 127.5).clip(0, 255).astype("uint8")
            frame_np = np.repeat(frame_np, 3, axis=0).transpose(1, 2, 0)
        
        # Resize frame to display size using numpy (faster than pygame transform)
        if frame_np.shape[:2] != (self.height, self.width):
            # Simple nearest neighbor scaling (much faster than smoothscale)
            h, w = frame_np.shape[:2]
            y_indices = np.linspace(0, h-1, self.height, dtype=np.int32)
            x_indices = np.linspace(0, w-1, self.width, dtype=np.int32)
            frame_np = frame_np[y_indices[:, None], x_indices]
        
        # Create surface directly from the processed frame (no buffer copy needed)
        surf = pg.surfarray.make_surface(frame_np.swapaxes(0, 1))
        return surf

    def optimize_frame_display(self, frame):
        """Optimized frame display with pre-allocated surfaces"""
        # Convert frame to numpy efficiently
        frame_np = frame.detach().cpu().float().numpy()
        
        # Optimize the conversion: do all operations in one go
        # frame_np: [c, h, w] in [-1,1], convert to [h, w, c] in [0,255]
        if frame_np.shape[0] == 3:
            # Optimized conversion for RGB
            frame_np = ((frame_np + 1) * 127.5).clip(0, 255).astype("uint8")
            frame_np = frame_np.transpose(1, 2, 0)
        elif frame_np.shape[0] == 1:
            # Optimized conversion for grayscale
            frame_np = ((frame_np + 1) * 127.5).clip(0, 255).astype("uint8")
            frame_np = np.repeat(frame_np, 3, axis=0).transpose(1, 2, 0)
        
        # Check if we need to recreate surfaces
        current_shape = frame_np.shape
        if self.last_frame_shape != current_shape:
            self.last_frame_shape = current_shape
            self.frame_surface = pg.surfarray.make_surface(frame_np.swapaxes(0, 1))
            #self.scaled_surface = pg.transform.smoothscale(self.frame_surface, (self.width, self.height))
            self.scaled_surface = self.frame_surface
        else:
            # Update existing surface directly (much faster)
            pg.surfarray.blit_array(self.frame_surface, frame_np.swapaxes(0, 1))
            #self.scaled_surface = pg.transform.smoothscale(self.frame_surface, (self.width, self.height))
            self.scaled_surface = self.frame_surface

        return self.scaled_surface

    def run(self):
        import torch
        while self.running:
            for event in pg.event.get():
                if event.type == pg.QUIT:
                    self.running = False
                elif event.type == pg.KEYDOWN:
                    if event.key == pg.K_y:
                        self.pipeline.init_buffers()
                    elif event.key == pg.K_u:
                        if hasattr(self.pipeline, "restart_from_buffer"):
                            self.pipeline.restart_from_buffer()
                    elif event.key == pg.K_f:
                        # Toggle fullscreen on F key
                        if self.screen.get_flags() & pg.FULLSCREEN:
                            self.screen = pg.display.set_mode((self.width, self.height), pg.HWSURFACE | pg.DOUBLEBUF)
                        else:
                            self.screen = pg.display.set_mode((self.width, self.height), pg.FULLSCREEN | pg.HWSURFACE | pg.DOUBLEBUF)
                    elif event.key == pg.K_o:
                        self.pipeline.up_sampling_steps()
                    elif event.key == pg.K_i:
                        self.pipeline.down_sampling_steps()
            # Get input
            mouse_delta = self.get_mouse_delta()
            button_state = self.get_button_state()

            # Prepare tensors
            mouse_tensor = torch.tensor(mouse_delta, dtype=torch.bfloat16).to('cuda')
            button_tensor = torch.tensor(button_state, dtype=torch.bool).to('cuda')

            # Model inference and timing
            t0 = time.time()
            frame = self.pipeline(mouse_tensor, button_tensor)
            t1 = time.time()
            latency = t1 - t0
            self.last_latency = latency
            self.last_fps = 1.0 / latency if latency > 0 else 0.0

            # Convert frame to numpy and display (optimized)
            t2 = time.time()
            try:
                if self.use_fast_display:
                    scaled_surface = self.ultra_fast_display(frame)
                else:
                    scaled_surface = self.optimize_frame_display(frame)
                self.screen.blit(scaled_surface, (0, 0))
            except Exception as e:
                print(f"Display error: {e}, trying simple display")
                scaled_surface = self.simple_display(frame)
                self.screen.blit(scaled_surface, (0, 0))
            t3 = time.time()

            pg_latency = round((t3 - t2)*1000,1)

            # Draw FPS/latency
            fps_text = f"FPS: {round(self.last_fps, 1)} | Latency: {round(self.last_latency*1000, 1)} ms | PG Latency: {pg_latency} ms | Steps: {self.pipeline.sampling_steps}"
            text_surf = self.font.render(fps_text, True, (0, 255, 0))
            self.screen.blit(text_surf, (10, 10))

            pg.display.flip()
            
            # Debug output
            #print(f"Frame shape: {frame.shape}, Latency: {latency*1000:.1f}ms, PG Latency: {pg_latency}ms")
            #self.clock.tick(self.fps)

        pg.quit() 


if __name__ == "__main__":
    game = Game()
    game.run()