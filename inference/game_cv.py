#!/usr/bin/env python3
"""
X11-based replacement for the pygame game loop.
  • Creates a 640 × 360 window (or user-supplied size)
  • Collects keyboard / mouse events            → builds 11-button vector
  • Computes scaled mouse deltas                → 2-vector
  • Feeds both to CausvidPipeline               → frame tensor [3,360,640]
  • Converts tensor → 24-bit RGB byte-buffer    → XPutImage
Latency & FPS are printed to stdout every second.
"""

from __future__ import annotations

from platform import system as psystem

if psystem != "darwin":
    import Xlib.Xatom as Xatom
    import Xlib.display
    import Xlib.X as X
    import Xlib.XK as XK
import numpy as np
import torch
import time
import torch.cuda

from .causvid_pipeline import CausvidPipeline
# from .game import DummyPipeline as CausvidPipeline


class GameCV:
    # Mapping from (keysym OR mouse-button) → position in the 11-button vector
    KEYMAP: dict[int, int] = {
        XK.XK_w: 0,
        XK.XK_a: 1,
        XK.XK_s: 2,
        XK.XK_d: 3,
        XK.XK_Shift_L: 4,
        XK.XK_space: 5,
        XK.XK_r: 6,
        XK.XK_f: 7,
        XK.XK_e: 8,
        # mouse buttons handled separately (1 & 3)
    }

    def __init__(self, width: int = 640, height: int = 360, mouse_scale: float = 0.01, fps: int = 60):
        self.width, self.height = width, height
        self.mouse_scale = mouse_scale
        self.target_frame_time = 1.0 / fps

        # X11 setup ----------------------------------------------------------
        self.disp = Xlib.display.Display()
        self.screen = self.disp.screen()
        self.win = self.screen.root.create_window(
            0,
            0,
            width,
            height,
            0,
            self.screen.root_depth,
            X.InputOutput,
            X.CopyFromParent,
            background_pixel=self.screen.black_pixel,
            event_mask=(X.ExposureMask | X.KeyPressMask | X.KeyReleaseMask | X.ButtonPressMask | X.ButtonReleaseMask | X.PointerMotionMask | X.StructureNotifyMask),
        )
        self.win.set_wm_name("Causvid Game - X11")
        self.gc = self.win.create_gc()
        self.win.map()

        # Handle graceful close via WM_DELETE_WINDOW
        self.WM_DELETE = self.disp.intern_atom("WM_DELETE_WINDOW")
        self.win.change_property(self.disp.intern_atom("WM_PROTOCOLS"), Xatom.ATOM, 32, [self.WM_DELETE])

        # Game state ---------------------------------------------------------
        self.pipeline = CausvidPipeline()
        self.button_state = [False] * 11
        self.last_mouse_pos: tuple[int, int] | None = None
        self.running = True

        # Stats
        self.pipe_fps_sum = 0.0  # pipeline-only
        self.total_fps_sum = 0.0  # pipeline + draw
        self.frame_counter = 0
        self.stats_t0 = time.time()

    # --------------------------------------------------------------------- #
    # Input Handling
    # --------------------------------------------------------------------- #
    def _handle_key(self, keysym: int, pressed: bool):
        # Quit on Escape / q
        if pressed and keysym in (XK.XK_Escape, XK.XK_q):
            self.running = False
            return

        # Pipeline control keys
        if pressed:
            if keysym == XK.XK_y:
                self.pipeline.init_buffers()
            elif keysym == XK.XK_u and hasattr(self.pipeline, "restart_from_buffer"):
                self.pipeline.restart_from_buffer()
            # Use up and down arrows to change mouse scaler
            if keysym == XK.XK_Up:
                self.pipeline.mouse_scaler += 0.01
                print(f"Mouse scaler: {self.pipeline.mouse_scaler}")
            elif keysym == XK.XK_Down:
                self.pipeline.mouse_scaler -= 0.01
                print(f"Mouse scaler: {self.pipeline.mouse_scaler}")

        # Regular movement / action keys
        if keysym in self.KEYMAP:
            self.button_state[self.KEYMAP[keysym]] = pressed

    def _handle_button(self, button: int, pressed: bool):
        if button == 1:  # Left mouse
            self.button_state[9] = pressed
        elif button == 3:  # Right mouse
            self.button_state[10] = pressed

    def _mouse_delta(self) -> list[float]:
        ptr = self.win.query_pointer()
        pos = (ptr.win_x, ptr.win_y)
        if self.last_mouse_pos is None:
            self.last_mouse_pos = pos
            return [0.0, 0.0]

        dx = (pos[0] - self.last_mouse_pos[0]) * self.mouse_scale
        dy = (pos[1] - self.last_mouse_pos[1]) * self.mouse_scale
        self.last_mouse_pos = pos
        # Clamp to [-1,1]
        dx = max(-1.0, min(1.0, dx))
        dy = max(-1.0, min(1.0, dy))
        return [dx, dy]

    # --------------------------------------------------------------------- #
    # Rendering helpers
    # --------------------------------------------------------------------- #
    @staticmethod
    def _tensor_to_ximage_bytes(frame: torch.Tensor) -> bytes:
        """
        Converts [3,H,W] RGB tensor in [-1,1] → little-endian 24-bit (packed 32) byte string.
        """
        np_frame = frame.numpy()
        # Convert to 0x00RRGGBB (little-endian 32-bit)
        r = np_frame[:, :, 0]
        g = np_frame[:, :, 1]
        b = np_frame[:, :, 2]
        packed = (b << 16) | (g << 8) | r
        return packed.flatten().tobytes()

    def _draw_frame(self, frame: torch.Tensor):
        data = self._tensor_to_ximage_bytes(frame)
        BPP = 4  # bytes per pixel (0x00RRGGBB)
        stride = self.width * BPP

        CHUNK_ROWS = 64  # keep every XPutImage < 200 kB
        for y in range(0, self.height, CHUNK_ROWS):
            h = min(CHUNK_ROWS, self.height - y)
            offset = y * stride
            self.win.put_image(
                self.gc,
                0,
                y,  # dest x,y
                self.width,
                h,
                X.ZPixmap,
                24,  # depth
                0,  # left pad
                data[offset : offset + h * stride],
            )

        self.disp.flush()

    # --------------------------------------------------------------------- #
    # Main loop
    # --------------------------------------------------------------------- #
    def run(self):
        while self.running:
            # ---------------- Event processing --------------------------- #
            while self.disp.pending_events():
                ev = self.disp.next_event()
                if ev.type == X.ClientMessage and ev.data[0] == self.WM_DELETE:
                    self.running = False
                elif ev.type in (X.KeyPress, X.KeyRelease):
                    keysym = self.disp.keycode_to_keysym(ev.detail, 0)
                    self._handle_key(keysym, ev.type == X.KeyPress)
                elif ev.type in (X.ButtonPress, X.ButtonRelease):
                    self._handle_button(ev.detail, ev.type == X.ButtonPress)
                # Ignore MotionNotify; we poll pointer each frame

            # ---------------- Inference & Render ------------------------- #
            mouse_delta = self._mouse_delta()
            mouse_tensor = torch.tensor(mouse_delta, dtype=torch.bfloat16, device="cuda")
            btn_tensor = torch.tensor(self.button_state, dtype=torch.bool, device="cuda")

            t_frame_start = time.time()

            # --- pipeline ------------------------------------------------ #
            frame, pipe_time = self.pipeline(mouse_tensor, btn_tensor)  # [3,360,640]

            # --- draw ---------------------------------------------------- #
            t1 = time.time()
            self._draw_frame(frame)
            draw_time = time.time() - t1  # seconds

            # --- accumulate stats --------------------------------------- #
            total_time = time.time() - t_frame_start  # = pipe_time+draw_time (+ε)
            self.pipe_fps_sum += 1.0 / max(pipe_time, 1e-6)
            self.total_fps_sum += 1.0 / max(total_time, 1e-6)
            self.frame_counter += 1

            # ---------------- Statistics -------------------------------- #
            now = time.time()
            if now - self.stats_t0 >= 1.0:
                avg_pipe_fps = self.pipe_fps_sum / max(self.frame_counter, 1)
                avg_total_fps = self.total_fps_sum / max(self.frame_counter, 1)
                print(f"[{time.strftime('%H:%M:%S')}] FPS (total): {avg_total_fps:5.1f} | FPS (pipeline): {avg_pipe_fps:5.1f} | Latency pipeline: {pipe_time * 1000:6.1f} ms | Latency draw: {draw_time * 1000:6.1f} ms")

                self.stats_t0 = now
                self.pipe_fps_sum = 0.0
                self.total_fps_sum = 0.0
                self.frame_counter = 0

        self.disp.close()


# ------------------------------------------------------------------------- #
if __name__ == "__main__":
    game = GameCV()  # default 640 × 360
    game.run()
