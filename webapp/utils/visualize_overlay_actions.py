import imageio as imio
import cv2
import numpy as np
import math
import torch
from typing import Optional
from contextlib import contextmanager


# Global configuration
KEYBINDS = ["W", "A", "S", "D", "LSHIFT", "SPACE", "R", "F", "E", "LMB", "RMB"]
MINIMUM_FRAME_SIZE = 150
# Colors (BGR format for OpenCV)
COLOR_PRESSED = (50, 200, 50)      # Green
COLOR_UNPRESSED = (100, 100, 100)  # Gray
COLOR_TEXT = (255, 255, 255)       # White
COLOR_BACKGROUND = (30, 30, 30)    # Dark gray
COLOR_MOUSE_ARROW = COLOR_PRESSED  # Green
COLOR_UNCERTAINTY = (100, 255, 255) # Yellow-ish
COLOR_LMB_SECTOR = COLOR_PRESSED  # Green
COLOR_RMB_SECTOR = COLOR_PRESSED  # Green

# Key dimensions
KEY_SIZE = 20
KEY_MARGIN = 5
SHIFT_WIDTH = int(KEY_SIZE * 2 + KEY_MARGIN)  # Two keys worth of width
SPACE_WIDTH = int(KEY_SIZE * 5)

# Mouse compass dimensions
COMPASS_RADIUS = 24  # 80% of 80
COMPASS_START_X_PERCENT = 0.80

# Mouse button arc dimensions
MOUSE_BUTTON_OFFSET = 6  # Pixels outside the main circle
MOUSE_BUTTON_THICKNESS = 6  # Increased thickness

# Arrow scaling parameters
ARROW_SCALE_FACTOR  = 1  # Base scaling factor for arrow length
ARROW_MIN_LENGTH    = 55      # Minimum arrow length in pixels
ARROW_MAX_SCALE     = 0.75     # Maximum scale relative to compass radius

START_X_PERCENT = 0.18
START_Y_PERCENT = 0.85


@contextmanager
def _rescale_icons(ratio: float):
    """
    Rescale all icon sizes to the ratio of the original video to the 512x512 video
    """
    global KEY_SIZE, KEY_MARGIN, SHIFT_WIDTH, SPACE_WIDTH, COMPASS_RADIUS, MOUSE_BUTTON_OFFSET, MOUSE_BUTTON_THICKNESS
    global ARROW_SCALE_FACTOR, ARROW_MIN_LENGTH, ARROW_MAX_SCALE
    global START_X_PERCENT, START_Y_PERCENT
    
    try:
        old_values = {
            "KEY_SIZE": KEY_SIZE,
            "KEY_MARGIN": KEY_MARGIN,
            "SHIFT_WIDTH": SHIFT_WIDTH,
            "SPACE_WIDTH": SPACE_WIDTH,
            "COMPASS_RADIUS": COMPASS_RADIUS,
            "MOUSE_BUTTON_OFFSET": MOUSE_BUTTON_OFFSET,
            "MOUSE_BUTTON_THICKNESS": MOUSE_BUTTON_THICKNESS,
            "ARROW_SCALE_FACTOR": ARROW_SCALE_FACTOR,
        }
        
        KEY_SIZE *= ratio ; KEY_SIZE = int(KEY_SIZE)
        KEY_MARGIN *= ratio ; KEY_MARGIN = int(KEY_MARGIN)
        SHIFT_WIDTH *= ratio ; SHIFT_WIDTH = int(SHIFT_WIDTH)
        SPACE_WIDTH *= ratio ; SPACE_WIDTH = int(SPACE_WIDTH)
        COMPASS_RADIUS *= ratio ; COMPASS_RADIUS = int(COMPASS_RADIUS)
        MOUSE_BUTTON_OFFSET *= ratio ; MOUSE_BUTTON_OFFSET = int(MOUSE_BUTTON_OFFSET)
        MOUSE_BUTTON_THICKNESS *= ratio ; MOUSE_BUTTON_THICKNESS = int(MOUSE_BUTTON_THICKNESS)
        ARROW_SCALE_FACTOR *= ratio ; ARROW_SCALE_FACTOR = float(ARROW_SCALE_FACTOR)
        # Note: ARROW_MIN_LENGTH and ARROW_MAX_SCALE are now calculated relative to COMPASS_RADIUS
        # so they don't need separate scaling
        yield
    finally:
        for key, value in old_values.items():
            globals()[key] = value


def _get_adaptive_positioning(frame_width: int, frame_height: int) -> tuple[float, float, float]:
    """
    Calculate adaptive positioning percentages based on frame dimensions.
    Returns (keyboard_start_y_percent, mouse_start_y_percent, start_x_percent)
    """
    aspect_ratio = frame_width / frame_height
    
    # Calculate how much vertical space the keyboard needs
    keyboard_height_needed = KEY_SIZE * 3 + KEY_MARGIN * 2 + 20  # 3 rows + margins + buffer
    
    # Adaptive Y positioning - ensure keyboard fits
    if frame_height <= keyboard_height_needed + 40:  # Very short frame
        keyboard_start_y_percent = 0.1  # Start near top
        mouse_start_y_percent = 0.6    # Place mouse in middle-bottom
    elif frame_height < 300:  # Short frame
        keyboard_start_y_percent = 0.4
        mouse_start_y_percent = 0.75
    else:  # Normal/tall frame
        keyboard_start_y_percent = 0.75
        mouse_start_y_percent = 0.85
    
    # Adaptive X positioning based on aspect ratio
    if aspect_ratio > 1.5:  # Wide frame
        start_x_percent = 0.08
    elif aspect_ratio < 0.75:  # Tall frame
        start_x_percent = 0.15
    else:  # Near square frame
        start_x_percent = 0.12
    
    return keyboard_start_y_percent, mouse_start_y_percent, start_x_percent



def _draw_buttons(
    frame: np.ndarray,
    button_sequence: list[bool],
) -> None:
    """
    Draw keyboard buttons on the frame with adaptive positioning.
    """
    frame_height, frame_width = frame.shape[:2]
    
    # Get adaptive positioning
    keyboard_start_y_percent, _, start_x_percent = _get_adaptive_positioning(frame_width, frame_height)
    
    # Starting position for keyboard layout
    start_x = int(frame_width * start_x_percent)
    start_y = int(frame_height * keyboard_start_y_percent)
    
    # Rest of the function remains the same...
    key_positions = {
        # Top row: W E R (W above S)
        "W": (start_x + (KEY_SIZE + KEY_MARGIN) * 1, start_y),
        "E": (start_x + (KEY_SIZE + KEY_MARGIN) * 2, start_y),
        "R": (start_x + (KEY_SIZE + KEY_MARGIN) * 3, start_y),
        
        # Middle row: A S D F
        "A": (start_x + (KEY_SIZE + KEY_MARGIN) * 0, start_y + KEY_SIZE + KEY_MARGIN),
        "S": (start_x + (KEY_SIZE + KEY_MARGIN) * 1, start_y + KEY_SIZE + KEY_MARGIN),
        "D": (start_x + (KEY_SIZE + KEY_MARGIN) * 2, start_y + KEY_SIZE + KEY_MARGIN),
        "F": (start_x + (KEY_SIZE + KEY_MARGIN) * 3, start_y + KEY_SIZE + KEY_MARGIN),
        
        # Bottom row: LSHIFT SPACE
        "LSHIFT": (start_x - (KEY_SIZE + KEY_MARGIN), start_y + (KEY_SIZE + KEY_MARGIN) * 2),
        "SPACE": (start_x + SHIFT_WIDTH + KEY_MARGIN, start_y + (KEY_SIZE + KEY_MARGIN) * 2),
    }
    
    # Draw each key (rest remains the same)
    for i, key in enumerate(KEYBINDS[:-2]):  # Exclude LMB and RMB
        if key in key_positions:
            x, y = key_positions[key]
            
            # Ensure keys stay within frame bounds
            if x < 0 or y < 0 or x + KEY_SIZE > frame_width or y + KEY_SIZE > frame_height:
                continue
            
            # Determine key dimensions
            if key == "LSHIFT":
                width = SHIFT_WIDTH
                height = KEY_SIZE
            elif key == "SPACE":
                width = SPACE_WIDTH
                height = KEY_SIZE
            else:
                width = KEY_SIZE
                height = KEY_SIZE
            
            # Final bounds check with actual key dimensions
            if x + width > frame_width or y + height > frame_height:
                continue
            
            # Determine color based on pressed state
            color = COLOR_PRESSED if button_sequence[i] else COLOR_UNPRESSED
            
            # Draw key background
            cv2.rectangle(frame, (x, y), (x + width, y + height), color, -1)
            
            # Draw key border
            cv2.rectangle(frame, (x, y), (x + width, y + height), COLOR_TEXT, 1)
            
            # Draw key label
            text_size = cv2.getTextSize(key, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
            text_x = x + (width - text_size[0]) // 2
            text_y = y + (height + text_size[1]) // 2
            cv2.putText(frame, key, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.5, COLOR_TEXT, 1, cv2.LINE_AA)

def _draw_mouse(
    frame: np.ndarray,
    LMB_on: bool,
    RMB_on: bool,
    mouse_delta: tuple[float, float],
    center: tuple[int, int],
) -> None:
    """
    Draw mouse compass with direction arrow and properly positioned labels.
    Arrow length is proportional to mouse movement magnitude.
    
    Args:
        frame: numpy array representing the image frame
        LMB_on: bool indicating if left mouse button is pressed
        RMB_on: bool indicating if right mouse button is pressed
        mouse_delta: tuple of floats (x, y) representing mouse direction
        center: tuple (x, y) for compass center position
    """
    # Draw compass circle
    cv2.circle(frame, center, COMPASS_RADIUS, COLOR_TEXT, 2)
    
    # Calculate outer radius for mouse buttons (slightly outside the main compass)
    button_radius = COMPASS_RADIUS + MOUSE_BUTTON_OFFSET
    
    # Draw LMB arc (top-left 45 degrees) - outside the main circle and thicker
    color_lmb = COLOR_LMB_SECTOR if LMB_on else (60, 60, 60)  # Gray when off
    cv2.ellipse(frame, center, (button_radius, button_radius), 
               0, 225, 270, color_lmb, MOUSE_BUTTON_THICKNESS)
    
    # Draw RMB arc (top-right 45 degrees) - outside the main circle and thicker
    color_rmb = COLOR_RMB_SECTOR if RMB_on else (60, 60, 60)  # Gray when off
    cv2.ellipse(frame, center, (button_radius, button_radius), 
               0, 270, 315, color_rmb, MOUSE_BUTTON_THICKNESS)
    
    # Fixed label positioning - spread them out more and position them better
    # LMB label - position it to the left of the compass
    text_loc_lmb = (center[0] - COMPASS_RADIUS - 15, center[1] - COMPASS_RADIUS + 5)
    cv2.putText(frame, 'LMB', text_loc_lmb, cv2.FONT_HERSHEY_SIMPLEX, 0.4, COLOR_TEXT, 1, cv2.LINE_AA)
    
    # RMB label - position it to the right of the compass  
    text_loc_rmb = (center[0] + COMPASS_RADIUS - 10, center[1] - COMPASS_RADIUS + 5)
    cv2.putText(frame, 'RMB', text_loc_rmb, cv2.FONT_HERSHEY_SIMPLEX, 0.4, COLOR_TEXT, 1, cv2.LINE_AA)
    
    # Calculate mouse direction
    mouse_x, mouse_y = mouse_delta
    magnitude = math.sqrt(mouse_x**2 + mouse_y**2)
    
    if magnitude > 0:
        # Normalize to unit vector
        unit_x = mouse_x / magnitude
        unit_y = mouse_y / magnitude
        
        # Fixed arrow scaling - ensure arrow stays within compass bounds
        # Calculate minimum and maximum arrow lengths relative to compass size
        min_arrow_length = COMPASS_RADIUS * 0.3  # 30% of compass radius
        max_arrow_length = COMPASS_RADIUS * 0.8  # 80% of compass radius (stays within circle)
        
        # Scale by magnitude but clamp to reasonable bounds
        arrow_scale = max(
            min_arrow_length,  # Minimum visible arrow
            min(
                COMPASS_RADIUS * magnitude * ARROW_SCALE_FACTOR,  # Proportional to magnitude
                max_arrow_length  # Cap at 80% of compass radius
            )
        )
        
        end_x = int(center[0] + unit_x * arrow_scale)
        end_y = int(center[1] - unit_y * arrow_scale)  # Negative because y-axis is inverted
        
        # Draw direction arrow with proportional length
        cv2.arrowedLine(frame, center, (end_x, end_y), COLOR_MOUSE_ARROW, 2, tipLength=0.4)
        
        # Optional: Display magnitude as text for debugging
        # magnitude_text = f"Mag: {magnitude:.2f}"
        # cv2.putText(frame, magnitude_text, (center[0] - 40, center[1] + button_radius + 40),
        #             cv2.FONT_HERSHEY_SIMPLEX, 0.4, COLOR_TEXT, 1, cv2.LINE_AA)
    
    # Draw center dot
    cv2.circle(frame, center, 2, COLOR_TEXT, -1)
    
    # Add "Mouse" label below the compass
    cv2.putText(frame, "Mouse", (center[0] - 20, center[1] + button_radius + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_TEXT, 1, cv2.LINE_AA)


def _draw_frame(
    frame: np.ndarray,
    buttons: list[bool] | torch.Tensor,
    mouse_delta: tuple[float, float],
) -> np.ndarray:
    """
    Overlay keyboard & mouse on a single frame with adaptive positioning.
    """
    buttons, lmb, rmb = buttons[:-2], buttons[-2], buttons[-1]
    _draw_buttons(frame, buttons)

    frame_height, frame_width = frame.shape[:2]
    
    # Get adaptive positioning
    _, mouse_start_y_percent, _ = _get_adaptive_positioning(frame_width, frame_height)
    
    # Calculate compass position
    compass_x_percent = COMPASS_START_X_PERCENT
    compass_x = int(frame_width * compass_x_percent) - COMPASS_RADIUS - MOUSE_BUTTON_OFFSET
    compass_y = int(frame_height * mouse_start_y_percent)

    # Ensure compass fits within frame boundaries
    min_x = COMPASS_RADIUS + MOUSE_BUTTON_OFFSET + 20
    max_x = frame_width - COMPASS_RADIUS - MOUSE_BUTTON_OFFSET - 20
    compass_x = max(min_x, min(compass_x, max_x))
    
    min_y = COMPASS_RADIUS + MOUSE_BUTTON_OFFSET + 30
    max_y = frame_height - COMPASS_RADIUS - MOUSE_BUTTON_OFFSET - 50
    compass_y = max(min_y, min(compass_y, max_y))

    _draw_mouse(frame,
                LMB_on=lmb, RMB_on=rmb,
                mouse_delta=mouse_delta,
                center=(compass_x, compass_y))
    return frame

def _draw_video(
    video: torch.Tensor,
    buttons: torch.Tensor,
    mouse_delta: torch.Tensor,
    save_path: Optional[str] = None,
    fps: int = 30,
    arrow_scale_factor: Optional[float] = None,
    arrow_max_scale: Optional[float] = None,
) -> np.ndarray:
    """
    Draw video with input device monitoring overlays.
    
    Args:
        video: torch tensor of video frames, [num_frames, 256, 256, 3]
        buttons: torch tensor of button states
        mouse_delta: torch tensor of mouse deltas between frames
        save_path: optional path to save video
        fps: frames per second for output video
        arrow_scale_factor: optional override for arrow scaling
        arrow_max_scale: optional override for maximum arrow scale
    """
    # Update global arrow parameters if provided
    global ARROW_SCALE_FACTOR, ARROW_MAX_SCALE
    if arrow_scale_factor is not None:
        ARROW_SCALE_FACTOR = arrow_scale_factor
    if arrow_max_scale is not None:
        ARROW_MAX_SCALE = arrow_max_scale

    video = video.float().cpu().detach().numpy()
    
    # Get original dimensions
    original_height, original_width = video.shape[1], video.shape[2]
    
    # Calculate scaling to meet minimum size while preserving aspect ratio
    if original_height < MINIMUM_FRAME_SIZE or original_width < MINIMUM_FRAME_SIZE:
        # Calculate scale factor to make smallest dimension equal to MINIMUM_FRAME_SIZE
        scale_factor = MINIMUM_FRAME_SIZE / min(original_height, original_width)
        new_height = int(original_height * scale_factor)
        new_width = int(original_width * scale_factor)
        
        frames = [cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_CUBIC) for frame in video]
        # UI scaling ratio - keep icons at base size since we're scaling up small frames
        ratio = 0.5
    else:
        frames = video
        # UI scaling ratio - scale icons proportional to how much larger the frame is than minimum
        # This ensures icons don't become too large on big frames
        ratio = min(original_height, original_width) / MINIMUM_FRAME_SIZE

    with _rescale_icons(ratio):
        frames = [
            _draw_frame(frame, buttons[i], mouse_delta[i])
            for i, frame in enumerate(frames)
        ]
    if save_path is not None:
        imio.mimsave(save_path, [f.astype(np.uint8) for f in frames], fps=fps)

    return np.stack(frames, axis=0) # [num_frames, 256, 256, 3]


# Example usage
if __name__ == "__main__":
    # Example button states (9 buttons, excluding LMB and RMB)
    button_states = [True, False, True, False, True, False, False, True, False]
    
    # Example mouse state
    LMB_pressed = True
    RMB_pressed = False
    mouse_direction = (0.7, 0.7)  # Diagonal up-right
    mouse_uncertainty = (0.2, 0.1)

    vidpath, mousepath, buttonpath = '.pt'

    video = torch.load(vidpath, map_location='cpu', mmap=True)
    mouse = torch.load(mousepath, map_location='cpu', mmap=True)
    buttons = torch.load(buttonpath, map_location='cpu', mmap=True) # [1, window_length, n_buttons]

    min_len = min(len(video), len(mouse), len(buttons))
    video = video[:min_len]
    mouse = mouse[:min_len]
    buttons = buttons[:min_len]

    video = video.float()
    buttons = buttons
    mouse = mouse
    
    # You can adjust arrow scaling parameters here
    _draw_video(video, buttons, mouse, 
                save_path="groundtruth.mp4",
                arrow_scale_factor=0.3,  # Adjust this to control arrow sensitivity
                arrow_max_scale=0.9)     # Maximum arrow length relative to compass
