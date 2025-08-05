import cv2
import torch

import torch.nn.functional as F

KEYBINDS = ["W","A","S","D","LSHIFT","SPACE","R","F","E", "LMB", "RMB"]
import os
import numpy as np

def draw_frame(frame, mouse, button):
    # frame is a torch tensor of shape [3,h,w]
    # mouse is [2,] tensor
    # button is list[bool]
    frame = frame[:3]  # Only ever take 3 channels

    #frame = F.interpolate(frame.unsqueeze(0),(512,512))
    frame = frame.squeeze(0)
    frame = frame.permute(1,2,0)
    frame = (frame + 1)*127.5
    frame = frame.float().cpu().numpy()
    frame = frame.astype(np.uint8)
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    """
    # Draw compass circle and mouse position in top left
    circle_center = (50, 50)  # Center of compass
    circle_radius = 40
    cv2.circle(frame, circle_center, circle_radius, (255,255,255), 1)  # Draw compass circle

    # Convert mouse coordinates (-1 to 1) to compass coordinates
    mouse_x = mouse[0].item() * circle_radius + circle_center[0]
    mouse_y = mouse[1].item() * circle_radius + circle_center[1]

    # Draw arrow from center to mouse position
    cv2.arrowedLine(frame, circle_center, (int(mouse_x), int(mouse_y)), (0,255,0), 2)

    # Draw button boxes along bottom
    box_width = 40
    box_height = 40
    margin = 5
    y_pos = frame.shape[0] - box_height - 10  # 10px from bottom

    # Calculate starting x to center the boxes
    total_width = (box_width + margin) * len(KEYBINDS) - margin
    start_x = (frame.shape[1] - total_width) // 2

    for i in range(len(KEYBINDS)):
        x = start_x + i * (box_width + margin)

        # Draw box
        color = (0,255,0) if button[i] else (0,0,255)  # Green if pressed, red if not
        cv2.rectangle(frame, (x, y_pos), (x + box_width, y_pos + box_height), color, -1)

        # Draw label
        label = KEYBINDS[i]
        text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
        text_x = x + (box_width - text_size[0]) // 2
        text_y = y_pos - 5  # 5px above box
        cv2.putText(frame, label, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
    """

    # Convert back to RGB for display

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = np.transpose(frame, (2, 0, 1))  # HWC -> CHW
    return frame

def draw_frames(frames, mouse_inputs, button_inputs):
    # frames is [b,n,c,h,w] tensor
    # mouse_inputs is [b,n,2]
    # button_inputs is [b,n,n_buttons]
    b, n = frames.shape[:2]
    out_frames = []
    for i in range(b):
        batch_frames = []
        for j in range(n):
            frame = frames[i,j]
            mouse = mouse_inputs[i,j]
            button = button_inputs[i,j]
            drawn = draw_frame(frame, mouse, button)
            batch_frames.append(drawn)
        out_frames.append(np.stack(batch_frames))
    return np.stack(out_frames)
