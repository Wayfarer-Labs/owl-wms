import cv2
import torch

import torch.nn.functional as F

KEYBINDS = ["W","A","S","D","LSHIFT","SPACE","R","F","E", "LMB", "RMB"]
import os
import numpy as np


def _put_text_with_box(img, text, x, y, fg, bg=(0,0,0), alpha=0.6,
                       font=cv2.FONT_HERSHEY_SIMPLEX, scale=0.6, thick=2, pad=4):
    (tw, th), bl = cv2.getTextSize(text, font, scale, thick)
    x0, y0, x1, y1 = x - pad, y - th - pad, x + tw + pad, y + bl + pad
    overlay = img.copy()
    cv2.rectangle(overlay, (x0, y0), (x1, y1), bg, -1)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
    cv2.putText(img, text, (x, y), font, scale, (0,0,0), thick + 2, cv2.LINE_AA)
    cv2.putText(img, text, (x, y), font, scale, fg, thick, cv2.LINE_AA)
    return (tw, th, bl)


def draw_frame(frame, mouse, button, labels=None, is_gt=None, prompts=None):
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

    # Draw compass circle and mouse position in top left
    circle_center = (50, 50)  # Center of compass
    circle_radius = 40
    cv2.circle(frame, circle_center, circle_radius, (255,255,255), 1)  # Draw compass circle

    if mouse is not None:
        # Convert mouse coordinates (-1 to 1) to compass coordinates
        mouse_x = mouse[0].item() * circle_radius + circle_center[0]
        mouse_y = mouse[1].item() * circle_radius + circle_center[1]

        # Draw arrow from center to mouse position
        cv2.arrowedLine(frame, circle_center, (int(mouse_x), int(mouse_y)), (0,255,0), 2)

    if button is not None:
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

    # Bottom prompt (wrapped, full-width box)
    if prompts:
        import textwrap
        font, scale, thick, gap = cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2, 4
        margin, pad = 10, 8
        lines = textwrap.wrap(str(prompts), width=60)
        if lines:
            sizes = [cv2.getTextSize(l, font, scale, thick)[0] for l in lines]
            total_h = sum(h for (_, h) in sizes) + gap * (len(lines) - 1)
            bottom_y = frame.shape[0] - margin
            if button is not None:
                bottom_y = y_pos - 10
            x0, y0 = 0, bottom_y - total_h - 2 * pad
            x1, y1 = frame.shape[1], bottom_y + pad
            overlay = frame.copy()
            cv2.rectangle(overlay, (x0, y0), (x1, y1), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
            y = y0 + pad
            for (l, (_, h)) in zip(lines, sizes):
                cv2.putText(frame, l, (margin, y + h), font, scale, (0, 0, 0), thick + 2, cv2.LINE_AA)
                cv2.putText(frame, l, (margin, y + h), font, scale, (255, 255, 255), thick, cv2.LINE_AA)
                y += h + gap

    # Top-right badge ("GT"/"AI") + labels inside ONE semi-transparent box
    x_margin, y_margin, gap, pad = 5, 5, 4, 6
    lines = []
    if is_gt is not None:
        lines.append(( "GT" if is_gt else "AI",
                       (0,165,255) if is_gt else (0,255,0),
                       0.7, 2))
    if labels:
        for k, v in labels.items():
            lines.append((f"{k}: {v}", (255,255,255), 0.6, 2))
    if lines:
        # Measure block
        sizes = [cv2.getTextSize(t, cv2.FONT_HERSHEY_SIMPLEX, sc, th)[0] for (t,_,sc,th) in lines]
        max_w = max(w for (w,h) in sizes)
        total_h = sum(h for (w,h) in sizes) + gap*(len(lines)-1)
        x_right = frame.shape[1] - x_margin
        x_text = x_right - max_w
        y_text = y_margin
        # Draw single background box
        x0, y0 = x_text - pad, y_text - pad
        x1, y1 = x_right + pad, y_text + total_h + pad
        overlay = frame.copy()
        cv2.rectangle(overlay, (x0, y0), (x1, y1), (0,0,0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
        # Draw texts with outline
        y = y_text
        for (text, color, sc, th), (tw, thh) in zip(lines, sizes):
            y += thh
            x = x_right - tw
            cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, sc, (0,0,0), th+2, cv2.LINE_AA)
            cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, sc, color, th, cv2.LINE_AA)
            y += gap

    # Convert back to RGB for display
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = np.transpose(frame, (2, 0, 1))  # HWC -> CHW
    return frame


def draw_frames(frames, mouse_inputs, button_inputs, labels=None, num_gt_frames=None, prompts=None):
    # frames is [b,n,c,h,w] tensor
    # mouse_inputs is [b,n,2]
    # button_inputs is [b,n,n_buttons]
    b, n = frames.shape[:2]
    out_frames = []
    labels_list = labels if isinstance(labels, list) else [labels] * b
    for i in range(b):
        batch_frames = []
        for j in range(n):
            frame = frames[i,j]
            mouse = mouse_inputs[i,j] if mouse_inputs is not None else None
            button = button_inputs[i,j] if button_inputs is not None else None
            is_gt = (j < num_gt_frames) if num_gt_frames is not None else None
            drawn = draw_frame(frame, mouse, button, labels=labels_list[i], is_gt=is_gt,
                               prompts=prompts[i] if prompts else None)
            batch_frames.append(drawn)
        out_frames.append(np.stack(batch_frames))
    return np.stack(out_frames)
