import cv2

import torch.nn.functional as F
import numpy as np


KEYBINDS = ["W","A","S","D","LSHIFT","SPACE","R","F","E", "LMB", "RMB"]


def draw_frame(frame, vec_3d):
    """
    frame         : torch.Tensor [3,H,W], values in [-1…1]
    vec_3d        : list of (label: str, vec: sequence of 3 floats)
    returns       : np.uint8 array [3,H,W] in RGB
    """
    # frame is a torch tensor of shape [3,h,w]
    # mouse is [2,] tensor
    # button is list[bool]
    frame = frame[:3].squeeze(0)
    frame = (frame.permute(1, 2, 0) + 1) * 127.5  # [H,W,3] float
    frame = frame.clamp(0, 255).byte().cpu().numpy()
    img = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    H, W = img.shape[:2]
    size = W // 10
    font = cv2.FONT_HERSHEY_SIMPLEX
    fs = size / 200.0

    for i, (label, vec) in enumerate(vecs):
        icon = render_vec_3d(vec, size=size)
        x0, y0 = i * size, H - size
        img[y0:, x0:x0+size] = icon
        cv2.putText(img, label, (x0+2, y0-2), font, fs, (0,0,0), 1, cv2.LINE_AA)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img.transpose(2, 0, 1)


def nothing(frame):
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


def render_vec_3d(vec, size=500, azim=100, elev=20):
    alpha, beta = np.deg2rad([azim, elev])
    center = size // 2
    scale = size / 3.0
    dash = 10
    gap = 5
    step = dash + gap

    def proj(x, y, z):
        u = np.cos(alpha) * x + np.sin(alpha) * y
        v = (
            -np.sin(alpha) * np.sin(beta) * x +
            np.cos(alpha) * np.sin(beta) * y +
            np.cos(beta) * z
        )
        return int(center + scale * u), int(center - scale * v)

    def cmap(v):
        t = (v + 1) * 0.5
        if t <= 0.5:
            g = int(255 * (t / 0.5))
            return (255, g, g)
        g = int(255 * ((1 - t) / 0.5))
        return (g, g, 255)

    x, y, z = vec
    pts = {
        name: proj(*coords)
        for name, coords in {
            "O": (0, 0, 0),
            "X": (x, 0, 0),
            "Y": (0, y, 0),
            "Z": (0, 0, z),
            "XY": (x, y, 0),
            "XZ": (x, 0, z),
            "YZ": (0, y, z),
            "XYZ": (x, y, z),
        }.items()
    }

    img = np.full((size, size, 3), 255, dtype=np.uint8)
    cv2.circle(img, pts["O"], 6, (0, 0, 0), -1)
    cv2.arrowedLine(
        img,
        pts["O"],
        pts["XYZ"],
        (0, 0, 0),
        2,
        tipLength=0.1,
    )

    cols = (cmap(x), cmap(y), cmap(z))
    edges = [
        ("O", "X", cols[0]),
        ("O", "Y", cols[1]),
        ("O", "Z", cols[2]),
        ("X", "XZ", cols[2]),
        ("Y", "YZ", cols[2]),
        ("XY", "XYZ", cols[2]),
        ("X", "XY", cols[1]),
        ("Y", "XY", cols[0]),
    ]

    for start, end, color in edges:
        x1, y1 = pts[start]
        x2, y2 = pts[end]
        dist = int(np.hypot(x2 - x1, y2 - y1))
        if dist == 0:
            continue
        vx = (x2 - x1) / dist
        vy = (y2 - y1) / dist
        for i in range(0, dist, step):
            j = min(dist, i + dash)
            sx = int(x1 + vx * i)
            sy = int(y1 + vy * i)
            ex = int(x1 + vx * j)
            ey = int(y1 + vy * j)
            cv2.line(img, (sx, sy), (ex, ey), color, 1)

    return img


def draw_frames(frames, mouse_inputs, button_inputs):
    # frames is [b,n,c,h,w] tensor
    # mouse_inputs is [b,n,2]
    # button_inputs is [b,n,n_buttons]
    b, n = frames.shape[:2]
    out_frames = []
    for i in range(b):
        batch_frames = []
        for j in range(n):
            frame = frames[i, j]
            mouse = mouse_inputs[i, j]
            button = button_inputs[i, j]

            ###
            vec_3d = [("HIP", mouse[[0,1,2]]), ("LSH", mouse[[6, 7, 8]])]
            ###

            drawn = draw_frame(frame, vec_3d)

            batch_frames.append(drawn)
        out_frames.append(np.stack(batch_frames))
    return np.stack(out_frames)
