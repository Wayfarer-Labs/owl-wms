import cv2
import numpy as np
from typing import Sequence

# =========================
# Config & constants
# =========================
FONT   = cv2.FONT_HERSHEY_SIMPLEX
WHITE  = (255, 255, 255)
BLACK  = (0,   0,   0)
YELLOW = (0, 255, 255)     # BGR
BLUE   = (120, 0,   0)     # dark blue, BGR

# Caption (top-left banner)
CAPTION = (
    "Moved to top-left, stretched so its RIGHT edge matches previous centered box right edge. "
    "Fixed scale, pad=4, strict clip."
)

# ANSI-like left block (no numpad), widths in "units"
_KEY_ROWS = [
    [("`",1),("1",1),("2",1),("3",1),("4",1),("5",1),("6",1),("7",1),("8",1),("9",1),("0",1),("-",1),("=",1),("Backspace",2)],
    [("Tab",1.5),("Q",1),("W",1),("E",1),("R",1),("T",1),("Y",1),("U",1),("I",1),("O",1),("P",1),("[",1),("]",1),("\\",1.5)],
    [("Caps",1.75),("A",1),("S",1),("D",1),("F",1),("G",1),("H",1),("J",1),("K",1),("L",1),(";",1),("'",1),("Enter",2.25)],
    [("Shift",2.25),("Z",1),("X",1),("C",1),("V",1),("B",1),("N",1),("M",1),(",",1),(".",1),("/",1),("Shift",2.75)],
    [("Ctrl",1.5),("Win",1.25),("Alt",1.25),("Space",6),("Alt",1.25),("Menu",1.25),("Ctrl",1.5)],
]

# Short labels for compact keys
_SHORT = {
    "Backspace": "Bksp", "Caps": "Caps", "Shift": "Shift", "Ctrl": "Ctrl",
    "Space": "Space", "Menu": "Menu", "Win": "Win", "Enter": "Enter", "Tab": "Tab",
}

# =========================
# Utilities
# =========================
def _fit_scale(text: str, font, scale: float, thick: int, max_w: int, min_scale: float = 0.35) -> float:
    ((tw, _), _) = cv2.getTextSize(str(text), font, scale, thick)
    if tw <= max_w or tw == 0:
        return scale
    return max(min_scale, scale * (max_w / tw))

def _center_text_in_rect(text: str, rect, scale: float, thick: int):
    (tw, th), _ = cv2.getTextSize(text, FONT, scale, thick)
    x0, y0, x1, y1 = rect
    x = int(x0 + (x1 - x0 - tw) / 2)
    y = int(y0 + (y1 - y0 - th) / 2 + th)
    return x, y, tw, th

def _wrap_fixed(text: str, scale: float, thick: int, inner_w: int):
    words, lines, cur = str(text).split(), [], ""
    for w in words:
        cand = (cur + " " + w).strip()
        if cv2.getTextSize(cand, FONT, scale, thick)[0][0] <= inner_w or not cur:
            cur = cand
        else:
            lines.append(cur)
            if cv2.getTextSize(w, FONT, scale, thick)[0][0] > inner_w:
                ell_w = cv2.getTextSize("…", FONT, scale, thick)[0][0]
                clip = w
                while clip and cv2.getTextSize(clip, FONT, scale, thick)[0][0] + ell_w > inner_w:
                    clip = clip[:-1]
                lines.append((clip + "…") if clip else "…")
                cur = ""
            else:
                cur = w
    if cur:
        lines.append(cur)
    return lines

# =========================
# Key codes: DirectInput (DIK_) scan codes per asawicki.info
# =========================
_DIK = {
    # number row
    "1":0x02, "2":0x03, "3":0x04, "4":0x05, "5":0x06, "6":0x07, "7":0x08, "8":0x09, "9":0x0A, "0":0x0B,
    "-":0x0C, "=":0x0D, "Backspace":0x0E, "Bksp":0x0E,
    "Tab":0x0F,
    # top letters
    "Q":0x10,"W":0x11,"E":0x12,"R":0x13,"T":0x14,"Y":0x15,"U":0x16,"I":0x17,"O":0x18,"P":0x19,"[":0x1A,"]":0x1B,"Enter":0x1C,
    "Ctrl":{0x1D, 0x9D},            # LCTRL, RCTRL
    # home row
    "A":0x1E,"S":0x1F,"D":0x20,"F":0x21,"G":0x22,"H":0x23,"J":0x24,"K":0x25,"L":0x26,";":0x27,"'":0x28,"`":0x29,
    "Shift":{0x2A, 0x36},          # LSHIFT, RSHIFT
    "\\":0x2B,
    # bottom row
    "Z":0x2C,"X":0x2D,"C":0x2E,"V":0x2F,"B":0x30,"N":0x31,"M":0x32,",":0x33,".":0x34,"/":0x35,
    "Alt":{0x38, 0xB8},            # LALT, RALT
    "Space":0x39, "Caps":0x3A,
    # win/menu keys
    "Win":{0xDB, 0xDC}, "Menu":0xDD,  # LWIN,RWIN,APPS
    # arrows (extended)
    "<":0xCB, "^":0xC8, ">":0xCD, "v":0xD0,  # left, up, right, down
}

def _vk_labels_from_code(code: int) -> set[str]:
    """Map ONLY the VK codes from the provided document to our drawn labels."""
    L: set[str] = set()
    # editing / control
    if code == 8:   L |= {"Backspace", "Bksp"}
    elif code == 9: L.add("Tab")
    elif code == 12: pass  # Numpad 5 (NumLock off) – not drawn
    elif code == 13: L.add("Enter")
    elif code == 16: L.add("Shift")
    elif code == 17: L.add("Ctrl")
    elif code == 18: L.add("Alt")
    elif code == 19: pass  # Pause/Break – not drawn
    elif code == 20: L.add("Caps")
    elif code == 27: pass  # Esc – not drawn
    elif code == 32: L.add("Space")
    elif code in (33,34,35,36,44,45,46): pass  # not drawn on left block
    # arrows
    elif code == 37: L.add("<")
    elif code == 38: L.add("^")
    elif code == 39: L.add(">")
    elif code == 40: L.add("v")
    # digits 0–9
    elif 48 <= code <= 57: L.add(chr(code))
    # letters A–Z
    elif 65 <= code <= 90: L.add(chr(code))
    # Win/Menu
    elif code in (91, 92): L.add("Win")
    elif code == 93: L.add("Menu")
    # function keys, numpad cluster, locks – not drawn here
    elif code in range(96, 112): pass
    elif code in range(112, 124): pass
    elif code in (144, 145): pass
    # explicit left/right modifiers
    elif code in (160, 161): L.add("Shift")
    elif code in (162, 163): L.add("Ctrl")
    return L

def _label_selected(lbl: str, pressed_codes: set[int]) -> bool:
    """Select keys strictly per the VK list from the document."""
    key = {"Bksp": "Backspace"}.get(lbl, lbl)
    for vk in pressed_codes:
        if key in _vk_labels_from_code(int(vk)):
            return True
    return False

# =========================
# Overlays: badge + banner
# =========================
def _render_badge_top_right(img: np.ndarray, is_gt: bool | None, labels: dict | None,
                            gt_scale=0.36, lbl_scale=0.32, thick=1,
                            outline_extra=3, gap=2, pad=6, bg_alpha=0.6):
    """Single semi-transparent box; colored GT/AI head; black outline under text."""
    if is_gt is None and not labels:
        return 0, 0, 0, 0
    H, W = img.shape[:2]
    head = ("GT", (0,165,255), gt_scale, thick, outline_extra) if is_gt else ("AI", (0,255,0), gt_scale, thick, outline_extra)
    meta = [(f"{k}: {v}", WHITE, lbl_scale, thick, outline_extra) for k, v in (labels or {}).items()]
    lines = [head] + meta
    sizes = [cv2.getTextSize(t, FONT, s, thick)[0] for (t, _, s, _, _) in lines]
    max_w = max(w for (w, h) in sizes) if sizes else 0
    total_h = sum(h for (w, h) in sizes) + gap * (len(lines) - 1)

    x1 = W - 6
    x0 = x1 - (max_w + 2 * pad)
    y0 = 0
    y1 = y0 + (total_h + 2 * pad)

    overlay = img.copy()
    cv2.rectangle(overlay, (x0, y0), (x1, y1), BLACK, -1)
    cv2.addWeighted(overlay, bg_alpha, img, 1 - bg_alpha, 0, img)

    y = y0 + pad
    for (text, color, s, th, outline), (tw, thh) in zip(lines, sizes):
        y += thh
        x = x1 - pad - tw
        cv2.putText(img, text, (x, y), FONT, s, BLACK, th + outline, cv2.LINE_AA)
        cv2.putText(img, text, (x, y), FONT, s, color, th, cv2.LINE_AA)
        y += gap

    return (x0, y0, x1, y1)

def _render_top_left_banner(img: np.ndarray, text: str, right_edge_x: int, max_h: int,
                            scale: float = 0.34, thick: int = 1, gap: int = 2, pad: int = 4,
                            bg_alpha: float = 0.42):
    x0, y0, x1, y1 = 0, 0, right_edge_x, max_h
    overlay = img.copy()
    cv2.rectangle(overlay, (x0, y0), (x1, y1), BLACK, -1)
    cv2.addWeighted(overlay, bg_alpha, img, 1 - bg_alpha, 0, img)

    inner_w = (x1 - x0) - 2 * pad
    lines = _wrap_fixed(text, scale, thick, inner_w)
    y = y0 + pad
    bottom = y1 - pad
    for l in lines:
        (tw, th), _ = cv2.getTextSize(l, FONT, scale, thick)
        if y + th > bottom:
            break
        x = max(x0 + pad, int((x0 + x1)//2 - tw/2))
        cv2.putText(img, l, (x, y + th), FONT, scale, BLACK, thick + 3, cv2.LINE_AA)
        cv2.putText(img, l, (x, y + th), FONT, scale, WHITE, thick, cv2.LINE_AA)
        y += th + gap

# =========================
# Keyboard (left block + aligned inverted-T arrows)
# =========================
def _draw_keyboard(img: np.ndarray, selected_vks: set[int],
                   pad: int = 12, gap: int = 6, scale_factor: float = 0.7,
                   base_label_scale: float = 0.42, key_alpha: float = 0.3,
                   arrow_gutter_mult: int = 5):
    H, W = img.shape[:2]

    def sum_units(row): return sum(w for _, w in row)
    widest_units = max(sum_units(r) for r in _KEY_ROWS)
    approx_gaps = len(_KEY_ROWS[0]) - 1
    base_unit = max(10, int((int(0.68 * (W - 2 * pad)) - approx_gaps * gap) / widest_units))
    unit = max(8, int(base_unit * scale_factor))

    def row_px(row):  # pixels for a row
        return int(round(sum(w * unit for _, w in row) + (len(row) - 1) * gap))

    block_w = max(row_px(r) for r in _KEY_ROWS)
    main_left = pad
    main_right = main_left + block_w

    total_h = 5 * unit + 4 * gap
    y1 = H - pad
    y0 = y1 - total_h

    overlay = img.copy()

    def draw_row(stage: str, keys, top_y, left_x):
        x = left_x
        for label, w in keys:
            lbl = _SHORT.get(label, label) if stage == "fg" else label
            wpx = int(round(w * unit))
            rect = (x, top_y, x + wpx, top_y + unit)
            sel = _label_selected(lbl, selected_vks)
            if stage == "bg":
                if sel:
                    # Draw to BOTH img and overlay so post-blend remains fully white
                    cv2.rectangle(img,     (rect[0], rect[1]), (rect[2], rect[3]), WHITE, -1, cv2.LINE_AA)
                    cv2.rectangle(overlay, (rect[0], rect[1]), (rect[2], rect[3]), WHITE, -1, cv2.LINE_AA)
                else:
                    cv2.rectangle(overlay, (rect[0], rect[1]), (rect[2], rect[3]), BLACK, -1, cv2.LINE_AA)
            else:
                # Keep 1px border on the key edge; if selected, grow outline OUTWARDS only.
                cv2.rectangle(img, (rect[0], rect[1]), (rect[2], rect[3]), BLACK, 1, cv2.LINE_AA)
                if sel:
                    for d in range(1, 3):  # add 2px outward -> same visual thickness as old 3px
                        cv2.rectangle(img, (rect[0]-d, rect[1]-d), (rect[2]+d, rect[3]+d), BLACK, 1, cv2.LINE_AA)
                s = _fit_scale(lbl, FONT, base_label_scale, 1, (rect[2] - rect[0]) - 6)
                tx, ty, _, _ = _center_text_in_rect(lbl, rect, s, 1)
                cv2.putText(img, lbl, (tx, ty), FONT, s, BLACK if sel else WHITE, 1, cv2.LINE_AA)
            x = rect[2] + gap

    # backgrounds for main block
    y = y0
    for row in _KEY_ROWS:
        draw_row("bg", row, y, main_left)
        y += unit + gap

    # arrows (inverted T, ↑ centered over ↓)
    gutter = arrow_gutter_mult * gap
    arrows_left = main_right + gutter
    bottom_y = y0 + unit * 3 + gap * 3
    left_r  = (arrows_left,                    bottom_y, arrows_left + unit,                    bottom_y + unit)
    down_r  = (arrows_left + (unit + gap),     bottom_y, arrows_left + 2*unit + gap,            bottom_y + unit)
    right_r = (arrows_left + 2*(unit + gap),   bottom_y, arrows_left + 3*unit + 2*gap,          bottom_y + unit)
    up_y    = bottom_y - (unit + gap)
    up_r    = (down_r[0], up_y, down_r[2], up_y + unit)

    for rect, label in [(left_r,"<"), (down_r,"v"), (right_r,">"), (up_r,"^")]:
        if _label_selected(label, selected_vks):
            # Same trick for arrows: paint both surfaces (white on press)
            cv2.rectangle(img,     (rect[0], rect[1]), (rect[2], rect[3]), WHITE, -1, cv2.LINE_AA)
            cv2.rectangle(overlay, (rect[0], rect[1]), (rect[2], rect[3]), WHITE, -1, cv2.LINE_AA)
        else:
            cv2.rectangle(overlay, (rect[0], rect[1]), (rect[2], rect[3]), BLACK, -1, cv2.LINE_AA)

    # blend non-selected keys
    cv2.addWeighted(overlay, key_alpha, img, 1 - key_alpha, 0, img)

    # foregrounds (labels + borders)
    y = y0
    for row in _KEY_ROWS:
        draw_row("fg", row, y, main_left)
        y += unit + gap

    def draw_arrow_fg(rect, label):
        sel = _label_selected(label, selected_vks)
        # Same outward-only outline behavior for arrows.
        cv2.rectangle(img, (rect[0], rect[1]), (rect[2], rect[3]), BLACK, 1, cv2.LINE_AA)
        if sel:
            for d in range(1, 3):
                cv2.rectangle(img, (rect[0]-d, rect[1]-d), (rect[2]+d, rect[3]+d), BLACK, 1, cv2.LINE_AA)
        s = _fit_scale(label, FONT, 0.42, 1, (rect[2] - rect[0]) - 6)
        tx, ty, _, _ = _center_text_in_rect(label, rect, s, 1)
        color = BLACK if sel else WHITE
        cv2.putText(img, label, (tx, ty), FONT, s, color, 1, cv2.LINE_AA)

    for rect, label in [(left_r,"<"), (down_r,"v"), (right_r,">"), (up_r,"^")]:
        draw_arrow_fg(rect, label)

# =========================
# Public API
# =========================
def draw_frame(frame, mouse, button, labels=None, is_gt=None, prompts=None):
    """
    Render one frame.
      frame:  torch tensor [3,H,W] in [-1,1]
      mouse:  torch tensor [2] with coords in [-1,1] or None
      button: set[int] VK codes pressed for this frame, or None
      labels: dict for top-right badge (e.g. {"fps":60,...})
      is_gt:  bool|None -> 'GT'/'AI' head
      prompts:       Optional prompt(s) for the top-left banner.
                     Accepts:
                       - str (same for all frames),
                       - list[str] length B (per batch item),
                       - list[list[str]] shape BxN (per frame).
    """
    # Prepare image (CHW [-1,1] -> HWC BGR [0,255])
    frame = frame[:3].permute(1, 2, 0)
    frame = ((frame + 1) * 127.5).float().cpu().numpy().astype(np.uint8)
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    H, W = frame.shape[:2]

    # Compass
    circle_center, circle_radius = (50, 50), 40
    cv2.circle(frame, circle_center, circle_radius, WHITE, 1)
    if mouse is not None:
        mouse = mouse * 0.25  # TODO clean up / HACK: scale mouse x,y to 1/4
        mx = int(mouse[0].item() * circle_radius + circle_center[0])
        my = int(mouse[1].item() * circle_radius + circle_center[1])
        cv2.arrowedLine(frame, circle_center, (mx, my), (0, 255, 0), 2)

    # Top-right badge and top-left caption banner
    bx0, by0, bx1, by1 = _render_badge_top_right(frame, is_gt=is_gt, labels=labels)
    badge_h = max(24, by1 - by0)  # robust default
    fixed_w = 420
    right_edge_x = (W - fixed_w) // 2 + fixed_w
    if prompts:
        _render_top_left_banner(frame, str(prompts), right_edge_x=right_edge_x, max_h=badge_h)

    # Keyboard with VK selection
    selected_vks = set(button) if button is not None else set()
    _draw_keyboard(frame, selected_vks)

    # Return CHW RGB
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return np.transpose(frame, (2, 0, 1))

def draw_frames(frames, mouse_inputs, button_inputs, labels=None, num_gt_frames=None, prompts=None):
    """
    frames:        torch tensor [B,N,C,H,W]
    mouse_inputs:  torch tensor [B,N,2] or None
    button_inputs: list[list[set[int]]] (pressed VK codes per frame)
    labels:        dict or list[dict] per batch item (for badge)
    num_gt_frames: int|None (first N frames flagged GT)
    prompts: str|None -> top-left banner text; if None, banner is hidden
    """
    B, N = frames.shape[:2]
    labels_list = labels if isinstance(labels, list) else [labels] * B

    def pick_prompt(i: int, j: int):
        if prompts is None:
            return None
        if isinstance(prompts, str):
            return prompts
        if isinstance(prompts, Sequence):
            try:
                pj = prompts[i]
                if isinstance(pj, Sequence) and not isinstance(pj, str):
                    return pj[j]
                return pj
            except Exception:
                return None

    def to_vk_set(x):
        if x is None: return set()
        if isinstance(x, set): return set(int(v) for v in x)
        try: return set(int(v) for v in x)
        except Exception: return set()

    out = []
    for i in range(B):
        row = []
        for j in range(N):
            is_gt = (j < num_gt_frames) if num_gt_frames is not None else None
            btn = to_vk_set(button_inputs[i][j]) if button_inputs is not None else None
            mouse = mouse_inputs[i, j] if mouse_inputs is not None else None
            img = draw_frame(frames[i, j], mouse, btn, labels=labels_list[i], is_gt=is_gt, prompts=pick_prompt(i, j))
            row.append(img)
        out.append(np.stack(row))
    return np.stack(out)
