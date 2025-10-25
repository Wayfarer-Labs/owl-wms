from rotary_embedding_torch import RotaryEmbedding
import torch
from torch import nn

import einops as eo
from einops._torch_specific import allow_ops_in_compiled_graph  # requires einops>=0.6.1
allow_ops_in_compiled_graph()


def get_rope_cls(cls_name):
    cls_name = cls_name.lower()
    if cls_name == "ortho":
        return OrthoRoPE
    elif cls_name == "motion":
        return MotionRoPE
    elif cls_name == "vid":
        return VidRoPE
    else:
        raise ValueError(f"Invalid RoPE class: {cls_name}")


def get_rope(config):
    cls = get_rope_cls(getattr(config, "rope_impl", "ortho"))
    return cls(config)


class RoPE(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        assert not getattr(self.config, "has_audio", False)

        freqs = self.get_freqs(config)
        self.cos = nn.Buffer(freqs.cos().contiguous(), persistent=False)
        self.sin = nn.Buffer(freqs.sin().contiguous(), persistent=False)

    def get_angles(self, pos_ids):
        t, y, x = pos_ids["t_pos"], pos_ids["y_pos"], pos_ids["x_pos"]  # [B,T]
        H, W = self.config.height, self.config.width
        torch._assert(y.max() < H and x.max() < W, "pos_ids out of bounds")
        flat = t * (H * W) + y * W + x                         # [B,T]
        idx = flat.reshape(-1).to(torch.long)
        cos = self.cos.index_select(0, idx).view(*flat.shape, -1)
        sin = self.sin.index_select(0, idx).view(*flat.shape, -1)
        return cos[:, None], sin[:, None]  # add head dim for broadcast

    @torch.autocast("cuda", enabled=False)
    def forward(self, x, pos_ids):
        assert self.cos.dtype == torch.float32
        cos, sin = self.get_angles(pos_ids)
        x0, x1 = x.float().unfold(-1, 2, 2).unbind(-1)
        y0 = x0 * cos - x1 * sin
        y1 = x1 * cos + x0 * sin
        return torch.cat((y0, y1), dim=-1).type_as(x)

    def get_freqs(self, config):
        raise NotImplementedError


class SeqRoPE(RoPE):
    """1D RoPE, no spatial rotation"""
    def get_angles(self, pos_ids):
        t = pos_ids["t_pos"]  # [B, T]
        T = self.config.n_frames
        torch._assert(t.max() < T, "t_pos out of bounds")
        idx = t.reshape(-1).to(torch.long)
        cos = self.cos.index_select(0, idx).view(*t.shape, -1)
        sin = self.sin.index_select(0, idx).view(*t.shape, -1)
        return cos[:, None], sin[:, None]

    def get_freqs(self, config):
        head_dim = config.d_model // config.n_heads
        torch._assert(head_dim % 2 == 0, "need even head_dim")
        freqs = RotaryEmbedding(dim=head_dim, freqs_for="lang", cache_if_possible=False)\
            .forward(torch.arange(config.n_frames))
        return freqs[..., ::2]


class OrthoRoPE(RoPE):
    """
    RoPE for rotation across orthogonal axes: time, height, and width
    Time: Geometric Spectrum -- rotates 1/2 of head dim
    Height / Width: Linear Spectrum -- rotates 1/4th of head dim each (1/2 combined)
    (Note: No subsampling applied in this implementation)
    """
    def get_freqs(self, config):
        H, W, T = config.height, config.width, config.n_frames
        head_dim = config.d_model // config.n_heads
        max_freq = getattr(config, 'rope_hw_max_freq', 32)  # should never be smaller than H

        freq_t = RotaryEmbedding(dim=head_dim // 4, freqs_for='lang')\
            .forward(torch.arange(T))
        freq_xy = RotaryEmbedding(dim=head_dim // 8, freqs_for='pixel', max_freq=max_freq)\
            .get_axial_freqs(H, W)
        return torch.cat([
            eo.repeat(freq_t, 't d -> (t h w) d', h=H, w=W),
            eo.repeat(freq_xy, 'h w d -> (t h w) d', t=T)
        ], dim=-1)


class VidRoPE(RoPE):
    """
    Video-only VideoRoPE (DL + ATS + LTA), precomputed angle table:
      • x,y occupy the global HIGH-frequency pairs (lower dims), interleaved by pair
      • t occupies the LOW-frequency tail (higher dims)
      • diagonal layout with ATS: t = T_s + δ * τ ; x = t + (w - W/2), y = t + (h - H/2)
    Pass per-sample T_s as `config.rope_ts_offset` before constructing, or rebuild per batch.
    """
    def get_freqs(self, config):
        H, W, T = int(config.height), int(config.width), int(config.n_frames)
        hd = int(config.d_model) // int(config.n_heads)
        torch._assert(hd % 2 == 0, "head_dim must be even")
        P = hd // 2                                 # rotary pairs

        # default split: x=3/8, y=3/8, t=1/4 of head_dim (by pairs)
        px = py = (3 * P) // 8
        pt = P - px - py
        torch._assert(px == py and px > 0 and pt > 0, "head_dim too small for x=y=3/8, t=1/4")

        theta = float(getattr(config, 'rope_theta', 10000.0))
        delta = float(getattr(config, 'rope_ats_stride', 2.0))   # δ
        T_s = float(getattr(config, 'rope_ts_offset', 0.0))    # per-sample text offset

        # ---- single global frequency ladder (size P), high -> low ----
        full = 1.0 / (theta ** (torch.arange(P, dtype=torch.float32) / P))

        # allocate topmost 2*px pairs to spatial, interleaved between x and y
        base_xy = full[: 2 * px]          # highest pairs for spatial
        base_x = base_xy[0::2][:px]      # even pairs -> x
        base_y = base_xy[1::2][:py]      # odd  pairs -> y

        # lowest tail to time (LTA)
        base_t = full[-pt:]

        # rotary angle generators for each axis
        re_x = RotaryEmbedding(dim=2 * px, custom_freqs=base_x, cache_if_possible=False)
        re_y = RotaryEmbedding(dim=2 * py, custom_freqs=base_y, cache_if_possible=False)
        re_t = RotaryEmbedding(dim=2 * pt, custom_freqs=base_t, cache_if_possible=False)

        # diagonal layout with ATS
        tpos = T_s + torch.arange(T, dtype=torch.float32) * delta     # [T]
        x_off = torch.arange(W, dtype=torch.float32) - ((W - 1) / 2.0)  # [W]
        y_off = torch.arange(H, dtype=torch.float32) - ((H - 1) / 2.0)  # [H]

        # rotary_embedding_torch returns 2 entries per pair; keep every other to get pair angles
        fx = re_x.forward(tpos[:, None, None] + x_off[None, None, :])[..., ::2]  # [T,1,W,px]
        fy = re_y.forward(tpos[:, None, None] + y_off[None, :, None])[..., ::2]  # [T,H,1,py]
        ft = re_t.forward(tpos)[..., ::2][:, None, None, :].expand(T, H, W, pt)  # [T,H,W,pt]

        # pairwise interleave x & y at the pair level
        fx = fx.expand(T, H, W, px)
        fy = fy.expand(T, H, W, py)
        fxy = torch.empty(T, H, W, px + py, dtype=fx.dtype, device=fx.device)
        fxy[..., 0::2] = fx
        fxy[..., 1::2] = fy

        freqs = torch.cat([fxy, ft], dim=-1)                            # [T,H,W,P]
        return freqs.reshape(T * H * W, P)


class MotionRoPE(RoPE):
    """
    https://arxiv.org/pdf/2502.05173
    RoPE implementing a diagonal layout where spatial coordinates are a linear function of time.
    This constant-velocity prior serves as a baseline for learning complex, non-linear motion.
    """
    def get_freqs(self, config):
        H = getattr(config, 'height', getattr(config, 'sample_size', None))
        W = getattr(config, 'width', getattr(config, 'sample_size', None))
        F = config.n_frames
        d_head = config.d_model // config.n_heads

        dims = {
            't': getattr(config, 'rope_dim_t', d_head * 2 // 8),
            'x': getattr(config, 'rope_dim_x', d_head * 3 // 8),
            'y': getattr(config, 'rope_dim_y', d_head * 3 // 8)
        }
        theta = getattr(config, 'rope_base', 10000.0)

        # TODO: paper is 3 FPS, uses delta=2.0, we have 60 FPS, so we might want to lower this
        # Rough heuristic for optimal parameter: delta = 1.0 -> objects tend to move one pixel per frame
        ats_delta = getattr(config, 'rope_ats_delta', 2.0)

        base_freqs = RotaryEmbedding(dim=sum(dims.values()), freqs_for='lang', theta=theta).freqs.float()

        freqs_spatial, freqs_t = torch.split(base_freqs, [(dims['x'] + dims['y']) // 2, dims['t'] // 2])
        freqs_x, freqs_y = freqs_spatial[::2], freqs_spatial[1::2]

        x_pos, y_pos, t_pos = self._create_positions(F, H, W, ats_delta)

        angles_x = x_pos[:, None] * freqs_x[None, :]
        angles_y = y_pos[:, None] * freqs_y[None, :]
        angles_t = t_pos[:, None] * freqs_t[None, :]

        interleaved_spatial = eo.rearrange(
            torch.stack([angles_x, angles_y], dim=-1),
            'b n two -> b (n two)'
        )

        freqs = torch.cat([interleaved_spatial, angles_t], dim=-1)

        if not getattr(config, "has_audio", False):
            freqs = freqs.view(config.n_frames, -1, freqs.size(-1))[:, :-1].flatten(0, 1)
        return freqs

    def _create_positions(self, n_frames, height, width, ats_delta):
        # Base 1D grids for time, height, and width
        t_grid = torch.arange(n_frames, dtype=torch.float32) * ats_delta
        h_grid = torch.linspace(-1., 1., steps=height, dtype=torch.float32) * ((height - 1) / 2.0)
        w_grid = torch.linspace(-1., 1., steps=width, dtype=torch.float32) * ((width - 1) / 2.0)

        # Create flattened position lists for video and audio
        t_video = eo.repeat(t_grid, 'f -> (f h w)', h=height, w=width)
        x_video = t_video + eo.repeat(w_grid, 'w -> (f h w)', f=n_frames, h=height)
        y_video = t_video + eo.repeat(h_grid, 'h -> (f h w)', f=n_frames, w=width)

        t_audio = eo.repeat(t_grid, 'f -> f')
        x_audio = t_audio
        y_audio = t_audio + (height - 1) / 2.0 + 1.0

        # Stack x, y, t components to process them together
        # Shape: (3, F*H*W) for video, (3, F) for audio
        stacked_video = torch.stack([x_video, y_video, t_video])
        stacked_audio = torch.stack([x_audio, y_audio, t_audio])

        # Reshape video to (d f n) and audio to (d f 1)
        stacked_video = eo.rearrange(stacked_video, 'd (f n) -> d f n', f=n_frames)
        stacked_audio = eo.rearrange(stacked_audio, 'd f -> d f 1')

        # Interleave by concatenating along the token dimension
        interleaved = torch.cat([stacked_video, stacked_audio], dim=2)

        # Flatten back into final (x, y, t) position lists
        x_pos, y_pos, t_pos = eo.rearrange(interleaved, 'd f n -> d (f n)').unbind(0)

        return x_pos, y_pos, t_pos
