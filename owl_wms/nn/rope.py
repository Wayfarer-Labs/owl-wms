from rotary_embedding_torch import RotaryEmbedding
import torch
from torch import nn
from torch.cuda.amp import autocast

import einops as eo
from einops._torch_specific import allow_ops_in_compiled_graph  # requires einops>=0.6.1
allow_ops_in_compiled_graph()


def get_rope_cls(cls_name):
    cls_name = cls_name.lower()
    if cls_name == "ortho":
        return OrthoRoPE
    elif cls_name == "motion":
        return MotionRoPE
    else:
        raise ValueError(f"Invalid RoPE class: {cls_name}")


class RoPE(nn.Module):
    def __init__(self, config):
        super().__init__()
        freqs = self.get_freqs(config)

        if not config.has_audio:
            # subclasses freqs include audio by default, remove last item from each frame
            freqs = freqs.view(config.n_frames, -1, freqs.size(-1))[:, :-1].flatten(0, 1)
            freqs = freqs

        self.cos = nn.Buffer(freqs.cos().contiguous(), persistent=False)
        self.sin = nn.Buffer(freqs.sin().contiguous(), persistent=False)

    @autocast(enabled=False)
    def forward(self, x, offset: int = 0):
        assert self.cos.dtype == torch.float32
        cos = self.cos[..., offset:offset + x.size(2), :]
        sin = self.sin[..., offset:offset + x.size(2), :]
        x0, x1 = x.float().unfold(-1, 2, 2).unbind(-1)
        y0 = x0 * cos - x1 * sin
        y1 = x1 * cos + x0 * sin
        return torch.cat((y0, y1), dim=-1).type_as(x)

    def get_freqs(self, config):
        raise NotImplementedError


class OrthoRoPE(RoPE):
    """
    RoPE for rotation across orthogonal axes: time, height, and width
    """
    def get_freqs(self, config):
        p = config.sample_size
        head_dim = config.d_model // config.n_heads

        pos_emb = RotaryEmbedding(
            dim=head_dim // 4,  # Using half dimension since we only need 1D rotation
            freqs_for='pixel',
            max_freq=256
        )
        # Rot features: (L, P+1, P+1, <pad>)
        freqs = pos_emb.get_axial_freqs(
            config.n_frames, p + 1, p + 1, 1, offsets=(0, 0, 0, 1)
        ).view(config.n_frames, p + 1, p + 1, -1)

        vid_freqs = freqs[:, :p, :p].reshape(config.n_frames, p**2, -1)  # top left square
        aud_freqs = freqs[:, -1, -1].unsqueeze(1)  # bottom right item

        freqs = torch.cat([vid_freqs, aud_freqs], dim=1).flatten(0, 1)
        return freqs[..., ::2]  # subsampling


class MotionRoPE(RoPE):
    """
    https://arxiv.org/pdf/2502.05173
    RoPE implementing a diagonal layout where spatial coordinates are a linear function of time.
    This constant-velocity prior serves as a baseline for learning complex, non-linear motion.
    """
    def get_freqs(self, config):
        H, W = config.sample_size, config.sample_size
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

        return torch.cat([interleaved_spatial, angles_t], dim=-1)

    def _create_positions(self, n_frames, height, width, ats_delta):
        # Base 1D grids for time, height, and width
        t_grid = torch.arange(n_frames, dtype=torch.float32) * ats_delta
        h_grid = torch.arange(height, dtype=torch.float32) - (height - 1) / 2.0
        w_grid = torch.arange(width, dtype=torch.float32) - (width - 1) / 2.0

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


def visaulize_rope_freqs():
    pos_emb = RotaryEmbedding(
        dim = dim_head//6, # Using half dimension since we only need 1D rotation
        freqs_for='pixel',
        max_freq=256
    )
    freqs = pos_emb.get_axial_freqs(16, 5, 5)
