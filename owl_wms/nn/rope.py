"""
Variants of RoPE were becoming heavy for embeddings so
I made a unique script for all of them here
"""

from rotary_embedding_torch import RotaryEmbedding, apply_rotary_emb
import torch
from torch import nn

import einops as eo
from einops._torch_specific import allow_ops_in_compiled_graph  # requires einops>=0.6.1
allow_ops_in_compiled_graph()

class FlatVideoRoPE(nn.Module):
    """
    RoPE on video + audio assuming each frame flat'd to [n_frame_toks+n_audio_toks]
    Rotate non-pad portion of feature-dims of q/k. Video 1/4th padded, audio 1/2 padded
    """
    def __init__(self, config):
        super().__init__()
        d_head = config.d_model // config.n_heads
        p = config.sample_size  # video is PxP pixels

        # Video freqs. Rot features: (L, P, P, <pad>)
        vid_freqs = RotaryEmbedding(d_head // 4, freqs_for="pixel", max_freq=256)\
            .get_axial_freqs(config.n_frames, p, p, 1, offsets=(0, 0, 0, 1))\
            .view(config.n_frames, p**2, -1)

        # Audio freqs. Rot features: (L, <pad>)
        aud_freqs = RotaryEmbedding(d_head // 2)\
            .get_axial_freqs(config.n_frames, 1, offsets=(0, 1))\
            .view(config.n_frames, 1, -1)

        # unified video / audio freqs. Shape: [n_frames, P^2 + 1, H]
        freqs = torch.cat([vid_freqs, aud_freqs], dim=1).flatten(0, 1)

        cos, sin = freqs.cos()[..., ::2], freqs.sin()[..., ::2]  # subsampling
        self.cos = nn.Buffer(cos.contiguous(), persistent=False)
        self.sin = nn.Buffer(sin.contiguous(), persistent=False)

    def forward(self, x: torch.Tensor, offset: int = 0) -> torch.Tensor:
        """x is either q or k. Shaped as [B, H, n_frames*tokens_per_frame, Dh]"""
        assert self.cos.dtype == torch.float32
        cos, sin = self.cos[..., offset:x.size(2) + offset, :], self.sin[..., offset:x.size(2) + offset, :]
        x0, x1 = x.float().unfold(-1, 2, 2).unbind(-1)
        y0 = x0 * cos - x1 * sin
        y1 = x1 * cos + x0 * sin
        return torch.cat((y0, y1), dim=-1).type_as(x)


class AVRoPE(nn.Module):
    """
    RoPE variant that treats audio as R+1,C+1 part of a frame
    """
    def __init__(self, config):
        super().__init__()


        dim_head = config.d_model // config.n_heads
        pos_emb = RotaryEmbedding(
            dim = dim_head//4, # Using half dimension since we only need 1D rotation
            freqs_for='pixel',
            max_freq=256
        )

        self.m = config.tokens_per_frame
        self.p = config.sample_size
        self.p2 = self.p**2

        freqs = pos_emb.get_axial_freqs(
            config.n_frames, self.p+1, self.p+1
        )
        cos,sin = freqs.cos()[..., ::2], freqs.sin()[..., ::2]
        self.cos = nn.Buffer(cos.contiguous(), persistent=False)
        self.sin = nn.Buffer(sin.contiguous(), persistent=False)

    def forward(self, x_video, x_audio, offset=0):
        # x_video : [b,h,nhw,d] (q or k)
        # x_audio : [b,h,n,d] (q or k)

        x_video = eo.rearrange(x_video, 'b h (n y x) d -> b h n y x d', y = self.p, x = self.p)
        # No need to reshape audio
        
        # Pad with new row and column on both
        pad_right = torch.zeros_like(x_video[...,-1:,:])
        pad_bottom = torch.zeros_like(x_video[...,-1:,:,:])

        x_video = torch.cat([x_video, pad_right], dim = -2)
        x_video = torch.cat([x_video, pad_bottom], dim = -3)

        x_video[:,:,:,-1,-1] = x_audio

        # Do stuff with freqs
        cos, sin = self.cos[..., offset:x_video.size(2) + offset, :], self.sin[..., offset:x_video.size(2) + offset, :]
        x0, x1 = x_video.float().unfold(-1, 2, 2).unbind(-1)
        y0 = x0 * cos - x1 * sin
        y1 = x1 * cos + x0 * sin
        x_video = torch.cat((y0, y1), dim=-1).type_as(x_video)

        x_audio = x_video[:,:,:,-1,-1].clone()
        x_video = x_video[:,:,:,:-1,:-1]

        x_video = eo.rearrange(x_video, 'b h n y x d -> b h (n y x) d')
        
        return x_video, x_audio

def visaulize_rope_freqs():
    pos_emb = RotaryEmbedding(
        dim = dim_head//6, # Using half dimension since we only need 1D rotation
        freqs_for='pixel',
        max_freq=256
    )
    freqs = pos_emb.get_axial_freqs(16, 5, 5)