"""
Variants of RoPE were becoming heavy for embeddings so 
I made a unique script for all of them here
"""

from rotary_embedding_torch import (
    RotaryEmbedding,
    apply_rotary_emb
)
import einops as eo
import torch
from torch import nn

class VideoRoPE(nn.Module):
    """
    Video RoPE embedding for when latents are 3D [n,h,w]
    """
    def __init__(self, config : 'TransformerConfig'):
        super().__init__()

        dim_head = config.d_model // config.n_heads
        self.pos_emb = RotaryEmbedding(
            dim = dim_head//8,
            freqs_for = 'pixel',
            max_freq = 256
        )
        n_patches = config.sample_size // config.patch_size
        self.tokens_per_frame = n_patches**2

        self.rearrange_in = lambda x: eo.rearrange(x, 'b h (n_t n_y n_x) d -> b h n_t n_y n_x d', n_y = n_patches)
        self.rearrange_out = lambda x: eo.rearrange(x, 'b h n_t n_y n_x d -> b h (n_t n_y n_x) d')
        self.get_freqs = lambda n_t: self.pos_emb.get_axial_freqs(n_t, n_patches, n_patches)

    def forward(self, q, k):
        # q k both [b,h,n,d]
        q = self.rearrange_in(q)
        k = self.rearrange_in(k)

        n_t = q.shape[2]
        freqs = self.get_freqs(n_t)

        q = apply_rotary_emb(freqs.float(), q.float()).to(q.dtype)
        k = apply_rotary_emb(freqs.float(), k.float()).to(k.dtype)

        q = self.rearrange_out(q)
        k = self.rearrange_out(k)
        
        return q, k

class FlatVideoRoPE(nn.Module):
    """
    Half-flat of RoPE that treats [n_frames, tokens_per_frame] as [n_frames, tokens_per_frame] image
    """
    def __init__(self, config):
        super().__init__()

        dim_head = config.d_model // config.n_heads
        self.pos_emb = RotaryEmbedding(
            dim = dim_head//4,
            freqs_for='pixel',
            max_freq=256
        )

        self.m = config.tokens_per_frame

    def pad_q(self, q, k):
        # Pad Q when it's needed for kv caching
        q_len = q.shape[2]
        k_len = k.shape[2]

    def forward(self, q, k):
        # q|k is [b,h,n_frames*tokens_per_frame,d]
        n = k.shape[2]//self.m
        m = self.m

        truncate = n
        if q.shape[2] < n * m:
            truncate = q.shape[2]//m # How many frames is q?

        q = eo.rearrange(q, 'b h (n m) d -> b h n m d', n=q.shape[2]//m,m=m)
        k = eo.rearrange(k, 'b h (n m) d -> b h n m d', n=n,m=m)

        with torch.no_grad():
            freqs = self.pos_emb.get_axial_freqs(n,m)
        q = apply_rotary_emb(freqs[-truncate:].detach(), q)
        k = apply_rotary_emb(freqs.detach(), k)

        q = eo.rearrange(q, 'b h n m d -> b h (n m) d')
        k = eo.rearrange(k, 'b h n m d -> b h (n m) d')

        if truncate is not None:
            q = q[:,:,-truncate*m:]

        return q,k



