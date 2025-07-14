"""
Variants of RoPE were becoming heavy for embeddings so
I made a unique script for all of them here
"""

from rotary_embedding_torch import (
    RotaryEmbedding,
    apply_rotary_emb
)
import torch
from torch import nn
import time


class FlatVideoRoPE(nn.Module):
    """
    RoPE on video + audio assuming each frame flat'd to [n_frame_toks+n_audio_toks]
    """
    def __init__(self, cfg):
        super().__init__()
        d_head = cfg.d_model // cfg.n_heads
        self.m = cfg.tokens_per_frame
        self.p = cfg.sample_size  # video is PxP pixels
        assert self.m == self.p**2 + 1

        # pre-compute cos / sin tables
        vid_ang = RotaryEmbedding(d_head // 4, freqs_for="pixel", max_freq=256)\
            .get_axial_freqs(config.n_frames, self.p, self.p)
        aud_ang = RotaryEmbedding(d_head // 2)\
            .get_axial_freqs(config.n_frames)

        self.register_buffer("vcos", vid_ang.cos(), persistent=False)
        self.register_buffer("vsin", vid_ang.sin(), persistent=False)
        self.register_buffer("acos", aud_ang.cos(), persistent=False)
        self.register_buffer("asin", aud_ang.sin(), persistent=False)

    @staticmethod
    def _rot(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        # Rotate first half of each channel
        rot_dim = cos.size(-1)
        n = x.shape[2]
        cos, sin = cos[:n], sin[:n]   # apply offset to cos
        x_rot, x_pass = x.float().split([rot_dim, x.size(-1) - rot_dim], dim=-1)
        x_pair = x_rot.view(*x_rot.shape[:-1], rot_dim // 2, 2)
        x_pair = torch.stack((-x_pair[..., 1], x_pair[..., 0]), dim=-1).view_as(x_rot)
        x_rot = x_rot * cos + x_pair * sin
        return torch.cat((x_rot, x_pass), dim=-1).type_as(x)

    def forward(self, q: torch.torch.Tensor, k: torch.torch.Tensor):
        assert self.vcos.dtype == torch.float32  # RoPE numerically unstable w/ bf16
        b, h, tq, d = q.shape
        nk, nq = k.size(2) // self.m, tq // self.m

        q = q.view(b, h, nq, self.m, d)
        k = k.view(b, h, nk, self.m, d)

        # split video / audio
        qv, qa = q[..., :self.p**2, :], q[..., self.p**2, :]
        kv, ka = k[..., :self.p**2, :], k[..., self.p**2, :]

        # HxH pixel video RoPE on q/k
        q_grid_shape = b, h, nq, self.p, self.p, d
        kv_grid_shape = b, h, nk, self.p, self.p, d
        qv = self._rot(qv.view(*q_grid_shape), self.vcos, self.vsin).reshape_as(qv)
        kv = self._rot(kv.view(*kv_grid_shape), self.vcos, self.vsin).reshape_as(kv)

        # audio RoPE on q/k
        qa = self._rot(qa, self.acos, self.asin).unsqueeze(-2)
        ka = self._rot(ka, self.acos, self.asin).unsqueeze(-2)

        # Recombine
        q = torch.cat((qv, qa), dim=-2).view(b, h, nq * self.m, d)
        k = torch.cat((kv, ka), dim=-2).view(b, h, nk * self.m, d)
        return q, k

class FrameRoPE(nn.Module):
    """
    RoPE variant that treats audio as R+1,C+1 part of a frame
    """
    def __init__(self, config):
        super().__init__()


        dim_head = config.d_model // config.n_heads
        pos_emb = RotaryEmbedding(
            dim = dim_head//6, # Using half dimension since we only need 1D rotation
            freqs_for='pixel',
            max_freq=256
        )

        self.m = config.tokens_per_frame
        self.p = config.sample_size
        self.p2 = self.p**2

        self.register_buffer(
            "freqs",
            pos_emb.get_axial_freqs(config.n_frames, self.p+1, self.p+1),
            persistent=False
        )

    def forward(self, q, k, offset=0):
        b,h,_,d = k.shape

        # q|k is [b,h,n_frames*tokens_per_frame,d]
        n = k.shape[2]//self.m  # Number of frames
        n_q = q.shape[2]//self.m
        m = self.m             # Tokens per frame

        # Reshape to [b,h,n,m,d]
        q = q.view(q.shape[0], q.shape[1], n_q, m, q.shape[3])
        k = k.view(k.shape[0], k.shape[1], n, m, k.shape[3])

        # Split out the video and audio
        # bhnmd-> bhn(16)d, bhnd
        q_video = q[:,:,:,:self.p2]
        q_video = q_video.view(
            b,
            h,
            n_q, self.p, self.p,
            d
        ) # b h n_q p p d
        k_video = k[:,:,:,:self.p2]
        k_video = k_video.view(
            b,
            h,
            n, self.p, self.p,
            d
        ) # b h n_q p p d
        q_audio = q[:,:,:,-1]
        k_audio = k[:,:,:,-1] # bhnd

        # Right pad and bottom pad
        with torch.no_grad():
            right_pad = torch.zeros(b, h, n, self.p, 1, d, device=k.device, dtype=k.dtype)
            bottom_pad = torch.zeros(b, h, n, 1, self.p+1, d, device=k.device, dtype=k.dtype)

        q_video = torch.cat([q_video, right_pad[:,:,:n_q]], dim = -2)
        q_video = torch.cat([q_video, bottom_pad[:,:,:n_q]], dim = -3)
        k_video = torch.cat([k_video, right_pad], dim = -2)
        k_video = torch.cat([k_video, bottom_pad], dim = -3)
        # all b h n|n_q (p+1) (p+1) d

        q_video[:,:,:,-1,-1] = q_audio
        k_video[:,:,:,-1,-1] = k_audio

        with torch.no_grad():
            freqs = self.freqs[:n].detach()

        q_video = apply_rotary_emb(freqs[-n_q:].detach(), q_video)
        k_video = apply_rotary_emb(freqs.detach(), k_video)

        q_audio = q_video[:,:,:,-1,-1]
        k_audio = k_video[:,:,:,-1,-1]

        q_video = q_video[:,:,:,:-1,:-1]
        k_video = k_video[:,:,:,:-1,:-1]

        q_video = q_video.reshape(
            b,
            h,
            n_q, self.p2,
            d
        ) # bhn(16)d
        q = torch.cat([q_video, q_audio.unsqueeze(-2)], dim = -2)

        k_video = k_video.reshape(
            b,
            h,
            n, self.p2,
            d
        )
        k = torch.cat([k_video, k_audio.unsqueeze(-2)], dim = -2)

        q = q.view(b,h,n_q*m,d)
        k = k.view(b,h,n*m,d)

        return q, k

def test_rope_speed():
    """
    Test the speed of RoPE implementation
    """
    from ..configs import TransformerConfig
    import time

    # Create test configs matching AV model
    config = TransformerConfig(
        n_heads=24,
        d_model=1536,
        tokens_per_frame=17,
        sample_size=4,
        n_frames = 60
    )

    # Create model and inputs
    rope = FlatVideoRoPE(config).cuda()

    batch_size = 32
    seq_len = 60 * config.tokens_per_frame # 60 frames
    d_head = config.d_model // config.n_heads

    q = torch.randn(batch_size, config.n_heads, seq_len, d_head).cuda()
    k = torch.randn(batch_size, config.n_heads, seq_len, d_head).cuda()

    # Warmup
    for _ in range(3):
        rope(q[:1], k[:1])
    torch.cuda.synchronize()

    # Benchmark
    n_trials = 100
    start = time.time()
    for _ in range(n_trials):
        rope(q, k)
    torch.cuda.synchronize()
    end = time.time()

    avg_time = (end - start) / n_trials * 1000 # Convert to ms
    print(f"Average RoPE time: {avg_time:.2f}ms")
    print(f"Batch size: {batch_size}, Sequence length: {seq_len}")
    print(f"Heads: {config.n_heads}, Head dim: {d_head}")


def test_flat_video_rope_integrity():
    from types import SimpleNamespace
    import numpy as np

    cfg = SimpleNamespace(
        d_model=512,
        n_heads=8,
        tokens_per_frame=17,      # 16 video + 1 audio
        sample_size=4,       # 4×4 → 16 video tokens
        n_frames=8
    )

    rope = FlatVideoRoPE(cfg)  # use cpu for cross-device seeded rng

    # use numpy rng - reproducable across machines
    rng = np.random.default_rng(seed=0)
    shape = (2, cfg.n_heads, cfg.n_frames * cfg.tokens_per_frame, cfg.d_model)
    q = torch.from_numpy(rng.standard_normal(size=shape).astype(np.float32))
    k = torch.from_numpy(rng.standard_normal(size=shape).astype(np.float32))

    with torch.no_grad():
        q_out, k_out = rope(q, k)

    checksum = (q_out.sum() + k_out.sum()).item()

    assert checksum == 484.2119140625, checksum
    print("FlatVideoRoPE implementation consistent")


if __name__ == "__main__":
    test_rope_speed()
