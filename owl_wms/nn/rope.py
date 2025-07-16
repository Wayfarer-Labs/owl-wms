"""
Variants of RoPE were becoming heavy for embeddings so
I made a unique script for all of them here
"""

from ret import RotaryEmbedding, apply_rotary_emb
import torch
from torch import nn


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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x is either q or k. Shaped as [B, H, n_frames*tokens_per_frame, Dh]"""
        assert self.cos.dtype == torch.float32
        cos, sin = self.cos[..., :x.size(2), :], self.sin[..., :x.size(2), :]
        x0, x1 = x.float().unfold(-1, 2, 2).unbind(-1)
        y0 = x0 * cos - x1 * sin
        y1 = x1 * cos + x0 * sin
        return torch.cat((y0, y1), dim=-1).type_as(x)


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


@torch.no_grad()
def test_flat_video_rope_integrity():
    from types import SimpleNamespace
    import numpy as np

    cfg = SimpleNamespace(d_model=512, n_heads=8, tokens_per_frame=17, sample_size=4, n_frames=100)
    rope = FlatVideoRoPE(cfg)

    # use numpy rng - reproducable across machines
    rng = np.random.default_rng(seed=0)
    shape = (2, cfg.n_heads, cfg.n_frames * cfg.tokens_per_frame, cfg.d_model // cfg.n_heads)
    q = torch.from_numpy(rng.standard_normal(size=shape).astype(np.float32)).bfloat16()
    k = torch.from_numpy(rng.standard_normal(size=shape).astype(np.float32)).bfloat16()

    q_out, k_out = rope(q), rope(k)

    checksum = q_out.float().sum() + k_out.float().sum()
    ref_impl_checksum = torch.tensor(1678.122802734375)
    assert torch.allclose(checksum, ref_impl_checksum, rtol=1e-6), checksum.item()
    print("FlatVideoRoPE implementation consistent")


@torch.no_grad()
def test_flat_video_rope_dot_product_invariance():
    from types import SimpleNamespace
    import numpy as np

    cfg = SimpleNamespace(d_model=512, n_heads=8, tokens_per_frame=17, sample_size=4, n_frames=100)
    rope = FlatVideoRoPE(cfg)

    B, H, L, Dh = 1, cfg.n_heads, cfg.tokens_per_frame * cfg.n_frames, cfg.d_model // cfg.n_heads

    rng = np.random.default_rng(seed=0)

    for dtype, rtol in zip([torch.float32, torch.bfloat16], [1e-4, 1e-2]):
        q = torch.from_numpy(rng.standard_normal((B, H, 1, Dh))).repeat(1, 1, L, 1).to(dtype)
        k = torch.from_numpy(rng.standard_normal((B, H, 1, Dh))).repeat(1, 1, L, 1).to(dtype)

        q, k = rope(q), rope(k)

        # q/k idxs, equidistant (53 frames away from one another)
        q0, q1, k0, k1 = [0, 30, 53, 83]
        q0, q1, k0, k1 = [idx * cfg.tokens_per_frame for idx in (q0, q1, k0, k1)]

        # per-head dot product
        q, k = q.float(), k.float()  # ignore dot product accumulation precision, only testing RoPE
        dp1 = (q[0, :, q0, :] * k[0, :, k0, :]).sum(-1)
        dp2 = (q[0, :, q1, :] * k[0, :, k1, :]).sum(-1)

        # they should be equal up to numerical tolerance
        if not torch.allclose(dp1, dp2, rtol=rtol):
            raise ValueError(f"dot product not equivalent. Per head output:\ndp1={dp1}\ndp2={dp2}")
        print("RoPE dot-product check passed")


if __name__ == "__main__":
    test_flat_video_rope_integrity()
    test_flat_video_rope_dot_product_invariance()
    test_rope_speed()
