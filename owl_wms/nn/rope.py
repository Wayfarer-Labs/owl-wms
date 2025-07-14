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
    def __init__(self, config):
        super().__init__()


        dim_head = config.d_model // config.n_heads
        self.pos_emb_video = RotaryEmbedding(
            dim = dim_head//4, # Using half dimension since we only need 1D rotation
            freqs_for='pixel',
            max_freq=256
        )
        self.pos_emb_audio = RotaryEmbedding(
            dim = dim_head//2
        )

        self.m = config.tokens_per_frame
        self.p = config.sample_size
        self.p2 = self.p**2

        self.register_buffer(
            "vid_freqs_cache",
            self.pos_emb_video.get_axial_freqs(config.n_frames, self.p, self.p),
            persistent=False
        )
        self.register_buffer(
            "audio_freqs_cache",
            self.pos_emb_audio.get_axial_freqs(config.n_frames),
            persistent=False
        )

    #@torch.compile(mode='max-autotune', dynamic=False, fullgraph=True)
    def forward(self, q, k):
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

        # Check for shape match between cache and k. If fail, reset cache
        with torch.no_grad():
            vid_freqs = self.vid_freqs_cache[:n]
            audio_freqs = self.audio_freqs_cache[:n]
            
        # Apply RoPE to video tokens
        q_video = apply_rotary_emb(vid_freqs[-n_q:].detach(), q_video)
        k_video = apply_rotary_emb(vid_freqs.detach(), k_video)

        # Apply RoPE to audio tokens
        q_audio = apply_rotary_emb(audio_freqs[-n_q:].detach(), q_audio)
        k_audio = apply_rotary_emb(audio_freqs.detach(), k_audio)

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

if __name__ == "__main__":
    test_rope_speed()

