from owl_wms.nn.rope import FlatVideoRoPE, FlatVideoRoPE_v2
from owl_wms.configs import TransformerConfig
import torch
import time

if __name__ == "__main__":
    # Set device (use cuda if available for fair benchmarking)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Config
    config = TransformerConfig(
        n_layers = 24,
        n_heads = 24,
        d_model = 1536,
        tokens_per_frame = 17,
        sample_size = 4,
        n_frames = 60
    )

    # Input shapes
    batch = 1
    n_heads = config.n_heads
    seq_len = config.n_frames * config.tokens_per_frame  # 60*17=1020
    d_head = config.d_model // config.n_heads

    # Generate random q and k
    q = torch.randn(batch, n_heads, seq_len, d_head, device=device)
    k = torch.randn(batch, n_heads, seq_len, d_head, device=device)

    # Run both RoPE implementations
    rope1 = FlatVideoRoPE(config).to(device).eval()
    rope2 = FlatVideoRoPE_v2(config).to(device).eval()

    # Warmup
    for _ in range(3):
        with torch.no_grad():
            _ = rope1(q, k)
            _ = rope2(q, k)
    if device.type == "cuda":
        torch.cuda.synchronize()

    # Benchmark rope1
    n_trials = 100
    with torch.no_grad():
        start = time.time()
        for _ in range(n_trials):
            q1, k1 = rope1(q, k)
        if device.type == "cuda":
            torch.cuda.synchronize()
        end = time.time()
    rope1_latency = (end - start) / n_trials * 1000  # ms

    # Benchmark rope2
    with torch.no_grad():
        start = time.time()
        for _ in range(n_trials):
            q2, k2 = rope2(q, k)
        if device.type == "cuda":
            torch.cuda.synchronize()
        end = time.time()
    rope2_latency = (end - start) / n_trials * 1000  # ms

    # Compute error between outputs (full sequence)
    q_err = (q1 - q2).abs().mean().item()
    k_err = (k1 - k2).abs().mean().item()

    print("q error (FlatVideoRoPE vs FlatVideoRoPE_v2):", q_err)
    print("k error (FlatVideoRoPE vs FlatVideoRoPE_v2):", k_err)

    # As a reference, compare two random tensors of same shape
    q_rand1 = torch.randn_like(q)
    q_rand2 = torch.randn_like(q)
    rand_err = (q_rand1 - q_rand2).abs().mean().item()
    print("Reference random error (randn vs randn):", rand_err)

    print("q1 shape:", q1.shape, "k1 shape:", k1.shape)
    print(f"FlatVideoRoPE latency: {rope1_latency:.3f} ms")
    print(f"FlatVideoRoPE_v2 latency: {rope2_latency:.3f} ms")

    # ---- KV Cache Correctness Test ----
    # Simulate a "cache" scenario: only the last tokens_per_frame tokens of q are given, but k is full
    q_cache = q[:, :, -config.tokens_per_frame:, :]  # [1, 24, 17, 64]
    # For v1, get the full output
    with torch.no_grad():
        q1_full, k1_full = rope1(q, k)
    # For v2, give only the cache q, but full k
    with torch.no_grad():
        q2_cache, k2_full = rope2(q_cache, k)

    # Compare k outputs (should be identical, since k is always full)
    k_cache_err = (k1_full - k2_full).abs().mean().item()
    print("KV cache test: k error (full k, v1 vs v2):", k_cache_err)

    # Compare q outputs: v2's q should match the last tokens_per_frame tokens of v1's q
    q1_cache = q1_full[:, :, -config.tokens_per_frame:, :]  # [1, 24, 17, 64]
    q_cache_err = (q1_cache - q2_cache).abs().mean().item()
    print("KV cache test: q error (last 17 q, v1 vs v2):", q_cache_err)

    # Print shapes for sanity
    print("q1_cache shape:", q1_cache.shape, "q2_cache shape:", q2_cache.shape)
    print("k1_full shape:", k1_full.shape, "k2_full shape:", k2_full.shape)