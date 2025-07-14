from owl_wms.nn.rope import FlatVideoRoPE, FlatVideoRoPE_v2
from owl_wms.configs import TransformerConfig
import torch
from torch import nn
import torch.nn.functional as F
import time

if __name__ == "__main__":
    import time

    # Set device (use cuda if available for fair benchmarking)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    q = torch.randn(1, 24, 1020, 64, device=device)
    k = torch.randn(1, 24, 1020, 64, device=device)

    config = TransformerConfig(
        n_layers = 24,
        n_heads = 24,
        d_model = 1536,
        tokens_per_frame = 17,
        sample_size = 4,
        n_frames = 60
    )

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

    # Compute error between outputs
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