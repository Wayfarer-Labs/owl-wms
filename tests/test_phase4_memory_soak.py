import os
import torch
import pytest

from owl_wms.utils.owl_vae_bridge import get_cod_yt_v2_decoder, make_batched_decode_fn


@pytest.mark.cuda
def test_trt_memory_soak_over_1000_iterations():
    os.environ.setdefault("TENSORRT_ENABLED", "true")
    decoder = get_cod_yt_v2_decoder().cuda().eval()
    decode_trt = make_batched_decode_fn(decoder, batch_size=8, use_tensorrt=True)

    b, n = 1, 8
    latent = torch.randn(b, n, 128, 8, 8, dtype=torch.float32, device="cuda")

    # Warm up
    for _ in range(10):
        _ = decode_trt(latent)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    mem_before = torch.cuda.memory_allocated()

    for _ in range(1000):
        _ = decode_trt(latent)
    torch.cuda.synchronize()
    mem_after = torch.cuda.memory_allocated()

    # Allow small fluctuations (< 128MB)
    assert (mem_after - mem_before) < 128 * 1024 * 1024


