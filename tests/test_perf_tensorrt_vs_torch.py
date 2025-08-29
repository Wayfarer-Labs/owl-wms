import os
import time
import logging

import torch

from owl_wms.utils.owl_vae_bridge import (
    get_cod_yt_v2_decoder,
    make_batched_decode_fn,
    TENSORRT_AVAILABLE,
    unwrap_weight_norm,
)


def _time_runs(decode_fn, x, warmup: int = 3, runs: int = 10) -> float:
    """Return average latency in milliseconds for decode_fn(x)."""
    # Warmup
    for _ in range(warmup):
        _ = decode_fn(x)
        torch.cuda.synchronize() if torch.cuda.is_available() else None

    # Measure
    start = time.perf_counter()
    for _ in range(runs):
        _ = decode_fn(x)
        torch.cuda.synchronize() if torch.cuda.is_available() else None
    end = time.perf_counter()
    return (end - start) * 1000.0 / runs


def test_perf_tensorrt_vs_compiled_torch():
    if not torch.cuda.is_available():
        logging.warning("CUDA not available; skipping performance comparison test")
        return

    # Load decoder and move to CUDA for both paths. Unwrap weight norm to avoid
    # parametrization device mismatches under torch.compile.
    decoder = get_cod_yt_v2_decoder()
    decoder = unwrap_weight_norm(decoder).cuda().eval()

    # Input latent video: [batch, n, c, h, w]
    batch_size = 4
    seq = 1
    c, h, w = 128, 8, 8
    x = torch.randn(batch_size, seq, c, h, w, device="cuda", dtype=torch.float32)

    # Baseline PyTorch (eager) decoder: mirror the batching logic without compile
    def make_baseline_decode_fn(decoder_module, bs: int):
        @torch.no_grad()
        def decode(t):
            b, n, c2, h2, w2 = t.shape
            t = t.view(b * n, c2, h2, w2).contiguous()
            pieces = t.split(bs)
            outs = []
            for piece in pieces:
                outs.append(decoder_module(piece).bfloat16())
            out = torch.cat(outs)
            _, c3, h3, w3 = out.shape
            return out.view(b, n, c3, h3, w3).contiguous()
        return decode

    baseline_decode = make_baseline_decode_fn(decoder, batch_size)
    baseline_ms = _time_runs(baseline_decode, x)

    # Compiled PyTorch fallback (force-disable TensorRT at creation time for construction)
    os.environ["TENSORRT_ENABLED"] = "false"
    torch_decode = make_batched_decode_fn(decoder, batch_size=batch_size, use_tensorrt=False)
    torch_ms = _time_runs(torch_decode, x)

    # TensorRT path, if available; otherwise, report and exit
    if not TENSORRT_AVAILABLE:
        logging.warning("TensorRT not available; measured PyTorch fallback only: %.2f ms", torch_ms)
        return

    # Enable TRT; builder may export ONNX on first run. This can be slow once but cached later.
    os.environ["TENSORRT_ENABLED"] = "true"
    trt_decode = make_batched_decode_fn(decoder, batch_size=batch_size, use_tensorrt=True)
    trt_ms = _time_runs(trt_decode, x)

    # Report results and speedup (multipliers)
    speedup_vs_baseline = (baseline_ms / trt_ms) if trt_ms > 0 else float('inf')
    speedup_vs_compiled = (torch_ms / trt_ms) if trt_ms > 0 else float('inf')
    print(f"PyTorch (baseline) avg latency: {baseline_ms:.2f} ms")
    print(f"PyTorch (compiled) avg latency: {torch_ms:.2f} ms")
    print(f"TensorRT avg latency: {trt_ms:.2f} ms")
    print(f"TensorRT is {speedup_vs_baseline:.2f}x faster than PyTorch (baseline) and {speedup_vs_compiled:.2f}x faster than PyTorch (compiled)")

    # Basic sanity assert: both paths produce tensors of same shape
    with torch.no_grad():
        out_torch = torch_decode(x)
        out_trt = trt_decode(x)
        out_baseline = baseline_decode(x)
    assert out_torch.shape == out_trt.shape == out_baseline.shape


