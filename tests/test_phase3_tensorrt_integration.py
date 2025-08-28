import os
import time
from pathlib import Path

import pytest
import torch

from owl_wms.utils.owl_vae_bridge import (
    get_cod_yt_v2_decoder,
    TensorRTVideoDecoder,
    make_batched_decode_fn,
    export_video_decoder_to_onnx,
    build_tensorrt_engine,
    TensorRTConfig,
)


def _ensure_engine(engine_path: Path, batch_size: int = 8) -> Path:
    engine_path = Path(engine_path)
    engine_path.parent.mkdir(parents=True, exist_ok=True)
    if not engine_path.exists():
        onnx_path = engine_path.with_suffix(".onnx")
        decoder = get_cod_yt_v2_decoder()
        export_video_decoder_to_onnx(decoder, onnx_path, batch_size=batch_size, validate_ort=False)
        ok = build_tensorrt_engine(onnx_path, engine_path, TensorRTConfig(max_batch_size=batch_size))
        assert ok and engine_path.exists(), "Failed to build TRT engine for tests"
    return engine_path


@pytest.mark.cuda
def test_trt_wrapper_parity_and_speed(tmp_path: Path):
    engine = _ensure_engine(Path("/root/owl-wms/tensorrt_cache/dcae_video_decoder.trt"))

    decoder = get_cod_yt_v2_decoder().cuda().eval()
    latent = torch.randn(8, 128, 8, 8, dtype=torch.float32, device="cuda")

    torch.cuda.synchronize(); t0 = time.perf_counter()
    with torch.no_grad():
        out_pt = decoder(latent).float()
    torch.cuda.synchronize(); t1 = time.perf_counter()

    trt_decoder = TensorRTVideoDecoder(engine)
    torch.cuda.synchronize(); t2 = time.perf_counter()
    out_trt = trt_decoder(latent).float()
    torch.cuda.synchronize(); t3 = time.perf_counter()

    assert out_pt.shape == out_trt.shape
    max_diff = (out_pt - out_trt).abs().max().item()
    assert max_diff < 0.1

    pt_ms = (t1 - t0) * 1000
    trt_ms = (t3 - t2) * 1000
    assert pt_ms / max(trt_ms, 1e-6) > 1.5  # at least 1.5x faster


@pytest.mark.cuda
def test_make_batched_decode_fn_trt_path(tmp_path: Path):
    os.environ.setdefault("TENSORRT_ENGINE_PATH", "/root/owl-wms/tensorrt_cache/dcae_video_decoder.trt")
    _ensure_engine(Path(os.environ["TENSORRT_ENGINE_PATH"]))

    decoder = get_cod_yt_v2_decoder().cuda().eval()
    decode_pt = make_batched_decode_fn(decoder, batch_size=8, use_tensorrt=False)
    decode_trt = make_batched_decode_fn(decoder, batch_size=8, use_tensorrt=True)

    b, n = 1, 2
    latent = torch.randn(b, n, 128, 8, 8, dtype=torch.float32, device="cuda")

    torch.cuda.synchronize(); t0 = time.perf_counter()
    out_pt = decode_pt(latent).float()
    torch.cuda.synchronize(); t1 = time.perf_counter()

    torch.cuda.synchronize(); t2 = time.perf_counter()
    out_trt = decode_trt(latent).float()
    torch.cuda.synchronize(); t3 = time.perf_counter()

    assert out_pt.shape == out_trt.shape
    max_diff = (out_pt - out_trt).abs().max().item()
    assert max_diff < 0.1

    pt_ms = (t1 - t0) * 1000
    trt_ms = (t3 - t2) * 1000
    assert pt_ms / max(trt_ms, 1e-6) > 1.5


@pytest.mark.cuda
def test_fallback_on_tensorrt_unavailable(monkeypatch):
    import owl_wms.utils.owl_vae_bridge as bridge
    # Force TensorRT path to be unavailable
    monkeypatch.setattr(bridge, "TENSORRT_AVAILABLE", False, raising=False)

    decoder = get_cod_yt_v2_decoder().cuda().eval()
    decode_fn = make_batched_decode_fn(decoder, batch_size=8, use_tensorrt=True)

    b, n = 1, 1
    latent = torch.randn(b, n, 128, 8, 8, dtype=torch.float32, device="cuda")
    out = decode_fn(latent)
    assert out.shape[0] == b and out.shape[1] == n  # ran via PyTorch fallback


@pytest.mark.cuda
def test_memory_stability_over_iterations():
    os.environ.setdefault("TENSORRT_ENGINE_PATH", "/root/owl-wms/tensorrt_cache/dcae_video_decoder.trt")
    _ensure_engine(Path(os.environ["TENSORRT_ENGINE_PATH"]))

    decoder = get_cod_yt_v2_decoder().cuda().eval()
    decode_trt = make_batched_decode_fn(decoder, batch_size=8, use_tensorrt=True)

    b, n = 1, 8
    latent = torch.randn(b, n, 128, 8, 8, dtype=torch.float32, device="cuda")

    # Warmup
    for _ in range(5):
        _ = decode_trt(latent)
        torch.cuda.synchronize()

    torch.cuda.reset_peak_memory_stats()
    mem_before = torch.cuda.memory_allocated()
    for _ in range(100):
        _ = decode_trt(latent)
    torch.cuda.synchronize()
    mem_after = torch.cuda.memory_allocated()

    # Allow small fluctuations (< 64MB)
    assert (mem_after - mem_before) < 64 * 1024 * 1024


