import os
import time
from pathlib import Path

import pytest
import torch

from owl_wms.utils.owl_vae_bridge import (
    get_cod_yt_v2_decoder,
    make_batched_decode_fn,
)


@pytest.mark.cuda
def test_env_var_toggles_and_monitoring_logs(caplog):
    # Ensure engine exists beforehand
    engine = Path("/root/owl-wms/tensorrt_cache/dcae_video_decoder.trt")
    assert engine.exists(), "Expected cached TRT engine not found; build earlier steps first"

    # Force INFO logging capture
    caplog.set_level("INFO")

    # 1) Disable TRT via env -> should only log PyTorch timing
    os.environ["TENSORRT_ENABLED"] = "false"
    dec = get_cod_yt_v2_decoder().cuda().eval()
    decode = make_batched_decode_fn(dec, batch_size=8, use_tensorrt=True)
    latent = torch.randn(1, 8, 128, 8, 8, device="cuda")
    with torch.no_grad():
        _ = decode(latent)
    assert any("PyTorch Video Decode inference time" in r.message for r in caplog.records)
    assert not any("TensorRT Video Decode inference time" in r.message for r in caplog.records)

    # 2) Enable TRT and set explicit engine path -> should log TensorRT timing
    caplog.clear()
    os.environ["TENSORRT_ENABLED"] = "true"
    os.environ["TENSORRT_ENGINE_PATH"] = str(engine)
    dec = get_cod_yt_v2_decoder().cuda().eval()
    decode = make_batched_decode_fn(dec, batch_size=8, use_tensorrt=True)
    latent = torch.randn(1, 8, 128, 8, 8, device="cuda")
    with torch.no_grad():
        _ = decode(latent)
    assert any("TensorRT Video Decode inference time" in r.message for r in caplog.records)


@pytest.mark.cuda
def test_precision_and_cache_dir_overrides(tmp_path: Path):
    # Override precision and cache dir; just verify execution completes
    os.environ["TENSORRT_ENABLED"] = "true"
    os.environ["TENSORRT_PRECISION"] = "fp32"
    os.environ["TENSORRT_CACHE_DIR"] = str(tmp_path / "trt_cache")

    dec = get_cod_yt_v2_decoder().cuda().eval()
    decode = make_batched_decode_fn(dec, batch_size=8, use_tensorrt=True)
    latent = torch.randn(1, 8, 128, 8, 8, device="cuda")
    with torch.no_grad():
        out = decode(latent)
    assert out.shape[-2:] == (360, 640)
    assert Path(os.environ["TENSORRT_CACHE_DIR"]).exists()


@pytest.mark.cuda
def test_engine_reuse():
    # Using default cached engine should run quickly on second call
    os.environ["TENSORRT_ENABLED"] = "true"
    os.environ.pop("TENSORRT_PRECISION", None)
    os.environ.pop("TENSORRT_CACHE_DIR", None)

    dec = get_cod_yt_v2_decoder().cuda().eval()
    decode = make_batched_decode_fn(dec, batch_size=8, use_tensorrt=True)
    latent = torch.randn(1, 8, 128, 8, 8, device="cuda")

    # Warm-up
    with torch.no_grad():
        _ = decode(latent)

    # Timed run
    torch.cuda.synchronize(); t0 = time.perf_counter()
    with torch.no_grad():
        _ = decode(latent)
    torch.cuda.synchronize(); t1 = time.perf_counter()
    elapsed_ms = (t1 - t0) * 1000
    assert elapsed_ms < 500, f"Expected fast cached engine decode, got {elapsed_ms:.2f}ms"


