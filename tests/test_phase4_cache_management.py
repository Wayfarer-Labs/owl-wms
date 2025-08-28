import os
from pathlib import Path
import pytest
import torch

from owl_wms.utils.owl_vae_bridge import (
    get_cod_yt_v2_decoder,
    make_batched_decode_fn,
    export_video_decoder_to_onnx,
    build_tensorrt_engine,
    TensorRTConfig,
)


@pytest.mark.cuda
def test_cache_pruning_by_size(tmp_path: Path):
    # Use a temporary cache dir
    cache_dir = tmp_path / "trt_cache"
    os.environ["TENSORRT_CACHE_DIR"] = str(cache_dir)
    os.environ["TENSORRT_ENABLED"] = "true"
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Build two engines with different filenames to fill cache
    onnx1 = tmp_path / "a.onnx"
    onnx2 = tmp_path / "b.onnx"
    dec = get_cod_yt_v2_decoder()
    export_video_decoder_to_onnx(dec, onnx1, batch_size=8, validate_ort=False)
    export_video_decoder_to_onnx(dec, onnx2, batch_size=8, validate_ort=False)

    eng1 = cache_dir / "engine_a.trt"
    eng2 = cache_dir / "engine_b.trt"
    build_tensorrt_engine(onnx1, eng1, TensorRTConfig())
    build_tensorrt_engine(onnx2, eng2, TensorRTConfig())

    # Set max cache size small enough to force pruning of at least one file
    os.environ["TENSORRT_MAX_CACHE_SIZE_MB"] = "1"  # 1MB

    # Trigger another build which should prune oldest
    onnx3 = tmp_path / "c.onnx"
    export_video_decoder_to_onnx(dec, onnx3, batch_size=8, validate_ort=False)
    eng3 = cache_dir / "engine_c.trt"
    build_tensorrt_engine(onnx3, eng3, TensorRTConfig())

    # After pruning, total size must be <= cap
    total_bytes = sum(p.stat().st_size for p in cache_dir.glob("*.trt") if p.exists())
    assert total_bytes <= 1 * 1024 * 1024


