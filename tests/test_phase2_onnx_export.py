import os
from pathlib import Path
import numpy as np
import onnx
import pytest
import torch
import logging
import time
import timeout_decorator

from owl_wms.utils.owl_vae_bridge import (
    get_cod_yt_v2_decoder,
    export_video_decoder_to_onnx,
)


def _onnxruntime_session(model_path: Path):
    import onnxruntime as ort
    ort.set_default_logger_severity(0)  # Verbose logging for hang diagnosis
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    try:
        session = ort.InferenceSession(str(model_path), providers=providers)
        logging.info(f"ONNX Runtime session created with providers: {session.get_providers()}")
        return session
    except Exception as e:
        logging.error(f"Failed to create ONNX Runtime session: {e}")
        raise

@pytest.mark.cuda
def test_export_and_validate(tmp_path: Path):
    import time
    import logging
    
    # Set up detailed logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    print("=== Starting test_export_and_validate ===")
    
    # Step 1: Load decoder
    print("Step 1: Loading decoder...")
    start_time = time.time()
    try:
        decoder = get_cod_yt_v2_decoder()
        print(f"✓ Decoder loaded in {time.time() - start_time:.2f}s")
        print(f"  - Device: {next(decoder.parameters()).device}")
        print(f"  - Dtype: {next(decoder.parameters()).dtype}")
    except Exception as e:
        print(f"✗ Decoder loading failed: {e}")
        raise
    
    # Step 2: Export to ONNX
    print("Step 2: Exporting to ONNX...")
    start_time = time.time()
    onnx_path = tmp_path / "video_decoder.onnx"
    
    try:
        ok = export_video_decoder_to_onnx(decoder, onnx_path, batch_size=8)
        print(f"✓ Export completed in {time.time() - start_time:.2f}s")
        print(f"  - Success: {ok}")
        print(f"  - File size: {onnx_path.stat().st_size / 1024 / 1024:.1f} MB")
    except Exception as e:
        print(f"✗ Export failed: {e}")
        import traceback
        traceback.print_exc()
        raise
    
    assert ok is True

    # Step 3: Structural validation
    print("Step 3: Structural validation...")
    start_time = time.time()
    try:
        onnx_model = onnx.load(str(onnx_path))
        onnx.checker.check_model(onnx_model)
        print(f"✓ Structural validation completed in {time.time() - start_time:.2f}s")
    except Exception as e:
        print(f"✗ Structural validation failed: {e}")
        raise
    
    print("=== test_export_and_validate completed successfully ===")


@pytest.mark.cuda
def test_onnxruntime_inference_and_tolerance(tmp_path: Path):
    import time
    import logging
    
    print("=== Starting test_onnxruntime_inference_and_tolerance ===")
    
    # Step 1: Load decoder and export
    print("Step 1: Loading decoder and exporting...")
    start_time = time.time()
    try:
        decoder = get_cod_yt_v2_decoder()
        onnx_path = tmp_path / "video_decoder.onnx"
        export_video_decoder_to_onnx(decoder, onnx_path, batch_size=8, validate_ort=False)  # Skip ORT in export
        print(f"✓ Export completed in {time.time() - start_time:.2f}s")
    except Exception as e:
        print(f"✗ Export failed: {e}")
        logging.error(f"Export failed: {e}", exc_info=True)
        raise

    # Step 2: Prepare input
    print("Step 2: Preparing input...")
    batch_size = 8
    latent = torch.randn(batch_size, 128, 8, 8, dtype=torch.float32)
    print(f"  - Input shape: {latent.shape}")

    # Step 3: PyTorch forward
    print("Step 3: PyTorch forward pass...")
    start_time = time.time()
    with torch.no_grad():
        pyt_model = decoder.cpu().eval()
        torch_out = pyt_model(latent).float()
    print(f"✓ PyTorch forward completed in {time.time() - start_time:.2f}s")
    print(f"  - Output shape: {torch_out.shape}")

    # Step 4: ONNX Runtime forward
    print("Step 4: ONNX Runtime forward pass...")
    start_time = time.time()
    try:
        sess = _onnxruntime_session(onnx_path)
        print("  - Session created successfully")
        
        @timeout_decorator.timeout(180, timeout_exception=TimeoutError)
        def run_ort():
            return sess.run(None, {"latent_video": latent.detach().cpu().numpy().astype(np.float32)})
        ort_out = run_ort()
        ort_out_t = torch.from_numpy(ort_out[0]).float()
        print(f"✓ ONNX Runtime forward completed in {time.time() - start_time:.2f}s")
        print(f"  - Output shape: {ort_out_t.shape}")
    except TimeoutError as te:
        print(f"✗ ONNX Runtime forward timed out: {te}")
        raise

    # Step 5: Numerical tolerance check
    print("Step 5: Checking numerical tolerance...")
    max_diff = torch.max(torch.abs(torch_out - ort_out_t)).item()
    mean_diff = torch.mean(torch.abs(torch_out - ort_out_t)).item()
    print(f"  - Max difference: {max_diff:.6f}")
    print(f"  - Mean difference: {mean_diff:.6f}")
    tolerance = 0.02
    assert max_diff < tolerance, f"Max diff {max_diff} exceeds tolerance {tolerance}"
    print("✓ Tolerance check passed")
    
    print("=== test_onnxruntime_inference_and_tolerance completed successfully ===")


@pytest.mark.cuda
@pytest.mark.parametrize("batch_size", [4, 8])
def test_batch_handling(tmp_path: Path, batch_size: int):
    import time
    import logging
    
    print(f"=== Starting test_batch_handling (batch_size={batch_size}) ===")
    
    # Step 1: Load decoder and export
    print("Step 1: Loading decoder and exporting...")
    start_time = time.time()
    try:
        decoder = get_cod_yt_v2_decoder()
        onnx_path = tmp_path / f"video_decoder_bs{batch_size}.onnx"
        export_video_decoder_to_onnx(decoder, onnx_path, batch_size=batch_size)
        print(f"✓ Export completed in {time.time() - start_time:.2f}s")
    except Exception as e:
        print(f"✗ Export failed: {e}")
        raise

    # Step 2: Test batch handling
    print("Step 2: Testing batch handling...")
    start_time = time.time()
    try:
        latent = torch.randn(batch_size, 128, 8, 8, dtype=torch.float32)
        print(f"  - Input shape: {latent.shape}")
        
        sess = _onnxruntime_session(onnx_path)
        print("  - Session created successfully")
        
        ort_out = sess.run(None, {"latent_video": latent.numpy().astype(np.float32)})
        print(f"✓ Batch handling completed in {time.time() - start_time:.2f}s")
        print(f"  - Output shape: {ort_out[0].shape}")
        
        assert isinstance(ort_out, list) and len(ort_out) == 1
        assert ort_out[0].shape[0] == batch_size
        print("✓ Batch size validation passed")
        
    except Exception as e:
        print(f"✗ Batch handling failed: {e}")
        raise
    
    print(f"=== test_batch_handling (batch_size={batch_size}) completed successfully ===")


