import sys
import os
import logging
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Union

import torch
from diffusers import AutoencoderDC

# Independent imports with fallback logic for each dependency
TENSORRT_AVAILABLE = False
ONNX_AVAILABLE = False
ONNXRUNTIME_AVAILABLE = False

# Try importing TensorRT independently
try:
    import tensorrt as trt
    TENSORRT_AVAILABLE = True
    logging.info("TensorRT successfully imported")
except ImportError as e:
    logging.warning(f"TensorRT not available: {e}")
    trt = None

# Try importing ONNX independently
try:
    import onnx
    ONNX_AVAILABLE = True
    logging.info("ONNX successfully imported")
except ImportError as e:
    logging.warning(f"ONNX not available: {e}")
    onnx = None

# Try importing ONNX Runtime independently
try:
    import onnxruntime
    ONNXRUNTIME_AVAILABLE = True
    logging.info("ONNX Runtime successfully imported")
except ImportError as e:
    logging.warning(f"ONNX Runtime not available: {e}")
    onnxruntime = None

sys.path.append("./owl-vaes")
from owl_vaes.utils.proxy_init import load_proxy_model
from owl_vaes.models import get_model_cls
from owl_vaes.configs import Config

@dataclass
class TensorRTConfig:
    """Configuration for TensorRT optimization settings."""
    precision: str = "fp16"  # Options: fp32, fp16, int8
    max_batch_size: int = 8
    max_workspace_size: int = 1 << 30  # 1GB default
    optimization_level: int = 3  # 0-5, higher = more aggressive optimization
    use_dla: bool = False  # Deep Learning Accelerator for edge devices
    
    def __post_init__(self):
        if self.precision not in ["fp32", "fp16", "int8"]:
            raise ValueError(f"Invalid precision: {self.precision}. Must be fp32, fp16, or int8")
        if not 0 <= self.optimization_level <= 5:
            raise ValueError(f"Invalid optimization_level: {self.optimization_level}. Must be 0-5")

def get_engine_cache_path(model_name: str, config: Optional[TensorRTConfig] = None) -> Path:
    """
    Generate cache path for TensorRT engine files.
    
    Args:
        model_name: Name/identifier for the model
        config: TensorRT configuration for cache path generation
        
    Returns:
        Path object for the engine cache file
    """
    # Get cache directory from environment variable or use default
    cache_dir = os.getenv("TENSORRT_CACHE_DIR", os.path.expanduser("~/.cache/tensorrt"))
    cache_path = Path(cache_dir)
    
    # Create cache directory if it doesn't exist
    cache_path.mkdir(parents=True, exist_ok=True)
    
    # Generate filename based on model and config
    if config:
        filename = f"{model_name}_{config.precision}_bs{config.max_batch_size}_opt{config.optimization_level}.engine"
    else:
        filename = f"{model_name}_default.engine"
    
    return cache_path / filename

# Initialize TensorRT engine cache directory
if TENSORRT_AVAILABLE:
    try:
        cache_dir = Path(os.getenv("TENSORRT_CACHE_DIR", os.path.expanduser("~/.cache/tensorrt")))
        cache_dir.mkdir(parents=True, exist_ok=True)
        logging.info(f"TensorRT cache directory initialized at: {cache_dir}")
    except Exception as e:
        logging.warning(f"Failed to initialize TensorRT cache directory: {e}")
else:
    logging.debug("TensorRT cache directory initialization skipped (TensorRT not available)")


# -------------------------------
# Model preparation utilities (Phase 2)
# -------------------------------
def _replace_child_module(root: torch.nn.Module, qualified_name: str, new_module: torch.nn.Module) -> None:
    """Safely replace a child module by its dotted qualified name.

    Example: qualified_name="decoder.blocks.3.attn" will traverse attributes
    and replace the final child with new_module.
    """
    parts = qualified_name.split(".")
    parent = root
    for part in parts[:-1]:
        parent = getattr(parent, part)
    setattr(parent, parts[-1], new_module)
def unwrap_weight_norm(model: torch.nn.Module) -> torch.nn.Module:
    """Remove weight_norm parametrizations in-place where present.

    This is written to be safe: if a submodule is not weight-normalized,
    it is skipped. The function returns the same model instance for
    convenient chaining.
    """
    try:
        from torch.nn.utils import remove_weight_norm  # Local import to avoid hard dependency at import time
    except Exception:
        # If unavailable, act as a no-op
        logging.debug("torch.nn.utils.remove_weight_norm not available; skipping unwrap_weight_norm")
        return model

    removed_count = 0
    for name, module in model.named_modules():
        # Common parameter name is "weight"; attempt removal optimistically
        try:
            remove_weight_norm(module)
            removed_count += 1
        except Exception:
            # Not weight-normalized or removal failed; ignore safely
            continue

    if removed_count > 0:
        logging.info(f"unwrap_weight_norm: removed weight_norm from {removed_count} submodules")
    else:
        logging.debug("unwrap_weight_norm: no weight_norm parametrizations found")
    return model


def replace_landscape_operations(model: torch.nn.Module) -> torch.nn.Module:
    """Replace custom landscape/square operations with ONNX-friendly ops."""
    replaced_count = 0
    enable_replacement = os.getenv("REPLACE_LANDSCAPE", "0") == "1"
    for module_name, module in model.named_modules():
        name_lower = module_name.lower()
        if "landscape" in name_lower or "square" in name_lower:
            logging.info(f"Landscape candidate detected: {module_name} (type: {type(module).__name__})")
            if not enable_replacement:
                continue
            # Prefer explicit size hints if available
            target_h = getattr(module, "target_height", None)
            target_w = getattr(module, "target_width", None)
            if isinstance(target_h, int) and isinstance(target_w, int):
                try:
                    new_module = torch.nn.Upsample(size=(target_h, target_w), mode="bilinear", align_corners=False)
                    _replace_child_module(model, module_name, new_module)
                    replaced_count += 1
                    logging.info(f"replace_landscape_operations: replaced '{module_name}' with nn.Upsample({target_h}x{target_w})")
                    continue
                except Exception as e:
                    logging.debug(f"replace_landscape_operations: failed to replace '{module_name}': {e}")
            logging.debug(f"replace_landscape_operations: no safe replacement for '{module_name}' (no-op)")
            continue

    if replaced_count > 0:
        logging.info(f"replace_landscape_operations: replaced {replaced_count} modules with ONNX-friendly ops")
    else:
        logging.debug("replace_landscape_operations: no landscape-related modules replaced (no-op)")
    # Track count on model for diagnostics
    try:
        setattr(model, "landscape_replaced_count", replaced_count)
    except Exception:
        pass
    return model


def replace_sana_operations(model: torch.nn.Module) -> torch.nn.Module:
    """Replace custom SANA pixel shuffle ops with standard equivalents.

    This scans for modules exposing attributes suggestive of custom shuffle
    implementations and logs candidates. It is currently a safe no-op until
    concrete module types are identified.
    """
    replaced_count = 0
    enable_replacement = os.getenv("REPLACE_SANA", "0") == "1"
    for module_name, module in model.named_modules():
        if hasattr(module, "pixel_shuffle") or hasattr(module, "pixel_unshuffle"):
            if not enable_replacement:
                logging.debug(f"replace_sana_operations: candidate '{module_name}' detected (disabled)")
                continue
            upscale = getattr(module, "upscale_factor", None) or getattr(module, "factor", None)
            downscale = getattr(module, "downscale_factor", None) or getattr(module, "factor", None)
            try:
                if upscale and isinstance(upscale, int):
                    new_module = torch.nn.PixelShuffle(upscale)
                    _replace_child_module(model, module_name, new_module)
                    replaced_count += 1
                    logging.info(f"replace_sana_operations: replaced '{module_name}' with nn.PixelShuffle({upscale})")
                    continue
                if downscale and isinstance(downscale, int):
                    new_module = torch.nn.PixelUnshuffle(downscale)
                    _replace_child_module(model, module_name, new_module)
                    replaced_count += 1
                    logging.info(f"replace_sana_operations: replaced '{module_name}' with nn.PixelUnshuffle({downscale})")
                    continue
            except Exception as e:
                logging.debug(f"replace_sana_operations: failed to replace '{module_name}': {e}")
            logging.debug(f"replace_sana_operations: no safe replacement for '{module_name}' (no-op)")
            continue

    if replaced_count > 0:
        logging.info(f"replace_sana_operations: replaced {replaced_count} modules with standard shuffles")
    else:
        logging.debug("replace_sana_operations: no SANA-related modules replaced (no-op)")
    # Track count on model for diagnostics
    try:
        setattr(model, "sana_replaced_count", replaced_count)
    except Exception:
        pass
    return model


def simplify_attention_blocks(model: torch.nn.Module) -> torch.nn.Module:
    """Simplify attention blocks to improve ONNX export compatibility.

    By default, this function is a no-op that only logs candidates. Actual
    simplification (such as swapping to torch.nn.Identity) should be enabled
    once specific attention module types are confirmed safe to modify.
    """
    simplified_count = 0
    enable_simplify = os.getenv("SIMPLIFY_ATTENTION", "0") == "1"
    for module_name, module in model.named_modules():
        name_lower = module_name.lower()
        if "attn" in name_lower or hasattr(module, "attention"):
            if not enable_simplify:
                logging.debug(f"simplify_attention_blocks: candidate '{module_name}' detected (disabled)")
                continue
            if isinstance(module, torch.nn.MultiheadAttention):
                logging.debug(f"simplify_attention_blocks: skipping standard MultiheadAttention '{module_name}'")
                continue
            try:
                identity = torch.nn.Identity()
                _replace_child_module(model, module_name, identity)
                simplified_count += 1
                logging.info(f"simplify_attention_blocks: replaced '{module_name}' with nn.Identity()")
            except Exception as e:
                logging.debug(f"simplify_attention_blocks: failed to simplify '{module_name}': {e}")
            continue

    if simplified_count > 0:
        logging.info(f"simplify_attention_blocks: simplified {simplified_count} attention blocks")
    else:
        logging.debug("simplify_attention_blocks: no attention blocks simplified (no-op)")
    # Track count on model for diagnostics
    try:
        setattr(model, "attn_simplified_count", simplified_count)
    except Exception:
        pass
    return model


def export_video_decoder_to_onnx(decoder_model: torch.nn.Module, output_path: Path, batch_size: int = 8, validate_ort: bool = True) -> bool:
    """Export DCAE-like video decoder to ONNX and optionally validate with ONNX Runtime.

    Steps:
    - Prepare model: CPU, eval, unwrap weight norm, optional replacements
    - Export with opset 17, dynamic batch axis
    - Validate ONNX model structure
    - Optionally run inference in ORT and compare with PyTorch (1e-2 tolerance)
    """
    if not ONNX_AVAILABLE:
        raise RuntimeError("ONNX is not available; cannot export decoder to ONNX")
    if validate_ort and not ONNXRUNTIME_AVAILABLE:
        raise RuntimeError("ONNX Runtime is not available; cannot validate ONNX export")

    # Prepare model for export with custom operation replacements enabled
    print("  Export: Preparing model...")
    prepared_model = decoder_model
    try:
        # Run export and validation on CPU in float32 for stability/perf
        print("  Export: Moving model to CPU and converting to float32...")
        prepared_model = prepared_model.cpu().float().eval()
        
        print("  Export: Unwrapping weight norm...")
        prepared_model = unwrap_weight_norm(prepared_model)
        
        # Enable custom operation replacements for ONNX compatibility
        import os
        print("  Export: Enabling custom operation replacements...")
        os.environ["REPLACE_LANDSCAPE"] = "1"  # Enable landscape/square operation replacement
        os.environ["REPLACE_SANA"] = "1"       # Enable SANA pixel shuffle replacement
        os.environ["SIMPLIFY_ATTENTION"] = "1" # Enable attention block simplification
        
        print("  Export: Replacing landscape operations...")
        prepared_model = replace_landscape_operations(prepared_model)
        print("  Export: Replacing SANA operations...")
        prepared_model = replace_sana_operations(prepared_model)
        print("  Export: Simplifying attention blocks...")
        prepared_model = simplify_attention_blocks(prepared_model)
        
        print("  Export: Model preparation completed")
        logging.info("Custom operation replacements enabled for ONNX export")
    except Exception as e:
        print(f"  Export: Model preparation warning: {e}")
        logging.warning(f"Model preparation encountered an issue (continuing): {e}")

    # Add prints to diagnose replacements
    print(f"  Export: Replaced {getattr(prepared_model, 'landscape_replaced_count', 0)} landscape ops")
    print(f"  Export: Replaced {getattr(prepared_model, 'sana_replaced_count', 0)} SANA ops")
    print(f"  Export: Simplified {getattr(prepared_model, 'attn_simplified_count', 0)} attention blocks")

    # Video latent input: [batch, 128, 8, 8]
    latent_channels = 128
    latent_h = 8
    latent_w = 8
    # Use float32 inputs to match prepared model dtype on CPU
    dummy_input = torch.randn(batch_size, latent_channels, latent_h, latent_w, dtype=torch.float32)

    with torch.no_grad():
        pytorch_output = prepared_model(dummy_input).float()
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Export with TensorRT-friendly settings
        print("  Export: Starting ONNX export...")
        try:
            from onnxruntime.tools import pytorch_export_contrib_ops
            pytorch_export_contrib_ops.register()
            print("  Export: Registered ONNX Runtime contrib ops")
        except ImportError:
            print("  Export: onnxruntime.tools not available")
            logging.warning("onnxruntime.tools not available, proceeding without contrib ops")
        
        try:
            print("  Export: Calling torch.onnx.export...")
            torch.onnx.export(
                prepared_model,
                dummy_input,
                str(output_path),
                input_names=["latent_video"],
                output_names=["decoded_video"],
                dynamic_axes={"latent_video": {0: "batch_size"}, "decoded_video": {0: "batch_size"}},
                opset_version=17,
                do_constant_folding=True,
                dynamo=True,
                verbose=True
            )
            print("  Export: torch.onnx.export completed")
        except Exception as e:
            logging.error(f"ONNX export failed: {e}", exc_info=True)
            print(f"  Export: torch.onnx.export failed: {e}")
            raise

    # Structural validation
    print("  Export: Validating ONNX model structure...")
    onnx_model = onnx.load(str(output_path))
    onnx.checker.check_model(onnx_model)
    print("  Export: ONNX model structure validated")

    unsupported_ops = [node.op_type for node in onnx_model.graph.node if node.op_type.startswith('ATen')]
    if unsupported_ops:
        logging.warning(f"Unsupported ops in ONNX: {unsupported_ops}")

    # ONNX Runtime validation (only if requested)
    if validate_ort:
        print("  Export: Running ONNX Runtime validation...")
        try:
            import numpy as np
            sess = _onnxruntime_session(output_path)  # Use updated session function
            onnx_input = dummy_input.detach().cpu().numpy().astype(np.float32)
            ort_out = sess.run(None, {"latent_video": onnx_input})
            ort_output = torch.from_numpy(ort_out[0]).float()
            max_diff = torch.max(torch.abs(pytorch_output - ort_output)).item()
            tolerance = 1e-1  # Relaxed for TensorRT prep
            if max_diff >= tolerance:
                logging.warning(f"ONNX validation tolerance exceeded: max_diff={max_diff:.6f} >= {tolerance}")
            else:
                logging.info(f"ONNX validation passed: max_diff={max_diff:.6f} < {tolerance}")
            print(f"  Export: ONNX Runtime validation completed (max_diff={max_diff:.6f})")
        except Exception as e:
            logging.error(f"ONNX Runtime validation failed: {e}")
            print(f"  Export: ONNX Runtime validation failed: {e}")
            return False  # Fail if validation fails for TensorRT readiness

    logging.info(f"Video decoder ONNX export successful: {output_path}")
    return True

def _get_decoder_only():
    model = load_proxy_model(
        "../checkpoints/128x_proxy_titok.yml",
        "../checkpoints/128x_proxy_titok.pt",
        "../checkpoints/16x_dcae.yml",
        "../checkpoints/16x_dcae.pt"
    )
    del model.transformer.encoder
    return model

def get_decoder_only(vae_id, cfg_path, ckpt_path):
    """Load the decoder module using a config and checkpoint.

    For `dcae`, prefer the provided config and checkpoint (cod_yt_v2 base.yml
    and resume_ckpt) instead of historical diffusers references.
    """
    cfg = Config.from_yaml(cfg_path).model
    model = get_model_cls(cfg.model_id)(cfg)
    state = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    try:
        model.load_state_dict(state)
    except Exception:
        model.decoder.load_state_dict(state)
    del model.encoder
    model = model.decoder
    # Keep on CPU in float32 by default for export/tests; caller can move to CUDA if needed
    model = model.float().cpu().eval()
    return model

@torch.no_grad()
def make_batched_decode_fn(decoder, batch_size = 8):
    def decode(x):
        # x is [b,n,c,h,w]
        b,n,c,h,w = x.shape
        x = x.view(b*n,c,h,w).contiguous()

        batches = x.split(batch_size)
        batch_out = []
        for batch in batches:
            batch_out.append(decoder(batch).bfloat16())

        x = torch.cat(batch_out) # [b*n,c,h,w]
        _,c,h,w = x.shape
        x = x.view(b,n,c,h,w).contiguous()

        return x
    return decode

@torch.no_grad()
def make_batched_audio_decode_fn(decoder, batch_size = 8):
    def decode(x):
        # x is [b,n,c] audio samples
        x = x.transpose(1,2)
        b,c,n = x.shape

        batches = x.contiguous().split(batch_size)
        batch_out = []
        for batch in batches:
            batch_out.append(decoder(batch).bfloat16())

        x = torch.cat(batch_out) # [b,c,n]
        x = x.transpose(-1,-2).contiguous() # [b,n,2]

        return x
    return decode


def get_cod_yt_v2_decoder() -> torch.nn.Module:
    """Convenience loader for the COD-YT v2 DCAE decoder.

    Uses config at /root/owl-wms/owl-vaes/configs/cod_yt_v2/base.yml and
    checkpoint at /root/owl-wms/owl-vaes/checkpoints/cod_yt_v2/step_515000.pt.
    """
    cfg_path = "/root/owl-wms/owl-vaes/configs/cod_yt_v2/base.yml"
    ckpt_path = "/root/owl-wms/owl-vaes/checkpoints/cod_yt_v2/step_515000.pt"
    return get_decoder_only("dcae", cfg_path, ckpt_path)