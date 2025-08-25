import sys
import os
import logging
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Union

import torch
from diffusers import AutoencoderDC

# TensorRT imports with fallback logic
TENSORRT_AVAILABLE = False
try:
    import tensorrt as trt
    import onnx
    import onnxruntime
    TENSORRT_AVAILABLE = True
    logging.info("TensorRT dependencies successfully imported")
except ImportError as e:
    logging.warning(f"TensorRT dependencies not available, falling back to PyTorch: {e}")
    # Set placeholder variables to avoid NameError
    trt = None
    onnx = None
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

# Initialize engine cache directory
if TENSORRT_AVAILABLE:
    try:
        cache_dir = Path(os.getenv("TENSORRT_CACHE_DIR", os.path.expanduser("~/.cache/tensorrt")))
        cache_dir.mkdir(parents=True, exist_ok=True)
        logging.info(f"TensorRT cache directory initialized at: {cache_dir}")
    except Exception as e:
        logging.warning(f"Failed to initialize TensorRT cache directory: {e}")

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
        if vae_id == "dcae":
            model_id = "mit-han-lab/dc-ae-f64c128-mix-1.0-diffusers"
            model = AutoencoderDC.from_pretrained(model_id).bfloat16().cuda().eval()
            del model.encoder
            return model.decoder
        else:
            cfg = Config.from_yaml(cfg_path).model
            model = get_model_cls(cfg.model_id)(cfg)
            try:
                model.load_state_dict(torch.load(ckpt_path, map_location='cpu',weights_only=False))
            except:
                model.decoder.load_state_dict(torch.load(ckpt_path, map_location='cpu',weights_only=False))
            del model.encoder
            model = model.decoder
            model = model.bfloat16().cuda().eval()
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