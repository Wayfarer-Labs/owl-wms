import torch
import torch.nn as nn
import time
from owl_wms.configs import Config
from owl_wms import from_pretrained
from owl_wms.nn.rope import RoPE

import transformer_engine.pytorch as te
from transformer_engine.pytorch.fp8 import fp8_autocast

# Configuration
N_FRAMES = 1
cfg_path = "configs/dit_v4_dmd.yml"
ckpt_path = "vid_dit_v4_dmd_7k.pt"

def cast_rope_buffers_to_fp32(module):
    """Cast RoPE buffers to fp32 for numerical stability"""
    for submodule in module.modules():
        if isinstance(submodule, RoPE):
            if hasattr(submodule, "cos"):
                submodule.cos = submodule.cos.float()
            if hasattr(submodule, "sin"):
                submodule.sin = submodule.sin.float()

def _swap_linear(module: nn.Module, replace_layernorm: bool = False):
    """
    Recursively replace nn.Linear with te.Linear.
    If replace_layernorm=True, also replace nn.LayerNorm with te.LayerNorm.
    """
    for name, child in list(module.named_children()):
        # Recurse first
        _swap_linear(child, replace_layernorm)

        # Swap Linear
        if isinstance(child, nn.Linear):
            new = te.Linear(
                in_features=child.in_features,
                out_features=child.out_features,
                bias=(child.bias is not None),
            ).to(device=child.weight.device, dtype=child.weight.dtype)

            with torch.no_grad():
                new.weight.copy_(child.weight)
                if child.bias is not None:
                    new.bias.copy_(child.bias)

            setattr(module, name, new)

        # (Optional) Swap LayerNorm – enable only if you want TE’s fused LN
        elif replace_layernorm and isinstance(child, nn.LayerNorm):
            # TE’s LayerNorm takes the hidden size (tuple OK) and eps
            new_ln = te.LayerNorm(
                child.normalized_shape,
                eps=child.eps,
                elementwise_affine=True,
            ).to(device=child.weight.device, dtype=child.weight.dtype)

            with torch.no_grad():
                if child.weight is not None:
                    new_ln.weight.copy_(child.weight)
                if child.bias is not None:
                    new_ln.bias.copy_(child.bias)

            setattr(module, name, new_ln)

def convert_to_te_linears(model: nn.Module, replace_layernorm: bool = False) -> nn.Module:
    model = model  # in-place by default; return for convenience
    _swap_linear(model, replace_layernorm=replace_layernorm)
    return model

@torch.no_grad()
def test_model_forward():
    print(f"Testing model forward pass with {N_FRAMES} frames...")
    
    # Load model
    print("Loading model...")
    model = from_pretrained(cfg_path, ckpt_path, return_decoder=False)
    
    # Convert to bfloat16 first, then replace linear layers
    model = model.core.eval().cuda().bfloat16()
    model.transformer.enable_decoding()
    
    # Replace attn blocks
    n_layers = model.config.n_layers
    config = model.config

    convert_to_te_linears(model)
    
    print("Done")
    # Cast RoPE buffers to fp32
    cast_rope_buffers_to_fp32(model)
    
    # Load config for model dimensions
    cfg = Config.from_yaml(cfg_path)
    
    print("Model loaded successfully!")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")
    
    # Create test inputs
    batch_size = 1
    
    def create_test_inputs():
        """Create randomized test inputs"""
        video = torch.randn(batch_size, N_FRAMES, 128, 8, 8, device='cuda', dtype=torch.bfloat16)
        ts = torch.randn(batch_size, N_FRAMES, device='cuda', dtype=torch.bfloat16)
        mouse = torch.randn(batch_size, N_FRAMES, 2, device='cuda', dtype=torch.bfloat16)
        btn = torch.randn(batch_size, N_FRAMES, 11, device='cuda', dtype=torch.bfloat16)
        return video, ts, mouse, btn
    
    # Create initial inputs to show shapes
    video, ts, mouse, btn = create_test_inputs()
    
    print(f"Input shapes:")
    print(f"  video: {video.shape}")
    print(f"  ts: {ts.shape}")
    print(f"  mouse: {mouse.shape}")
    print(f"  btn: {btn.shape}")
    
    # Check initial VRAM
    torch.cuda.empty_cache()
    initial_memory = torch.cuda.memory_allocated() / 1024**3
    print(f"Initial VRAM usage: {initial_memory:.2f} GB")
    
    # Warmup runs
    print("Running 5 warmup iterations...")
    for i in range(5):
        video, ts, mouse, btn = create_test_inputs()
        output = model(video, ts, mouse, btn)
        print(f"  Warmup {i+1}/5 completed")
    
    # Timing runs with CUDA events
    print("Running 10 timed iterations...")
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    times = []
    for i in range(10):
        video, ts, mouse, btn = create_test_inputs()
        
        torch.cuda.synchronize()
        start_event.record()
        
        with fp8_autocast(enabled=True):
            output = model(video, ts, mouse, btn)
        
        end_event.record()
        torch.cuda.synchronize()
        
        elapsed_time = start_event.elapsed_time(end_event)  # milliseconds
        times.append(elapsed_time)
        print(f"  Iteration {i+1}/10: {elapsed_time:.2f} ms")
    
    # Calculate statistics
    avg_time = sum(times) / len(times)
    min_time = min(times)
    max_time = max(times)
    fps = 1000.0 / avg_time  # Convert ms to fps
    
    peak_memory = torch.cuda.max_memory_allocated() / 1024**3
    current_memory = torch.cuda.memory_allocated() / 1024**3
    
    print(f"\nTiming Results:")
    print(f"  Average time: {avg_time:.2f} ms")
    print(f"  Min time: {min_time:.2f} ms")
    print(f"  Max time: {max_time:.2f} ms")
    print(f"  Average FPS: {fps:.2f}")
    print(f"\nMemory Results:")
    print(f"  Output shape: {output.shape}")
    print(f"  Peak VRAM usage: {peak_memory:.2f} GB")
    print(f"  Current VRAM usage: {current_memory:.2f} GB")
    print(f"  VRAM increase: {current_memory - initial_memory:.2f} GB")

if __name__ == "__main__":
    test_model_forward()
