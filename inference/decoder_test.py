import torch
from torch import nn
from owl_wms import from_pretrained
import gc


import torch, torch_tensorrt
from torch_tensorrt.dynamo import compile

cfg_path = "configs/dit_v4_dmd.yml"
ckpt_path = "vid_dit_v4_dmd_7k.pt"

_, decoder = from_pretrained(cfg_path, ckpt_path, return_decoder=True)
decoder = decoder.eval().cuda().bfloat16()
#decoder = torch.compile(decoder, mode = 'max-autotune', fullgraph = True, dynamic = False)
decoder = 
# Clear cache 
torch.cuda.empty_cache()
gc.collect()

# Configuration
BATCH_SIZE = 1

@torch.no_grad()
def test_decoder():
    print(f"Testing decoder with batch size {BATCH_SIZE}...")
    
    print("Decoder loaded successfully!")
    print(f"Decoder parameters: {sum(p.numel() for p in decoder.parameters()) / 1e6:.1f}M")
    
    def create_test_inputs():
        """Create randomized test inputs"""
        x = torch.randn(BATCH_SIZE, 128, 8, 8, device='cuda', dtype=torch.bfloat16)
        return x
    
    # Create initial inputs to show shapes
    x = create_test_inputs()
    
    print(f"Input shapes:")
    print(f"  x: {x.shape}")
    
    # Check initial VRAM
    torch.cuda.empty_cache()
    initial_memory = torch.cuda.memory_allocated() / 1024**3
    print(f"Initial VRAM usage: {initial_memory:.2f} GB")
    
    # Warmup runs
    print("Running 5 warmup iterations...")
    for i in range(5):
        x = create_test_inputs()
        output = decoder(x)
        print(f"  Warmup {i+1}/5 completed")
    
    # Timing runs with CUDA events
    print("Running 10 timed iterations...")
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    times = []
    for i in range(10):
        x = create_test_inputs()
        
        torch.cuda.synchronize()
        start_event.record()
        
        output = decoder(x)
        
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
    test_decoder()