"""
Speed test for model forward pass with KV caching.
Tests single frame forward pass with a pre-filled KV cache.
"""
import argparse
import torch
import yaml
from omegaconf import OmegaConf
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from owl_wms.models.world import WorldModel
from owl_wms.nn.kv_cache import StaticKVCache
from inference.utils.timing import time_function
from owl_wms.nn.rope import cast_rope_buffers_to_fp32


def load_config(config_path: str):
    """Load config from yaml file."""
    with open(config_path, 'r') as f:
        cfg_dict = yaml.safe_load(f)

    # Convert to OmegaConf for easier attribute access
    cfg = OmegaConf.create(cfg_dict)
    return cfg


def create_dummy_inputs(config, n_frames: int, batch_size: int = 1, device: str = 'cuda'):
    """
    Create randomized dummy inputs for the model.

    Args:
        config: Model configuration
        n_frames: Number of frames
        batch_size: Batch size for inputs
        device: Device to create tensors on

    Returns:
        Dictionary of model inputs
    """
    # Get dimensions from config
    channels = config.model.channels
    height = config.model.height
    width = config.model.width
    n_controller_inputs = config.model.n_controller_inputs

    # Create dummy inputs
    # x: [B, N, C, H, W]
    x = torch.randn(batch_size, n_frames, channels, height, width,
                    device=device, dtype=torch.bfloat16)

    # sigma: [B, N] - noise level
    sigma = torch.rand(batch_size, n_frames, device=device, dtype=torch.bfloat16)

    # frame_timestamp: [B, N]
    frame_timestamp = torch.arange(n_frames, device=device, dtype=torch.long).repeat(batch_size, 1)

    # controller_inputs: [B, N, I]
    controller_inputs = torch.randn(batch_size, n_frames, n_controller_inputs,
                                   device=device, dtype=torch.bfloat16)

    return {
        'x': x,
        'sigma': sigma,
        'frame_timestamp': frame_timestamp,
        'controller_inputs': controller_inputs
    }


def run_speed_test_with_cache(
    config_path: str,
    batch_size: int = 1,
    n_cache_frames: int = 60,
    n_warmup: int = 10,
    n_eval: int = 100,
    device: str = 'cuda'
):
    """
    Run speed test on model forward pass with KV caching.

    Args:
        config_path: Path to model config yaml
        batch_size: Batch size for inference
        n_cache_frames: Number of frames to test with (cache will have n-1 frames)
        n_warmup: Number of warmup iterations
        n_eval: Number of evaluation iterations
        device: Device to run on
    """
    print(f"Loading config from: {config_path}")
    cfg = load_config(config_path)

    print(f"Creating model...")
    model = WorldModel(cfg.model).to(device).eval().bfloat16()
    cast_rope_buffers_to_fp32(model)

    # Count parameters
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model has {n_params:,} parameters")

    print(f"\nModel configuration:")
    print(f"  - d_model: {cfg.model.d_model}")
    print(f"  - n_layers: {cfg.model.n_layers}")
    print(f"  - n_heads: {cfg.model.n_heads}")
    print(f"  - Input shape: {batch_size} x 1 x {cfg.model.channels} x {cfg.model.height} x {cfg.model.width}")
    print(f"\nKV Cache setup:")
    print(f"  - Total frames to test: {n_cache_frames}")
    print(f"  - Cache frames (history): {n_cache_frames - 1}")
    print(f"  - New frame to process: 1")

    # Create KV cache
    print(f"\nCreating and pre-filling KV cache...")
    kv_cache = StaticKVCache(
        cfg.model,
        max_seq_len=n_cache_frames,
        batch_size=batch_size,
        dtype=torch.bfloat16
    ).to(device)

    # Pre-fill cache with n_cache_frames - 1 frames
    n_history_frames = n_cache_frames - 1
    history_inputs = create_dummy_inputs(cfg, n_frames=n_history_frames, batch_size=batch_size, device=device)

    # Run a forward pass to populate the cache
    with torch.inference_mode():
        _ = model(
            x=history_inputs['x'],
            sigma=history_inputs['sigma'],
            frame_timestamp=history_inputs['frame_timestamp'],
            controller_inputs=history_inputs['controller_inputs'],
            kv_cache=kv_cache
        )

    print(f"  - Cache populated with {n_history_frames} frames")
    print(f"  - KV cache offset: {kv_cache.kv_offset[0].item()} tokens")

    # Create inputs for single new frame
    # Frame timestamp should continue from where history left off
    new_frame_inputs = create_dummy_inputs(cfg, n_frames=1, batch_size=batch_size, device=device)
    new_frame_inputs['frame_timestamp'] = torch.tensor([[n_history_frames]], device=device, dtype=torch.long)

    model = torch.compile(model)

    # Define forward pass function with cache
    @torch.inference_mode()
    def forward_pass_cached(x, sigma, frame_timestamp, controller_inputs):
        return model(
            x=x,
            sigma=sigma,
            frame_timestamp=frame_timestamp,
            controller_inputs=controller_inputs,
            kv_cache=kv_cache
        )

    # Run timing
    print(f"\nRunning speed test:")
    print(f"  - Warmup iterations: {n_warmup}")
    print(f"  - Evaluation iterations: {n_eval}")

    min_lat, max_lat, avg_lat, min_fps, max_fps, avg_fps = time_function(
        forward_pass_cached,
        new_frame_inputs['x'],
        new_frame_inputs['sigma'],
        new_frame_inputs['frame_timestamp'],
        new_frame_inputs['controller_inputs'],
        n_warmup_calls=n_warmup,
        n_eval_calls=n_eval
    )

    # Print results
    print(f"\n{'='*60}")
    print(f"RESULTS (with KV caching, {n_history_frames} cached frames):")
    print(f"{'='*60}")
    print(f"\nLatency (ms):")
    print(f"  - Min:     {min_lat:.2f} ms")
    print(f"  - Max:     {max_lat:.2f} ms")
    print(f"  - Average: {avg_lat:.2f} ms")

    print(f"\nThroughput (FPS):")
    print(f"  - Min:     {min_fps:.2f} fps")
    print(f"  - Max:     {max_fps:.2f} fps")
    print(f"  - Average: {avg_fps:.2f} fps")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(
        description='Speed test for model forward pass with KV caching'
    )
    parser.add_argument(
        '--config',
        type=str,
        help='Path to model config yaml file',
        default='configs/waypoint/pilot_short_attn.yml'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=1,
        help='Batch size for inference (default: 1)'
    )
    parser.add_argument(
        '--n-cache-frames',
        type=int,
        default=60,
        help='Number of frames to test with - cache will have n-1 frames (default: 60)'
    )
    parser.add_argument(
        '--n-warmup',
        type=int,
        default=10,
        help='Number of warmup iterations (default: 10)'
    )
    parser.add_argument(
        '--n-eval',
        type=int,
        default=100,
        help='Number of evaluation iterations (default: 100)'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        help='Device to run on (default: cuda)'
    )

    args = parser.parse_args()

    # Verify config file exists
    if not Path(args.config).exists():
        print(f"Error: Config file not found at {args.config}")
        sys.exit(1)

    # Run speed test
    run_speed_test_with_cache(
        config_path=args.config,
        batch_size=args.batch_size,
        n_cache_frames=args.n_cache_frames,
        n_warmup=args.n_warmup,
        n_eval=args.n_eval,
        device=args.device
    )


if __name__ == '__main__':
    main()
