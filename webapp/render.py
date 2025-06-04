import torch
from pathlib import Path

from webapp.models import load_model
from webapp.samplers import create_sampler
from webapp.action_builder import ActionSequenceGenerator, ActionConfig, ActionPattern

# Global configuration - easy to modify
SEQUENCE_LENGTH = 128
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
OUTPUT_DIR = "generated_videos"
SAMPLER_TYPE = 'cfg'
DEFAULT_PATTERN = ActionPattern.LOOK_AROUND


def setup_output_dir():
    """Create output directory."""
    Path(OUTPUT_DIR).mkdir(exist_ok=True)


def generate_dummy_actions(pattern=DEFAULT_PATTERN, length=SEQUENCE_LENGTH):
    """Generate dummy actions for video conditioning."""
    config = ActionConfig(
        sequence_length=length,
        device=DEVICE,
        dtype=torch.float32
    )
    
    generator = ActionSequenceGenerator(config)
    mouse, buttons = generator.generate_pattern(pattern)
    
    # Add batch dimension
    return mouse.unsqueeze(0), buttons.unsqueeze(0)


def synthesize_video(mouse_actions, button_actions, model, sampler):
    """Generate video using model and sampler."""
    batch_size, sequence_length = mouse_actions.shape[:2]
    
    # Create dummy latent batch - hardcoded dimensions from config
    dummy_batch = torch.randn(
        batch_size, sequence_length, 128, 16, 16,
        device=DEVICE, dtype=torch.float32
    )
    
    # Ensure actions are on correct device
    mouse_actions = mouse_actions.to(DEVICE)
    button_actions = button_actions.to(DEVICE)
    
    # Generate video
    with torch.no_grad():
        video = sampler(
            dummy_batch=dummy_batch,
            mouse=mouse_actions,
            btn=button_actions
        )
    
    return video


def save_video(video_tensor, filename="generated_video"):
    """Save video tensor to file."""
    setup_output_dir()
    import time
    timestamp = int(time.time())
    path = Path(OUTPUT_DIR) / f"{filename}_{timestamp}.pt"
    torch.save(video_tensor.cpu(), path)
    return str(path)


def render_video(pattern=DEFAULT_PATTERN, length=SEQUENCE_LENGTH, verbose=True):
    """Simple video rendering - just load, generate, save."""
    if verbose:
        print(f"🎬 Rendering video with pattern: {pattern.value}")
    
    # Load model and create sampler
    if verbose:
        print("Loading model...")
    model = load_model(device=DEVICE, verbose=verbose)
    
    if verbose:
        print("Creating sampler...")
    sampler = create_sampler(SAMPLER_TYPE, model)
    
    # Generate actions
    if verbose:
        print("Generating actions...")
    mouse, buttons = generate_dummy_actions(pattern, length)
    
    # Synthesize video
    if verbose:
        print("Synthesizing video...")
    video = synthesize_video(mouse, buttons, model, sampler)
    
    # Save video
    if verbose:
        print("Saving video...")
    path = save_video(video, f"render_{pattern.value}")
    
    if verbose:
        print(f"✅ Done! Video saved to: {path}")
        print(f"   Video shape: {video.shape}")
    
    return path


if __name__ == "__main__":
    # Simple usage examples
    print("🎮 Simple OWL-WMS Video Renderer")
    
    # Render with default settings
    render_video()
    
    # Render with different pattern
    render_video(ActionPattern.AIM_AND_SHOOT, length=64)


