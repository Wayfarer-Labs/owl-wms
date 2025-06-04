#!/usr/bin/env python3
"""
Demo script for OWL-WMS video generation pipeline.

This script demonstrates how to use the complete pipeline to generate videos
with different action patterns and configurations.
"""

import argparse
import sys
from pathlib import Path

# Add webapp to path
sys.path.append(str(Path(__file__).parent))

from webapp.render import render_video, save_video
from webapp.action_builder import ActionPattern, ActionSequenceGenerator, ActionConfig, MouseGenerator, ButtonGenerator


def demo_quick_renders():
    """Demonstrate quick rendering with different patterns."""
    print("🎮 Demo: Quick Renders")
    print("-" * 30)
    
    patterns_to_try = [
        ActionPattern.LOOK_AROUND,
        ActionPattern.AIM_AND_SHOOT,
        ActionPattern.SPRINT_FORWARD,
        ActionPattern.CIRCLE_STRAFE
    ]
    
    for pattern in patterns_to_try:
        print(f"Generating video with pattern: {pattern.value}")
        result = quick_render(
            pattern=pattern,
            sequence_length=60,
            output_dir=f"demo_output/{pattern.value}"
        )
        print(f"✅ Generated: {result['video_path']}")
        print(f"   Render time: {result['render_time']:.2f}s")
        print()


def demo_batch_render():
    """Demonstrate batch rendering with multiple patterns."""
    print("🎮 Demo: Batch Render")
    print("-" * 30)
    
    patterns = [
        ActionPattern.WALK_FORWARD,
        ActionPattern.STRAFE_LEFT,
        ActionPattern.STRAFE_RIGHT,
        ActionPattern.WALK_BACKWARD
    ]
    
    print(f"Generating batch video with {len(patterns)} different patterns...")
    result = batch_render(
        patterns=patterns,
        sequence_length=60,
        output_dir="demo_output/batch"
    )
    
    print(f"✅ Generated batch: {result['video_path']}")
    print(f"   Video shape: {result['video_shape']}")
    print(f"   Render time: {result['render_time']:.2f}s")
    print()


def demo_custom_actions():
    """Demonstrate custom action sequence creation."""
    print("🎮 Demo: Custom Action Sequences")
    print("-" * 30)
    
    # Create a complex custom sequence
    config = ActionConfig(sequence_length=120, random_seed=42)
    generator = ActionSequenceGenerator(config)
    
    builder = generator.generate_custom_sequence()
    
    # Build a complex sequence: idle -> look around -> aim and move -> sprint
    mouse_sequence, button_sequence = (builder
        .add_mouse_segment(0, 30, MouseGenerator.idle)
        .add_button_segment(0, 30, ButtonGenerator.idle)
        
        .add_mouse_segment(30, 60, MouseGenerator.look_around, speed=0.4, amplitude=0.6)
        .add_button_segment(30, 60, ButtonGenerator.idle)
        
        .add_mouse_segment(60, 90, MouseGenerator.aim_tracking, target_speed=0.2)
        .add_button_segment(60, 90, ButtonGenerator.hold_buttons, button_names=["W", "A"])
        
        .add_mouse_segment(90, 120, MouseGenerator.look_around, speed=0.6)
        .add_button_segment(90, 120, ButtonGenerator.hold_buttons, button_names=["W", "LSHIFT"])
        .build())
    
    # Use these custom actions with the renderer
    render_config = RenderConfig(
        sequence_length=120,
        action_pattern=ActionPattern.IDLE,  # This will be overridden
        random_seed=42,
        output_dir="demo_output/custom"
    )
    
    renderer = VideoRenderer(render_config)
    
    # Manually set the actions and render
    renderer.load_model()
    renderer.create_sampler()
    
    print("Generating video with custom action sequence...")
    generated_video = renderer.synthesize_video(
        mouse_sequence.unsqueeze(0),  # Add batch dimension
        button_sequence.unsqueeze(0)
    )
    
    # Save the results
    action_paths = renderer.save_actions(
        mouse_sequence.unsqueeze(0), 
        button_sequence.unsqueeze(0), 
        "custom_demo"
    )
    video_path = renderer.save_video(generated_video, "custom_demo")
    
    print(f"✅ Generated custom video: {video_path}")
    print(f"   Action files: {action_paths}")
    print()


def demo_action_patterns():
    """Demonstrate all available action patterns."""
    print("🎮 Demo: All Action Patterns")
    print("-" * 30)
    
    config = ActionConfig(sequence_length=60)
    generator = ActionSequenceGenerator(config)
    
    print("Available action patterns:")
    for pattern in ActionPattern:
        print(f"  - {pattern.value}")
        
        # Generate and show stats for each pattern
        mouse, buttons = generator.generate_pattern(pattern)
        mouse_range = f"[{mouse.min().item():.3f}, {mouse.max().item():.3f}]"
        button_count = buttons.sum().item()
        
        print(f"    Mouse range: {mouse_range}, Button presses: {button_count:.0f}")
    
    print()


def demo_with_args():
    """Demo with command line arguments."""
    parser = argparse.ArgumentParser(description="OWL-WMS Video Generation Demo")
    parser.add_argument("--pattern", type=str, default="look_around",
                       choices=[p.value for p in ActionPattern],
                       help="Action pattern to use")
    parser.add_argument("--length", type=int, default=60,
                       help="Sequence length")
    parser.add_argument("--output", type=str, default="demo_output",
                       help="Output directory")
    parser.add_argument("--device", type=str, default="auto",
                       help="Device to use (cuda/cpu/auto)")
    parser.add_argument("--cfg-scale", type=float, default=1.3,
                       help="CFG scale for sampling")
    parser.add_argument("--seed", type=int, default=None,
                       help="Random seed")
    
    args = parser.parse_args()
    
    # Find the pattern enum
    pattern = None
    for p in ActionPattern:
        if p.value == args.pattern:
            pattern = p
            break
    
    if pattern is None:
        print(f"❌ Unknown pattern: {args.pattern}")
        return
    
    print(f"🎮 Custom Demo: {pattern.value}")
    print("-" * 30)
    
    config = RenderConfig(
        sequence_length=args.length,
        action_pattern=pattern,
        device=args.device,
        output_dir=args.output,
        cfg_scale=args.cfg_scale,
        random_seed=args.seed
    )
    
    renderer = VideoRenderer(config)
    result = renderer.render_video(filename_prefix=f"demo_{pattern.value}")
    
    print(f"✅ Generated: {result['video_path']}")
    print(f"   Shape: {result['video_shape']}")
    print(f"   Time: {result['render_time']:.2f}s")


def main():
    """Run all demos or handle command line arguments."""
    if len(sys.argv) > 1:
        # Handle command line arguments
        demo_with_args()
        return
    
    print("🎬 OWL-WMS Video Generation Demo Suite")
    print("=" * 50)
    print()
    
    try:
        # Run all demos
        demo_action_patterns()
        demo_quick_renders()
        demo_batch_render()
        demo_custom_actions()
        
        print("🎉 All demos completed successfully!")
        print("📁 Check the demo_output/ directory for generated videos.")
        print()
        print("💡 Tip: Run with command line arguments for custom generation:")
        print("   python demo.py --pattern aim_and_shoot --length 128 --seed 42")
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 