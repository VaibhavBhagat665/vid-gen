"""
Generate videos using your fine-tuned model

Usage:
    python generate_finetuned.py --checkpoint ./checkpoints/checkpoint-epoch-30 --prompt "a person playing basketball"
"""

import argparse
import torch
from diffusers import DiffusionPipeline


def generate_video(checkpoint_path, prompt, output_path="output.mp4", num_frames=16, width=320, height=320):
    """Generate video from fine-tuned model."""
    
    print("=" * 60)
    print("Loading Fine-Tuned Model")
    print("=" * 60)
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Prompt: {prompt}")
    print()
    
    # Load your fine-tuned model
    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    
    pipeline = DiffusionPipeline.from_pretrained(
        checkpoint_path,
        torch_dtype=torch.float16 if device != "cpu" else torch.float32,
    ).to(device)
    
    print("✓ Model loaded successfully")
    print()
    
    # Generate video
    print("Generating video...")
    print(f"Resolution: {width}x{height}")
    print(f"Frames: {num_frames}")
    print()
    
    video_frames = pipeline(
        prompt=prompt,
        num_frames=num_frames,
        height=height,
        width=width,
        num_inference_steps=50,
    ).frames[0]
    
    # Save video
    from diffusers.utils import export_to_video
    export_to_video(video_frames, output_path, fps=8)
    
    print("=" * 60)
    print(f"✓ Video saved to: {output_path}")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate video from fine-tuned model")
    parser.add_argument("--checkpoint", type=str, required=True,
                       help="Path to checkpoint directory (e.g., ./checkpoints/checkpoint-epoch-30)")
    parser.add_argument("--prompt", type=str, required=True,
                       help="Text prompt for video generation")
    parser.add_argument("--output", type=str, default="output.mp4",
                       help="Output video path")
    parser.add_argument("--num_frames", type=int, default=16,
                       help="Number of frames to generate")
    parser.add_argument("--width", type=int, default=320,
                       help="Video width")
    parser.add_argument("--height", type=int, default=320,
                       help="Video height")
    
    args = parser.parse_args()
    
    generate_video(
        checkpoint_path=args.checkpoint,
        prompt=args.prompt,
        output_path=args.output,
        num_frames=args.num_frames,
        width=args.width,
        height=args.height
    )
