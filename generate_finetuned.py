"""
Generate videos using your fine-tuned model

Usage:
    python generate_finetuned.py --checkpoint ./checkpoints/checkpoint-epoch-30 --prompt "a person playing basketball"
"""

import argparse
import torch
from diffusers import DiffusionPipeline


def generate_video(checkpoint_path, prompt, output_path="output.mp4", num_frames=8, width=256, height=256):
    """Generate video from fine-tuned model."""
    
    from pathlib import Path
    
    print("=" * 60)
    print("Loading Fine-Tuned Model")
    print("=" * 60)
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Prompt: {prompt}")
    print()
    
    # Verify checkpoint exists
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        print(f"❌ Error: Checkpoint not found at {checkpoint_path}")
        print()
        print("Available checkpoints:")
        checkpoints_dir = Path("./checkpoints")
        if checkpoints_dir.exists():
            for cp in sorted(checkpoints_dir.iterdir()):
                if cp.is_dir():
                    print(f"  - {cp}")
        else:
            print("  No checkpoints directory found!")
        print()
        print("Make sure you:")
        print("1. Completed training")
        print("2. Have checkpoints in ./checkpoints/")
        print("3. Use the correct checkpoint path")
        return
    
    # Load your fine-tuned model
    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    
    print("Loading model components...")
    try:
        pipeline = DiffusionPipeline.from_pretrained(
            str(checkpoint_path),
            torch_dtype=torch.float16 if device != "cpu" else torch.float32,
        )
        
        # Memory optimizations for Mac
        if device == "mps":
            print("Applying Mac/MPS memory optimizations...")
            # Enable CPU offloading to save memory
            pipeline.enable_model_cpu_offload()
            # Enable attention slicing
            pipeline.enable_attention_slicing(1)
            print("  ✓ CPU offloading enabled")
            print("  ✓ Attention slicing enabled")
        else:
            pipeline = pipeline.to(device)
        
        print("✓ Model loaded successfully")
        print()
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        print()
        print("This might mean:")
        print("1. The checkpoint is corrupted")
        print("2. Training didn't save properly")
        print("3. Missing model files in checkpoint")
        return
    
    # Clear memory before generation
    if device == "mps":
        import gc
        gc.collect()
        torch.mps.empty_cache()
    
    # Generate video
    print("Generating video...")
    print(f"Resolution: {width}x{height}")
    print(f"Frames: {num_frames}")
    print("Note: This may take a few minutes on Mac...")
    print()
    
    try:
        video_frames = pipeline(
            prompt=prompt,
            num_frames=num_frames,
            height=height,
            width=width,
            num_inference_steps=50,
        ).frames[0]
    except RuntimeError as e:
        if "out of memory" in str(e):
            print("❌ Out of memory error!")
            print()
            print("Try reducing:")
            print("  --width 256 --height 256")
            print("  --num_frames 8")
            print()
            print("Example:")
            print(f"  python generate_finetuned.py --checkpoint {checkpoint_path} \\")
            print(f"    --prompt \"{prompt}\" \\")
            print("    --width 256 --height 256 --num_frames 8")
            return
        else:
            raise

    
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
    parser.add_argument("--num_frames", type=int, default=8,
                       help="Number of frames to generate (default: 8 for memory efficiency)")
    parser.add_argument("--width", type=int, default=256,
                       help="Video width (default: 256 for memory efficiency)")
    parser.add_argument("--height", type=int, default=256,
                       help="Video height (default: 256 for memory efficiency)")
    
    args = parser.parse_args()
    
    generate_video(
        checkpoint_path=args.checkpoint,
        prompt=args.prompt,
        output_path=args.output,
        num_frames=args.num_frames,
        width=args.width,
        height=args.height
    )
