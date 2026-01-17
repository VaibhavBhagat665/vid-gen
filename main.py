import argparse
import os
import sys
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent / "src"))

from src.model_loader import ModelManager
from src.video_generator import VideoGenerator
from src.ui_interface import launch_ui
from config.settings import DEFAULT_CONFIG


def parse_args():
    parser = argparse.ArgumentParser(description="Generate videos from text prompts locally")
    parser.add_argument("--prompt", type=str, help="Text prompt for video generation")
    parser.add_argument("--output", type=str, default=None, help="Output video path")
    parser.add_argument("--duration", type=int, default=3, help="Video duration in seconds")
    parser.add_argument("--width", type=int, default=512, help="Video width")
    parser.add_argument("--height", type=int, default=512, help="Video height")
    parser.add_argument("--fps", type=int, default=8, help="Frames per second")
    parser.add_argument("--steps", type=int, default=20, help="Inference steps")
    parser.add_argument("--model", type=str, default="zeroscope", 
                       choices=["zeroscope", "modelscope"], help="Model to use")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--ui", action="store_true", help="Launch web interface")
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Create outputs directory
    os.makedirs("outputs", exist_ok=True)
    
    if args.ui:
        print("Launching web interface...")
        launch_ui()
        return
    
    if not args.prompt:
        print("Error: Please provide a prompt with --prompt or use --ui for web interface")
        return
    
    print(f"Initializing {args.model} model...")
    model_manager = ModelManager()
    model = model_manager.load_model(args.model)
    
    generator = VideoGenerator(model, args.model)
    
    # Generate output filename if not provided
    if not args.output:
        safe_prompt = "".join(c for c in args.prompt[:30] if c.isalnum() or c in (' ', '-', '_')).rstrip()
        safe_prompt = safe_prompt.replace(' ', '_')
        args.output = f"outputs/{safe_prompt}_{args.model}.mp4"
    
    print(f"Generating video: '{args.prompt}'")
    print(f"Output: {args.output}")
    
    try:
        video_path = generator.generate_video(
            prompt=args.prompt,
            output_path=args.output,
            duration=args.duration,
            width=args.width,
            height=args.height,
            fps=args.fps,
            num_inference_steps=args.steps,
            seed=args.seed
        )
        print(f"Video generated successfully: {video_path}")
    except Exception as e:
        print(f"Error generating video: {str(e)}")


if __name__ == "__main__":
    main()