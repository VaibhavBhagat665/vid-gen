"""
Main Training Script for Text-to-Video Model Fine-tuning
Optimized for Mac/Apple Silicon with MPS backend

Usage:
    python train.py --data_root ./data/hdvila --max_samples 1000
"""

import argparse
import os
import sys
from pathlib import Path
import torch

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from diffusers import DiffusionPipeline
from src.datasets import get_dataset, collate_fn
from src.trainer import VideoTrainer
from config.training_config import (
    DEVICE,
    DATASET_CONFIG,
    MODEL_CONFIG,
    TRAINING_CONFIG,
    LOGGING_CONFIG
)


def parse_args():
    parser = argparse.ArgumentParser(description="Train text-to-video model")
    
    # Dataset arguments
    parser.add_argument("--dataset", type=str, default="ucf101", 
                       choices=["hdvila", "webvid", "ucf101"], help="Dataset to use")
    parser.add_argument("--data_root", type=str, required=True,
                       help="Root directory containing dataset")
    parser.add_argument("--metadata_file", type=str, default="metadata.csv",
                       help="Metadata CSV file name")
    parser.add_argument("--max_samples", type=int, default=None,
                       help="Maximum training samples (for testing)")
    parser.add_argument("--max_val_samples", type=int, default=500,
                       help="Maximum validation samples")
    
    # Model arguments
    parser.add_argument("--model", type=str, default="zeroscope",
                       choices=["zeroscope", "modelscope"], help="Base model")
    parser.add_argument("--resume_from", type=str, default=None,
                       help="Checkpoint path to resume from")
    
    # Training arguments
    parser.add_argument("--batch_size", type=int, default=1,
                       help="Batch size (keep at 1 for Mac)")
    parser.add_argument("--num_epochs", type=int, default=50,
                       help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=5e-6,
                       help="Learning rate")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=16,
                       help="Gradient accumulation steps")
    
    # Resolution arguments
    parser.add_argument("--width", type=int, default=256,
                       help="Video width")
    parser.add_argument("--height", type=int, default=256,
                       help="Video height")
    parser.add_argument("--num_frames", type=int, default=16,
                       help="Number of frames per video")
    
    # Logging arguments
    parser.add_argument("--log_dir", type=str, default="./logs",
                       help="Directory for logs")
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints",
                       help="Directory for checkpoints")
    parser.add_argument("--use_wandb", action="store_true",
                       help="Use Weights & Biases for logging")
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    print("=" * 60)
    print("Text-to-Video Model Training")
    print("=" * 60)
    
    # Check device
    print(f"Device: {DEVICE}")
    if DEVICE == "mps":
        print("✓ Using Apple Silicon GPU (MPS backend)")
    elif DEVICE == "cpu":
        print("⚠️  Warning: MPS not available. Training will be VERY slow on CPU!")
        print("   Make sure you're running on Mac with Apple Silicon (M1/M2/M3)")
    
    # Update configs with args
    dataset_config = DATASET_CONFIG.copy()
    dataset_config.update({
        "name": args.dataset,
        "data_root": args.data_root,
        "metadata_file": args.metadata_file,
        "max_train_samples": args.max_samples,
        "max_val_samples": args.max_val_samples,
        "num_frames": args.num_frames,
        "resolution": (args.width, args.height),
    })
    
    training_config = TRAINING_CONFIG.copy()
    training_config.update({
        "batch_size": args.batch_size,
        "num_epochs": args.num_epochs,
        "learning_rate": args.learning_rate,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
    })
    
    logging_config = LOGGING_CONFIG.copy()
    logging_config.update({
        "log_dir": args.log_dir,
        "checkpoint_dir": args.checkpoint_dir,
        "use_wandb": args.use_wandb,
    })
    
    # Merge configs
    full_config = {
        **training_config,
        **logging_config,
        "model_id": MODEL_CONFIG["model_id"],
    }
    
    # Create datasets
    print("\n" + "=" * 60)
    print("Loading Datasets")
    print("=" * 60)
    
    train_dataset = get_dataset(
        args.dataset,
        data_root=args.data_root,
        metadata_file=args.metadata_file,
        num_frames=args.num_frames,
        resolution=(args.width, args.height),
        split="train",
        max_samples=args.max_samples
    )
    
    val_dataset = get_dataset(
        args.dataset,
        data_root=args.data_root,
        metadata_file=args.metadata_file,
        num_frames=args.num_frames,
        resolution=(args.width, args.height),
        split="val",
        max_samples=args.max_val_samples
    ) if args.max_val_samples > 0 else None
    
    if len(train_dataset) == 0:
        print("❌ Error: No training samples found!")
        print(f"   Please check that {args.data_root} contains videos and metadata.")
        print(f"   Expected structure:")
        print(f"   {args.data_root}/")
        print(f"   ├── videos/")
        print(f"   │   ├── video1.mp4")
        print(f"   │   └── video2.mp4")
        print(f"   └── metadata.csv")
        return
    
    # Load model
    print("\n" + "=" * 60)
    print("Loading Model")
    print("=" * 60)
    print(f"Loading {args.model} model: {MODEL_CONFIG['model_id']}")
    print("This may take a while on first run...")
    
    model = DiffusionPipeline.from_pretrained(
        MODEL_CONFIG["model_id"],
        torch_dtype=torch.float16 if DEVICE == "mps" else torch.float32,
    )
    
    # Don't move to device yet - trainer will do it
    print("✓ Model loaded successfully")
    
    # Create trainer
    print("\n" + "=" * 60)
    print("Initializing Trainer")
    print("=" * 60)
    
    trainer = VideoTrainer(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        config=full_config,
        device=DEVICE
    )
    
    # Resume from checkpoint if specified
    if args.resume_from:
        print(f"Resuming from checkpoint: {args.resume_from}")
        trainer.load_checkpoint(args.resume_from)
    
    # Start training
    print("\n" + "=" * 60)
    print("Starting Training")
    print("=" * 60)
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset) if val_dataset else 0}")
    print(f"Batch size: {args.batch_size}")
    print(f"Gradient accumulation: {args.gradient_accumulation_steps}")
    print(f"Effective batch size: {args.batch_size * args.gradient_accumulation_steps}")
    print(f"Total epochs: {args.num_epochs}")
    print(f"Resolution: {args.width}x{args.height}")
    print(f"Frames per video: {args.num_frames}")
    print("=" * 60)
    print()
    
    try:
        trainer.train()
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user")
        print("Saving checkpoint...")
        trainer.save_checkpoint("checkpoint-interrupted")
        print("✓ Checkpoint saved. You can resume training with --resume_from")
    except Exception as e:
        print(f"\n\n❌ Training failed with error: {e}")
        import traceback
        traceback.print_exc()
        print("\nTrying to save checkpoint...")
        try:
            trainer.save_checkpoint("checkpoint-error")
            print("✓ Emergency checkpoint saved")
        except:
            print("❌ Could not save checkpoint")
    
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print(f"Checkpoints saved to: {args.checkpoint_dir}")
    print(f"Logs saved to: {args.log_dir}")


if __name__ == "__main__":
    main()
