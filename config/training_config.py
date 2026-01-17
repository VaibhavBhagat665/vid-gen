"""
Training Configuration for Mac Apple Silicon
Optimized for MPS (Metal Performance Shaders) backend
"""

import torch

# Training device
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

# Dataset configuration - OPTIMIZED FOR 200GB STORAGE
DATASET_CONFIG = {
    "name": "hdvila",  # or "webvid"
    "data_root": "./data/hdvila",  # Path to dataset
    "metadata_file": "metadata.csv",
    "num_frames": 16,  # Frames per video
    "resolution": (320, 320),  # Medium resolution for better quality
    "max_train_samples": 10000,  # 10K videos = ~200GB dataset
    "max_val_samples": 500,  # Validation samples
}

# Model configuration
MODEL_CONFIG = {
    "model_name": "zeroscope",  # Base model to fine-tune
    "model_id": "cerspense/zeroscope_v2_576w",
    "enable_gradient_checkpointing": True,  # Save memory
    "enable_xformers": False,  # Not available on Mac
}

# Training hyperparameters (Mac optimized)
TRAINING_CONFIG = {
    "device": DEVICE,
    "learning_rate": 5e-6,  # Lower LR for fine-tuning
    "batch_size": 1,  # Small batch size for Mac memory
    "num_epochs": 30,  # 30 epochs is good for 10K videos
    "warmup_steps": 500,
    "gradient_accumulation_steps": 16,  # Simulate batch_size=16
    "max_grad_norm": 1.0,
    "optimizer": "adamw",
    "betas": (0.9, 0.999),
    "weight_decay": 0.01,
    "eps": 1e-8,
    "scheduler": "cosine",
    "mixed_precision": "fp16" if DEVICE == "mps" else "no",  # FP16 for memory
    "ema_decay": 0.9999,
    "gradient_checkpointing": True,
    "enable_tf32": False,  # Not supported on Mac
    "dataloader_num_workers": 0,  # Mac works best with 0 workers
    "pin_memory": False,  # Not beneficial on Mac unified memory
}

# Logging and checkpointing - OPTIMIZED FOR 200GB STORAGE
LOGGING_CONFIG = {
    "log_every_n_steps": 10,
    "val_every_n_steps": 500,
    "save_checkpoint_every_n_steps": 1000,
    "generate_samples_every_n_steps": 500,
    "num_sample_prompts": 4,
    "use_wandb": False,  # Set to True if you install wandb
    "use_tensorboard": True,
    "log_dir": "./logs",
    "checkpoint_dir": "./checkpoints",
    "output_dir": "./training_outputs",
    # Storage management (keeps only last N checkpoints)
    "max_checkpoints_to_keep": 3,  # Keep only last 3 checkpoints (~20GB)
    "delete_old_checkpoints": True,  # Auto-delete old checkpoints
}

# Sample prompts for validation
SAMPLE_PROMPTS = [
    "a cat playing with a ball of yarn",
    "sunset over the ocean with waves",
    "person walking through a forest",
    "city street at night with neon lights",
    "bird flying in slow motion",
    "waterfall in a tropical rainforest"
]

# Print configuration on import
if __name__ != "__main__":
    print(f"Training Device: {DEVICE}")
    if DEVICE == "mps":
        print("✓ MPS backend available - using Apple Silicon GPU")
    elif DEVICE == "cpu":
        print("⚠ Warning: MPS not available, using CPU (will be slow)")
