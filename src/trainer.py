"""
Trainer class for text-to-video model fine-tuning
Optimized for Mac/Apple Silicon with MPS backend
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
from tqdm.auto import tqdm
import numpy as np
from typing import Optional, Dict, List
import os
import time

from diffusers import DDPMScheduler
from transformers import CLIPTokenizer


class VideoTrainer:
    """
    Trainer for fine-tuning text-to-video diffusion models.
    """
    
    def __init__(
        self,
        model,
        train_dataset,
        val_dataset,
        config: Dict,
        device: str = "mps"
    ):
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.config = config
        self.device = device
        
        # Move model to device
        print(f"Moving model to {device}...")
        self.model = self.model.to(device)
        
        # Enable gradient checkpointing if configured and supported
        if config.get("gradient_checkpointing", False):
            if hasattr(self.model, 'unet'):
                # Check if model supports gradient checkpointing
                if hasattr(self.model.unet, '_supports_gradient_checkpointing') and self.model.unet._supports_gradient_checkpointing:
                    self.model.unet.enable_gradient_checkpointing()
                    print("✓ Gradient checkpointing enabled")
                else:
                    print("⚠️  Gradient checkpointing not supported by this model, skipping...")
                    print("   (This is normal for UNet3DConditionModel - training will use more memory)")

        
        # Setup optimizer
        self.optimizer = self._setup_optimizer()
        
        # Setup scheduler
        self.lr_scheduler = self._setup_scheduler()
        
        # Setup noise scheduler for training
        self.noise_scheduler = DDPMScheduler.from_pretrained(
            config.get("model_id", "cerspense/zeroscope_v2_576w"),
            subfolder="scheduler"
        )
        
        # Setup tokenizer
        self.tokenizer = CLIPTokenizer.from_pretrained(
            config.get("model_id", "cerspense/zeroscope_v2_576w"),
            subfolder="tokenizer"
        )
        
        # Setup data loaders
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=config.get("batch_size", 1),
            shuffle=True,
            num_workers=config.get("dataloader_num_workers", 0),
            pin_memory=config.get("pin_memory", False),
            drop_last=True
        )
        
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=config.get("batch_size", 1),
            shuffle=False,
            num_workers=0,
            pin_memory=False
        ) if val_dataset else None
        
        # Setup logging
        self.log_dir = Path(config.get("log_dir", "./logs"))
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        if config.get("use_tensorboard", True):
            self.writer = SummaryWriter(log_dir=self.log_dir)
        else:
            self.writer = None
        
        # Setup checkpointing
        self.checkpoint_dir = Path(config.get("checkpoint_dir", "./checkpoints"))
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Training state
        self.global_step = 0
        self.current_epoch = 0
        
        # EMA model (optional)
        self.use_ema = config.get("ema_decay", None) is not None
        if self.use_ema:
            self.ema_decay = config["ema_decay"]
            self.ema_model = self._create_ema_model()
    
    def _setup_optimizer(self):
        """Setup AdamW optimizer."""
        trainable_params = list(filter(lambda p: p.requires_grad, self.model.unet.parameters()))
        
        optimizer = torch.optim.AdamW(
            trainable_params,
            lr=self.config.get("learning_rate", 5e-6),
            betas=self.config.get("betas", (0.9, 0.999)),
            weight_decay=self.config.get("weight_decay", 0.01),
            eps=self.config.get("eps", 1e-8)
        )
        
        print(f"✓ Optimizer setup: AdamW with LR={self.config.get('learning_rate')}")
        return optimizer
    
    def _setup_scheduler(self):
        """Setup learning rate scheduler."""
        from torch.optim.lr_scheduler import CosineAnnealingLR
        
        total_steps = len(self.train_loader) * self.config.get("num_epochs", 50)
        
        scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=total_steps,
            eta_min=1e-7
        )
        
        return scheduler
    
    def _create_ema_model(self):
        """Create EMA (Exponential Moving Average) copy of model."""
        # For simplicity, we'll skip EMA on Mac to save memory
        return None
    
    def _encode_text(self, captions: List[str]):
        """Encode text captions using CLIP tokenizer."""
        text_inputs = self.tokenizer(
            captions,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt"
        )
        
        text_input_ids = text_inputs.input_ids.to(self.device)
        
        # Get text embeddings from text encoder
        with torch.no_grad():
            encoder_hidden_states = self.model.text_encoder(text_input_ids)[0]
        
        return encoder_hidden_states
    
    def train_step(self, batch: Dict) -> float:
        """
        Single training step.
        
        Args:
            batch: Dictionary with 'videos' and 'captions'
            
        Returns:
            Loss value
        """
        videos = batch['videos'].to(self.device)  # (B, T, C, H, W)
        captions = batch['captions']
        
        # Encode text
        encoder_hidden_states = self._encode_text(captions)
        
        # Rearrange video to (B, C, T, H, W) for diffusion model
        videos = videos.permute(0, 2, 1, 3, 4)
        
        # Sample noise
        noise = torch.randn_like(videos)
        
        # Sample timesteps
        batch_size = videos.shape[0]
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps,
            (batch_size,), device=self.device
        ).long()
        
        # Add noise to videos
        noisy_videos = self.noise_scheduler.add_noise(videos, noise, timesteps)
        
        # Predict noise
        model_pred = self.model.unet(
            noisy_videos,
            timesteps,
            encoder_hidden_states=encoder_hidden_states
        ).sample
        
        # Compute loss (MSE between predicted and actual noise)
        loss = F.mse_loss(model_pred.float(), noise.float(), reduction="mean")
        
        return loss
    
    def train(self):
        """Main training loop."""
        print("=" * 60)
        print("Starting Training")
        print("=" * 60)
        print(f"Device: {self.device}")
        print(f"Training samples: {len(self.train_dataset)}")
        print(f"Batch size: {self.config.get('batch_size')}")
        print(f"Gradient accumulation steps: {self.config.get('gradient_accumulation_steps')}")
        print(f"Effective batch size: {self.config['batch_size'] * self.config['gradient_accumulation_steps']}")
        print(f"Number of epochs: {self.config.get('num_epochs')}")
        print("=" * 60)
        
        self.model.unet.train()
        
        for epoch in range(self.config.get("num_epochs", 50)):
            self.current_epoch = epoch
            epoch_loss = 0.0
            
            progress_bar = tqdm(
                self.train_loader,
                desc=f"Epoch {epoch + 1}/{self.config.get('num_epochs')}"
            )
            
            for step, batch in enumerate(progress_bar):
                # Training step
                loss = self.train_step(batch)
                loss = loss / self.config.get("gradient_accumulation_steps", 1)
                loss.backward()
                
                # Gradient accumulation
                if (step + 1) % self.config.get("gradient_accumulation_steps", 1) == 0:
                    # Clip gradients
                    torch.nn.utils.clip_grad_norm_(
                        self.model.unet.parameters(),
                        self.config.get("max_grad_norm", 1.0)
                    )
                    
                    # Optimizer step
                    self.optimizer.step()
                    self.lr_scheduler.step()
                    self.optimizer.zero_grad()
                    
                    self.global_step += 1
                    
                    # Logging
                    if self.global_step % self.config.get("log_every_n_steps", 10) == 0:
                        lr = self.optimizer.param_groups[0]['lr']
                        
                        if self.writer:
                            self.writer.add_scalar("train/loss", loss.item(), self.global_step)
                            self.writer.add_scalar("train/lr", lr, self.global_step)
                        
                        progress_bar.set_postfix({
                            "loss": f"{loss.item():.4f}",
                            "lr": f"{lr:.2e}"
                        })
                    
                    # Save checkpoint
                    if self.global_step % self.config.get("save_checkpoint_every_n_steps", 1000) == 0:
                        self.save_checkpoint(f"checkpoint-step-{self.global_step}")
                
                epoch_loss += loss.item()
            
            # Epoch end
            avg_epoch_loss = epoch_loss / len(self.train_loader)
            print(f"Epoch {epoch + 1} completed. Average loss: {avg_epoch_loss:.4f}")
            
            if self.writer:
                self.writer.add_scalar("train/epoch_loss", avg_epoch_loss, epoch)
            
            # Save checkpoint at epoch end
            self.save_checkpoint(f"checkpoint-epoch-{epoch + 1}")
        
        print("Training completed!")
        if self.writer:
            self.writer.close()
    
    def save_checkpoint(self, checkpoint_name: str):
        """Save model checkpoint and manage storage."""
        checkpoint_path = self.checkpoint_dir / checkpoint_name
        checkpoint_path.mkdir(parents=True, exist_ok=True)
        
        # Save pipeline
        self.model.save_pretrained(checkpoint_path)
        
        # Save training state
        state = {
            'global_step': self.global_step,
            'epoch': self.current_epoch,
            'optimizer_state': self.optimizer.state_dict(),
            'lr_scheduler_state': self.lr_scheduler.state_dict(),
        }
        
        torch.save(state, checkpoint_path / "training_state.pt")
        
        print(f"✓ Checkpoint saved: {checkpoint_path}")
        
        # Auto-cleanup old checkpoints to save storage
        if self.config.get("delete_old_checkpoints", False):
            self._cleanup_old_checkpoints()
    
    def _cleanup_old_checkpoints(self):
        """Delete old checkpoints to save storage space."""
        max_to_keep = self.config.get("max_checkpoints_to_keep", 3)
        
        # Get all checkpoint directories sorted by modification time
        checkpoints = []
        for path in self.checkpoint_dir.iterdir():
            if path.is_dir() and path.name.startswith("checkpoint-"):
                checkpoints.append(path)
        
        # Sort by modification time (newest first)
        checkpoints.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        
        # Delete old checkpoints beyond max_to_keep
        for old_checkpoint in checkpoints[max_to_keep:]:
            try:
                import shutil
                shutil.rmtree(old_checkpoint)
                print(f"  🗑️  Deleted old checkpoint: {old_checkpoint.name}")
            except Exception as e:
                print(f"  ⚠️  Could not delete {old_checkpoint.name}: {e}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint."""
        checkpoint_path = Path(checkpoint_path)
        
        # Load training state
        state_path = checkpoint_path / "training_state.pt"
        if state_path.exists():
            state = torch.load(state_path, map_location=self.device)
            self.global_step = state['global_step']
            self.current_epoch = state['epoch']
            self.optimizer.load_state_dict(state['optimizer_state'])
            self.lr_scheduler.load_state_dict(state['lr_scheduler_state'])
            
            print(f"✓ Checkpoint loaded from: {checkpoint_path}")
            print(f"  Resuming from step {self.global_step}, epoch {self.current_epoch}")
        else:
            print(f"Warning: Training state not found at {state_path}")
