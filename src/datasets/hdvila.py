"""
HD-VILA-100M Dataset Loader for Text-to-Video Training
Optimized for Mac/Apple Silicon with MPS backend
"""

import os
import json
import csv
import torch
from torch.utils.data import Dataset
import cv2
import numpy as np
from PIL import Image
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import random


class HDVILADataset(Dataset):
    """
    HD-VILA-100M dataset loader for text-to-video model training.
    
    Dataset structure:
    - videos/: Contains MP4 video files
    - metadata.csv: Contains video_id, caption, duration, category, etc.
    
    Args:
        data_root: Root directory containing videos and metadata
        metadata_file: Path to CSV metadata file
        num_frames: Number of frames to extract from each video
        resolution: Target resolution (width, height)
        split: 'train' or 'val'
        max_samples: Maximum number of samples to load (for testing)
    """
    
    def __init__(
        self,
        data_root: str,
        metadata_file: str = "metadata.csv",
        num_frames: int = 16,
        resolution: Tuple[int, int] = (256, 256),
        split: str = "train",
        max_samples: Optional[int] = None,
        transform=None
    ):
        self.data_root = Path(data_root)
        self.videos_dir = self.data_root / "videos"
        self.num_frames = num_frames
        self.resolution = resolution
        self.split = split
        self.transform = transform
        
        # Load metadata
        self.samples = self._load_metadata(
            self.data_root / metadata_file, 
            max_samples
        )
        
        print(f"Loaded {len(self.samples)} samples for {split} split")
    
    def _load_metadata(
        self, 
        metadata_path: Path, 
        max_samples: Optional[int]
    ) -> List[Dict]:
        """Load and parse metadata CSV file."""
        samples = []
        
        if not metadata_path.exists():
            print(f"Warning: Metadata file not found: {metadata_path}")
            return samples
        
        with open(metadata_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for idx, row in enumerate(reader):
                if max_samples and idx >= max_samples:
                    break
                
                # Expected columns: video_id, caption, duration, category, split
                video_id = row.get('video_id', row.get('videoid', ''))
                caption = row.get('caption', row.get('name', row.get('description', '')))
                
                # Filter by split if specified in metadata
                if 'split' in row and row['split'] != self.split:
                    continue
                
                video_path = self.videos_dir / f"{video_id}.mp4"
                
                samples.append({
                    'video_id': video_id,
                    'video_path': video_path,
                    'caption': caption,
                    'duration': float(row.get('duration', 0)),
                    'category': row.get('category', 'unknown')
                })
        
        return samples
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def _extract_frames(self, video_path: Path) -> Optional[torch.Tensor]:
        """
        Extract frames from video file.
        
        Returns:
            Tensor of shape (num_frames, C, H, W) or None if failed
        """
        if not video_path.exists():
            return None
        
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return None
        
        # Get video properties
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        if total_frames == 0:
            cap.release()
            return None
        
        # Sample frames uniformly
        frame_indices = np.linspace(0, total_frames - 1, self.num_frames, dtype=int)
        
        frames = []
        for frame_idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            
            if not ret:
                continue
            
            # Convert BGR to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Resize
            frame = cv2.resize(frame, self.resolution, interpolation=cv2.INTER_LANCZOS4)
            
            # Convert to PIL Image for potential transforms
            frame = Image.fromarray(frame)
            frames.append(frame)
        
        cap.release()
        
        if len(frames) < self.num_frames:
            # Pad with last frame if needed
            while len(frames) < self.num_frames:
                frames.append(frames[-1] if frames else Image.new('RGB', self.resolution))
        
        # Convert to tensor: (T, C, H, W)
        frames_tensor = torch.stack([
            torch.from_numpy(np.array(frame)).permute(2, 0, 1) for frame in frames
        ])
        
        # Normalize to [-1, 1]
        frames_tensor = frames_tensor.float() / 127.5 - 1.0
        
        return frames_tensor
    
    def __getitem__(self, idx: int) -> Dict[str, any]:
        """
        Get a single sample.
        
        Returns:
            Dictionary containing:
            - 'video': Tensor of shape (num_frames, C, H, W)
            - 'caption': Text caption string
            - 'video_id': Video identifier
        """
        sample = self.samples[idx]
        
        # Extract frames
        frames = self._extract_frames(sample['video_path'])
        
        # If video loading fails, try another random sample
        if frames is None:
            print(f"Failed to load video: {sample['video_id']}, trying another...")
            return self.__getitem__(random.randint(0, len(self) - 1))
        
        # Apply transforms if provided
        if self.transform:
            frames = self.transform(frames)
        
        return {
            'video': frames,  # (T, C, H, W)
            'caption': sample['caption'],
            'video_id': sample['video_id']
        }


def collate_fn(batch: List[Dict]) -> Dict[str, any]:
    """
    Custom collate function for batching.
    
    Args:
        batch: List of samples from __getitem__
        
    Returns:
        Batched dictionary with:
        - 'videos': (B, T, C, H, W)
        - 'captions': List of caption strings
        - 'video_ids': List of video IDs
    """
    videos = torch.stack([item['video'] for item in batch])
    captions = [item['caption'] for item in batch]
    video_ids = [item['video_id'] for item in batch]
    
    return {
        'videos': videos,
        'captions': captions,
        'video_ids': video_ids
    }
