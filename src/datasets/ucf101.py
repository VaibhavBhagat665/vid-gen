"""
UCF-101 Dataset Loader for Text-to-Video Training
Action recognition videos from Kaggle
"""

import os
import csv
import torch
from torch.utils.data import Dataset
import cv2
import numpy as np
from PIL import Image
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import random


class UCF101Dataset(Dataset):
    """
    UCF-101 dataset loader.
    
    Dataset from: https://www.kaggle.com/datasets/abdallahwagih/ucf101-videos
    
    Structure:
    - videos/: Contains video files organized by action class
        - ApplyEyeMakeup/
        - ApplyLipstick/
        - etc.
    - metadata.csv: video_id, caption (action class), category, split
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
        metadata_path = self.data_root / metadata_file
        if metadata_path.exists():
            self.samples = self._load_metadata(metadata_path, max_samples)
        else:
            # Auto-generate from video files
            self.samples = self._generate_metadata(max_samples)
        
        print(f"Loaded {len(self.samples)} UCF-101 samples for {split} split")
    
    def _generate_metadata(self, max_samples: Optional[int]) -> List[Dict]:
        """Generate metadata from video file structure."""
        samples = []
        
        if not self.videos_dir.exists():
            print(f"Warning: Videos directory not found: {self.videos_dir}")
            return samples
        
        # UCF-101 has action classes as subdirectories
        for action_dir in self.videos_dir.iterdir():
            if not action_dir.is_dir():
                continue
            
            action_class = action_dir.name
            
            for video_file in action_dir.glob("*.avi"):
                if max_samples and len(samples) >= max_samples:
                    break
                
                video_id = video_file.stem
                
                # Create caption from action class (convert CamelCase to sentence)
                caption = self._action_to_caption(action_class)
                
                # Train/val split (80/20)
                is_train = len(samples) % 5 != 0
                
                if (self.split == "train" and is_train) or (self.split == "val" and not is_train):
                    samples.append({
                        'video_id': video_id,
                        'video_path': video_file,
                        'caption': caption,
                        'category': action_class,
                        'duration': 0,  # Unknown
                    })
        
        return samples
    
    def _action_to_caption(self, action_class: str) -> str:
        """Convert action class name to caption."""
        # Split camelCase: ApplyEyeMakeup -> Apply Eye Makeup
        import re
        caption = re.sub('([a-z])([A-Z])', r'\1 \2', action_class)
        return f"a person {caption.lower()}"
    
    def _load_metadata(
        self, 
        metadata_path: Path, 
        max_samples: Optional[int]
    ) -> List[Dict]:
        """Load metadata from CSV."""
        samples = []
        
        with open(metadata_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for idx, row in enumerate(reader):
                if max_samples and idx >= max_samples:
                    break
                
                video_id = row.get('video_id', '')
                caption = row.get('caption', row.get('name', ''))
                split_val = row.get('split', 'train')
                
                if split_val != self.split:
                    continue
                
                # Find video file
                category = row.get('category', '')
                video_path = self.videos_dir / category / f"{video_id}.avi"
                
                if not video_path.exists():
                    # Try without category
                    video_path = self.videos_dir / f"{video_id}.avi"
                
                samples.append({
                    'video_id': video_id,
                    'video_path': video_path,
                    'caption': caption,
                    'category': category,
                    'duration': float(row.get('duration', 0)),
                })
        
        return samples
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def _extract_frames(self, video_path: Path) -> Optional[torch.Tensor]:
        """Extract frames from video file."""
        if not video_path.exists():
            return None
        
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return None
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
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
            
            # Convert to PIL Image
            frame = Image.fromarray(frame)
            frames.append(frame)
        
        cap.release()
        
        # Pad if needed
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
        """Get a single sample."""
        sample = self.samples[idx]
        
        # Extract frames
        frames = self._extract_frames(sample['video_path'])
        
        # If video loading fails, try another
        if frames is None:
            return self.__getitem__(random.randint(0, len(self) - 1))
        
        # Apply transforms
        if self.transform:
            frames = self.transform(frames)
        
        return {
            'video': frames,  # (T, C, H, W)
            'caption': sample['caption'],
            'video_id': sample['video_id']
        }
