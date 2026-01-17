"""
WebVid Dataset Loader (Fallback/Supplementary)
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


class WebVidDataset(Dataset):
    """
    WebVid-10M dataset loader as fallback option.
    
    Similar structure to HD-VILA but may have different metadata format.
    """
    
    def __init__(
        self,
        data_root: str,
        metadata_file: str = "results_10M_train.csv",
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
        
        print(f"Loaded {len(self.samples)} WebVid samples for {split} split")
    
    def _load_metadata(
        self,
        metadata_path: Path,
        max_samples: Optional[int]
    ) -> List[Dict]:
        """Load WebVid metadata."""
        samples = []
        
        if not metadata_path.exists():
            print(f"Warning: Metadata file not found: {metadata_path}")
            return samples
        
        with open(metadata_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for idx, row in enumerate(reader):
                if max_samples and idx >= max_samples:
                    break
                
                # WebVid format: videoid, page_dir, name, duration, etc.
                video_id = row.get('videoid', '')
                page_dir = row.get('page_dir', '')
                caption = row.get('name', '')
                
                # Construct video path
                video_path = self.videos_dir / page_dir / f"{video_id}.mp4"
                
                samples.append({
                    'video_id': video_id,
                    'video_path': video_path,
                    'caption': caption,
                    'duration': float(row.get('duration', 0)),
                })
        
        return samples
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def _extract_frames(self, video_path: Path) -> Optional[torch.Tensor]:
        """Extract frames from video - same as HD-VILA."""
        if not video_path.exists():
            return None
        
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return None
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames == 0:
            cap.release()
            return None
        
        frame_indices = np.linspace(0, total_frames - 1, self.num_frames, dtype=int)
        
        frames = []
        for frame_idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            
            if not ret:
                continue
            
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, self.resolution, interpolation=cv2.INTER_LANCZOS4)
            frame = Image.fromarray(frame)
            frames.append(frame)
        
        cap.release()
        
        while len(frames) < self.num_frames:
            frames.append(frames[-1] if frames else Image.new('RGB', self.resolution))
        
        frames_tensor = torch.stack([
            torch.from_numpy(np.array(frame)).permute(2, 0, 1) for frame in frames
        ])
        
        frames_tensor = frames_tensor.float() / 127.5 - 1.0
        
        return frames_tensor
    
    def __getitem__(self, idx: int) -> Dict[str, any]:
        """Get a single sample."""
        sample = self.samples[idx]
        
        frames = self._extract_frames(sample['video_path'])
        
        if frames is None:
            return self.__getitem__(random.randint(0, len(self) - 1))
        
        if self.transform:
            frames = self.transform(frames)
        
        return {
            'video': frames,
            'caption': sample['caption'],
            'video_id': sample['video_id']
        }
