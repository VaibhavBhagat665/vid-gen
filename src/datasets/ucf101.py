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
        metadata_file: str = "train.csv",  # Changed default
        num_frames: int = 16,
        resolution: Tuple[int, int] = (256, 256),
        split: str = "train",
        max_samples: Optional[int] = None,
        transform=None
    ):
        self.data_root = Path(data_root)
        # Kaggle format: train/ and test/ folders
        self.videos_dir = self.data_root / split  # Use split name directly
        self.num_frames = num_frames
        self.resolution = resolution
        self.split = split
        self.transform = transform
        
        # Load metadata - Kaggle format
        if split == "train":
            metadata_path = self.data_root / "train.csv"
        else:
            metadata_path = self.data_root / "test.csv"
            
        if metadata_path.exists():
            self.samples = self._load_metadata_kaggle(metadata_path, max_samples)
        else:
            # Fallback: generate from video files
            self.samples = self._generate_metadata(max_samples)
        
        print(f"Loaded {len(self.samples)} UCF-101 samples for {split} split")
    
    def _load_metadata_kaggle(
        self, 
        metadata_path: Path, 
        max_samples: Optional[int]
    ) -> List[Dict]:
        """Load metadata from Kaggle CSV format."""
        samples = []
        
        with open(metadata_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            
            # DEBUG: Print CSV columns
            first_row = None
            for idx, row in enumerate(reader):
                if idx == 0:
                    print(f"DEBUG: CSV columns: {list(row.keys())}")
                    print(f"DEBUG: First row: {row}")
                    first_row = row
                
                if max_samples and len(samples) >= max_samples:
                    break
                
                # Kaggle CSV might have different column names
                # Try common variations
                video_id = row.get('video_id', row.get('filename', row.get('id', row.get('video_name', f'video{idx}'))))
                caption = row.get('caption', row.get('label', row.get('class', row.get('tag', ''))))
                
                # Video file path - assuming it's in train/ or test/
                video_path = self.videos_dir / f"{video_id}"
                
                # Try different extensions
                if not video_path.exists():
                    for ext in ['.avi', '.mp4', '.mkv', '.AVI', '.MP4', '.MKV']:
                        test_path = self.videos_dir / f"{video_id}{ext}"
                        if test_path.exists():
                            video_path = test_path
                            break
                
                # Create caption from label if needed
                if not caption and 'label' in row:
                    caption = self._action_to_caption(row['label'])
                elif not caption and 'tag' in row:
                    caption = self._action_to_caption(row['tag'])
                elif not caption:
                    caption = "a person performing an action"
                
                # Only add if video file exists!
                if video_path.exists():
                    samples.append({
                        'video_id': video_id,
                        'video_path': video_path,
                        'caption': caption,
                        'category': row.get('label', row.get('class', row.get('tag', 'unknown'))),
                        'duration': float(row.get('duration', 0)),
                    })
                elif idx < 5:  # Debug first few failures
                    print(f"DEBUG: Video not found: {video_path}")
                    print(f"  Tried: {video_id} with extensions .avi, .mp4, .mkv")
        
        print(f"DEBUG: Loaded {len(samples)} samples from {metadata_path}")
        return samples
    
    def _generate_metadata(self, max_samples: Optional[int]) -> List[Dict]:
        """Generate metadata from video file structure."""
        samples = []
        
        if not self.videos_dir.exists():
            print(f"Warning: Videos directory not found: {self.videos_dir}")
            return samples
        
        # For Kaggle format - videos are directly in train/ or test/
        for video_file in self.videos_dir.glob("*.*"):
            if video_file.suffix.lower() not in ['.avi', '.mp4', '.mkv']:
                continue
                
            if max_samples and len(samples) >= max_samples:
                break
            
            video_id = video_file.stem
            
            # Try to extract action class from filename
            # Common formats: "classname_video123" or just "video123"
            parts = video_id.split('_')
            if len(parts) > 1:
                action_class = parts[0]
            else:
                action_class = "action"
            
            caption = self._action_to_caption(action_class)
            
            # Train/val split (80/20)
            is_train = len(samples) % 5 != 0
            
            if (self.split == "train" and is_train) or (self.split == "val" and not is_train):
                samples.append({
                    'video_id': video_id,
                    'video_path': video_file,
                    'caption': caption,
                    'category': action_class,
                    'duration': 0,
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
    
    def __getitem__(self, idx: int, _retry_count: int = 0) -> Dict[str, any]:
        """Get a single sample."""
        sample = self.samples[idx]
        
        # Extract frames
        frames = self._extract_frames(sample['video_path'])
        
        # If video loading fails, try another (with recursion limit)
        if frames is None and _retry_count < 10:
            return self.__getitem__(random.randint(0, len(self) - 1), _retry_count + 1)
        elif frames is None:
            # All retries failed - return dummy frames
            print(f"Warning: Could not load video {sample['video_id']}, using dummy frames")
            dummy_frames = torch.zeros(self.num_frames, 3, *self.resolution)
            frames = dummy_frames
        
        # Apply transforms
        if self.transform:
            frames = self.transform(frames)
        
        return {
            'video': frames,  # (T, C, H, W)
            'caption': sample['caption'],
            'video_id': sample['video_id']
        }
