"""
Download UCF-101 Dataset using Kagglehub
Saves to ./data/ucf101 in your project directory

Usage:
    python download_ucf101.py
"""

import kagglehub
import shutil
from pathlib import Path
import os


def download_ucf101():
    """Download UCF-101 and move to project directory."""
    
    print("=" * 60)
    print("UCF-101 Dataset Download")
    print("=" * 60)
    print()
    print("Dataset: UCF-101 Action Recognition")
    print("Size: ~7.2GB")
    print("Videos: 13,320")
    print()
    
    # Target directory in your project
    target_dir = Path("./data/ucf101")
    target_dir.mkdir(parents=True, exist_ok=True)
    
    print("Downloading from Kaggle...")
    print("(This may take 10-20 minutes depending on internet speed)")
    print()
    
    try:
        # Download using kagglehub (downloads to cache)
        cache_path = kagglehub.dataset_download("abdallahwagih/ucf101-videos")
        
        print(f"✓ Downloaded to cache: {cache_path}")
        print()
        
        # Copy from cache to project directory
        print(f"Copying to project directory: {target_dir}")
        print("(This may take a few minutes...)")
        print()
        
        cache_path = Path(cache_path)
        
        # Copy all files from cache to target
        if cache_path.is_dir():
            # If cache_path is a directory, copy its contents
            for item in cache_path.iterdir():
                dest = target_dir / item.name
                
                if item.is_dir():
                    if dest.exists():
                        shutil.rmtree(dest)
                    shutil.copytree(item, dest)
                    print(f"  Copied: {item.name}/")
                else:
                    shutil.copy2(item, dest)
                    print(f"  Copied: {item.name}")
        else:
            # If cache_path is a file, copy just that file
            shutil.copy2(cache_path, target_dir / cache_path.name)
        
        print()
        print("=" * 60)
        print("Download Complete!")
        print("=" * 60)
        print(f"Dataset location: {target_dir.absolute()}")
        print()
        
        # Show structure
        print("Dataset structure:")
        videos_dir = target_dir / "videos"
        if videos_dir.exists():
            action_classes = list(videos_dir.iterdir())[:5]
            print(f"  {target_dir}/")
            print(f"    videos/")
            for action_class in action_classes:
                if action_class.is_dir():
                    num_videos = len(list(action_class.glob("*.avi")))
                    print(f"      {action_class.name}/ ({num_videos} videos)")
            print(f"      ... (101 action classes total)")
        else:
            print(f"  Check: {target_dir}")
        
        print()
        print("Next: Start training!")
        print("  python train.py \\")
        print("    --dataset ucf101 \\")
        print("    --data_root ./data/ucf101 \\")
        print("    --max_samples 10000 \\")
        print("    --num_epochs 30")
        print()
        
        return True
        
    except Exception as e:
        print(f"✗ Error: {e}")
        print()
        print("Troubleshooting:")
        print("1. Install kagglehub: pip install kagglehub")
        print("2. Make sure you're logged in to Kaggle")
        print("3. Accept the dataset terms on Kaggle website")
        print("   https://www.kaggle.com/datasets/abdallahwagih/ucf101-videos")
        return False


if __name__ == "__main__":
    download_ucf101()
