"""
Fix UCF-101 Dataset in Colab
Run this in a Colab cell to properly download and setup the dataset

Copy-paste this entire code into a Colab cell and run it
"""

import kagglehub
import shutil
from pathlib import Path

print("=" * 60)
print("UCF-101 Dataset Setup for Colab")
print("=" * 60)
print()

# Download using kagglehub
print("📥 Downloading UCF-101 from Kaggle...")
print("   This may take 10-15 minutes")
print()

cache_path = kagglehub.dataset_download("abdallahwagih/ucf101-videos")
cache_path = Path(cache_path)

print(f"✓ Downloaded to: {cache_path}")
print()

# Create target directory
target_dir = Path('./data/ucf101')
target_dir.mkdir(parents=True, exist_ok=True)

print("📦 Copying to project directory...")
print(f"   Target: {target_dir.absolute()}")
print()

# Copy all contents
for item in cache_path.iterdir():
    dest = target_dir / item.name
    
    if item.is_dir():
        print(f"  Copying folder: {item.name}/")
        if dest.exists():
            shutil.rmtree(dest)
        shutil.copytree(item, dest)
    else:
        print(f"  Copying file: {item.name}")
        shutil.copy2(item, dest)

print()

# Verify structure
videos_dir = target_dir / 'videos'
if videos_dir.exists():
    categories = [d for d in videos_dir.iterdir() if d.is_dir()]
    num_videos = sum(len(list(cat.glob('*.avi'))) for cat in categories)
    
    print("=" * 60)
    print("✅ Dataset Setup Complete!")
    print("=" * 60)
    print(f"Location: {target_dir.absolute()}")
    print(f"Action categories: {len(categories)}")
    print(f"Total videos: {num_videos:,}")
    print()
    print("Ready to train! Run the training cell now.")
else:
    print("=" * 60)
    print("❌ Error: videos/ directory not found")
    print("=" * 60)
    print()
    print("Dataset structure:")
    for item in target_dir.iterdir():
        print(f"  - {item.name}")
    print()
    print("Expected: videos/ folder with .avi files")
