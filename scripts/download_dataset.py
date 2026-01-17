"""
Dataset Download Script
Download HD-VILA-100M or WebVid datasets

Usage:
    python scripts/download_dataset.py --dataset hdvila --output_dir ./data/hdvila --num_samples 1000
"""

import argparse
import os
import csv
import subprocess
from pathlib import Path
from tqdm import tqdm
import requests


def download_hdvila_metadata(output_dir: Path):
    """
    Download HD-VILA-100M metadata.
    
    Note: You'll need to get the metadata CSV from the official source:
    https://github.com/microsoft/XPretrain/tree/main/hd-vila-100m
    """
    print("=" * 60)
    print("HD-VILA-100M Metadata Download")
    print("=" * 60)
    print()
    print("To download HD-VILA-100M metadata, please:")
    print("1. Visit: https://github.com/microsoft/XPretrain/tree/main/hd-vila-100m")
    print("2. Download the metadata CSV files")
    print(f"3. Place them in: {output_dir}/")
    print()
    print("The metadata files typically include:")
    print("  - hdvila_100m.csv (full dataset)")
    print("  - Category-specific CSV files")
    print()
    print("Once downloaded, the dataset structure should be:")
    print(f"{output_dir}/")
    print("├── metadata.csv (rename from downloaded file)")
    print("└── videos/ (will be created during download)")
    print()


def download_videos_from_csv(
    csv_path: Path,
    output_dir: Path,
    max_samples: int = None,
    resolution: str = "720"
):
    """
    Download videos from CSV using yt-dlp.
    
    Args:
        csv_path: Path to metadata CSV
        output_dir: Directory to save videos
        max_samples: Maximum number of videos to download
        resolution: Video resolution (720, 480, 360)
    """
    videos_dir = output_dir / "videos"
    videos_dir.mkdir(parents=True, exist_ok=True)
    
    # Check if yt-dlp is installed
    try:
        subprocess.run(["yt-dlp", "--version"], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("❌ Error: yt-dlp not found!")
        print("Please install it with: pip install yt-dlp")
        print("Or on Mac with Homebrew: brew install yt-dlp")
        return
    
    # Read CSV
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    
    if max_samples:
        rows = rows[:max_samples]
    
    print(f"Downloading {len(rows)} videos...")
    print(f"Output directory: {videos_dir}")
    print()
    
    success_count = 0
    fail_count = 0
    
    for row in tqdm(rows, desc="Downloading videos"):
        video_id = row.get('video_id', row.get('videoid', ''))
        
        # Construct YouTube URL
        # Note: HD-VILA uses YouTube video IDs
        url = f"https://www.youtube.com/watch?v={video_id}"
        
        output_path = videos_dir / f"{video_id}.mp4"
        
        # Skip if already downloaded
        if output_path.exists():
            success_count += 1
            continue
        
        # Download with yt-dlp
        cmd = [
            "yt-dlp",
            "-f", f"bestvideo[height<={resolution}][ext=mp4]+bestaudio[ext=m4a]/best[height<={resolution}][ext=mp4]",
            "--merge-output-format", "mp4",
            "-o", str(output_path),
            "--quiet",
            "--no-warnings",
            url
        ]
        
        try:
            subprocess.run(cmd, check=True, timeout=60)
            success_count += 1
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
            fail_count += 1
            # Remove partial file if exists
            if output_path.exists():
                output_path.unlink()
    
    print()
    print("=" * 60)
    print("Download Complete!")
    print("=" * 60)
    print(f"✓ Successfully downloaded: {success_count} videos")
    print(f"✗ Failed downloads: {fail_count} videos")
    print(f"📁 Videos saved to: {videos_dir}")


def create_sample_metadata(output_dir: Path, num_samples: int = 10):
    """
    Create a sample metadata CSV for testing.
    
    This creates dummy entries - replace with real data for actual training.
    """
    metadata_path = output_dir / "metadata.csv"
    
    print(f"Creating sample metadata with {num_samples} entries...")
    
    with open(metadata_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['video_id', 'caption', 'duration', 'category', 'split'])
        writer.writeheader()
        
        for i in range(num_samples):
            writer.writerow({
                'video_id': f'sample_{i:04d}',
                'caption': f'Sample video {i}',
                'duration': 10.0,
                'category': 'test',
                'split': 'train' if i < num_samples * 0.8 else 'val'
            })
    
    print(f"✓ Sample metadata created: {metadata_path}")
    print()
    print("⚠️  Note: This is dummy data for testing only!")
    print("   For real training, download actual HD-VILA-100M metadata.")


def main():
    parser = argparse.ArgumentParser(description="Download video dataset")
    parser.add_argument("--dataset", type=str, default="hdvila",
                       choices=["hdvila", "webvid"], help="Dataset to download")
    parser.add_argument("--output_dir", type=str, required=True,
                       help="Output directory for dataset")
    parser.add_argument("--metadata_csv", type=str, default=None,
                       help="Path to metadata CSV (if already downloaded)")
    parser.add_argument("--num_samples", type=int, default=None,
                       help="Number of videos to download (None = all)")
    parser.add_argument("--resolution", type=str, default="720",
                       choices=["1080", "720", "480", "360"],
                       help="Video resolution to download")
    parser.add_argument("--create_sample", action="store_true",
                       help="Create sample metadata for testing")
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if args.create_sample:
        create_sample_metadata(output_dir, args.num_samples or 10)
        return
    
    if args.dataset == "hdvila":
        if not args.metadata_csv:
            download_hdvila_metadata(output_dir)
        else:
            download_videos_from_csv(
                Path(args.metadata_csv),
                output_dir,
                args.num_samples,
                args.resolution
            )
    else:
        print(f"Dataset '{args.dataset}' download not yet implemented.")
        print("Please download WebVid manually from:")
        print("https://github.com/m-bain/webvid")


if __name__ == "__main__":
    main()
