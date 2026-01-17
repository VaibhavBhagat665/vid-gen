"""
Download Video Dataset Metadata (WORKING DATASETS ONLY)
Supports: MSR-VTT (easiest, 10K videos) and WebVid metadata

Usage:
    # MSR-VTT (recommended - small, complete, ready to use)
    python download_webvid_metadata.py --dataset msrvtt
    
    # WebVid metadata only (you'll need to download videos separately)
    python download_webvid_metadata.py --dataset webvid --num_samples 10000
"""

import os
import argparse
from datasets import load_dataset
import pandas as pd


def download_msrvtt():
    """
    Download MSR-VTT dataset (RECOMMENDED - Actually works!)
    - 10K videos already available
    - Smaller size (~20GB total)
    - Complete and ready to use
    """
    print("=" * 60)
    print("MSR-VTT Dataset Download")
    print("=" * 60)
    print()
    print("Dataset: MSR-VTT (Microsoft Research Video to Text)")
    print("Size: 10,000 videos")
    print("Categories: 20 types")
    print("Annotations: 200K clip-sentence pairs")
    print()
    
    output_dir = "./data/msrvtt"
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        print("Downloading from Hugging Face: friedrichor/MSR-VTT")
        print("This may take a few minutes...")
        print()
        
        # Load MSR-VTT from Hugging Face with proper config
        # Available configs: train_9k, train_7k, etc.
        dataset = load_dataset("friedrichor/MSR-VTT", "train_9k", split="train", trust_remote_code=True)
        
        print(f"✓ Downloaded {len(dataset)} entries")
        print()
        
        # Convert to DataFrame
        df = pd.DataFrame(dataset)
        print("Columns:", list(df.columns))
        print()
        
        # Create metadata CSV in expected format
        metadata = []
        for idx, row in df.iterrows():
            video_id = row.get('video_id', f'video{idx}')
            caption = row.get('caption', row.get('sentences', [''])[0] if 'sentences' in row else '')
            
            metadata.append({
                'video_id': video_id,
                'videoid': video_id,
                'caption': caption,
                'name': caption,
                'duration': 10.0,  # MSR-VTT videos are ~10 seconds
                'category': row.get('category', 'unknown'),
                'split': 'train' if idx < 9000 else 'val'
            })
        
        meta_df = pd.DataFrame(metadata)
        
        # Save metadata
        output_file = os.path.join(output_dir, "metadata.csv")
        meta_df.to_csv(output_file, index=False)
        
        print(f"✓ Saved metadata to: {output_file}")
        print()
        
        # Summary
        print("=" * 60)
        print("MSR-VTT Metadata Ready!")
        print("=" * 60)
        print(f"Training samples: {len(meta_df[meta_df['split']=='train'])}")
        print(f"Validation samples: {len(meta_df[meta_df['split']=='val'])}")
        print()
        print("Next: Download actual videos")
        print("  Option 1: From Kaggle (easiest)")
        print("    https://www.kaggle.com/datasets/vishnutheepb/msrvtt")
        print()
        print("  Option 2: Use video downloader")
        print("    python scripts/download_dataset.py \\")
        print(f"      --metadata_csv {output_file} \\")
        print("      --output_dir ./data/msrvtt")
        print()
        
        return True
        
    except Exception as e:
        print(f"✗ Error: {e}")
        print()
        print("Make sure 'datasets' is installed: pip install datasets")
        return False


def download_webvid_metadata(num_samples=10000):
    """
    Download WebVid-10M metadata (metadata only, videos separate)
    """
    print("=" * 60)
    print("WebVid-10M Metadata Download")
    print("=" * 60)
    print()
    
    output_dir = "./data/webvid"
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        print(f"Downloading {num_samples} entries from WebVid...")
        print("Source: TempoFunk/webvid-10M (metadata only)")
        print()
        
        # Load WebVid metadata from Hugging Face
        dataset = load_dataset(
            "TempoFunk/webvid-10M",
            split=f"train[:{num_samples}]",
            trust_remote_code=True
        )
        
        print(f"✓ Downloaded {len(dataset)} entries")
        print()
        
        # Convert to DataFrame
        df = pd.DataFrame(dataset)
        print("Columns:", list(df.columns))
        print()
        
        # Add split column
        df['split'] = 'train'
        df.loc[df.index >= len(df) - 500, 'split'] = 'val'
        
        # Add page_dir if not present
        if 'page_dir' not in df.columns:
            df['page_dir'] = df.index // 1000
            df['page_dir'] = df['page_dir'].apply(lambda x: f"page_{x:03d}")
        
        # Save
        output_file = os.path.join(output_dir, "results_10M_train.csv")
        df.to_csv(output_file, index=False)
        
        print(f"✓ Saved metadata to: {output_file}")
        print()
        
        # Summary
        print("=" * 60)
        print("WebVid Metadata Ready!")
        print("=" * 60)
        print(f"Entries: {len(df)}")
        print()
        print("⚠️  Note: This is METADATA ONLY")
        print("   You still need to download actual videos")
        print()
        print("Download videos with:")
        print("  python scripts/download_dataset.py \\")
        print(f"    --metadata_csv {output_file} \\")
        print("    --output_dir ./data/webvid \\")
        print("    --num_samples {num_samples}")
        print()
        
        return True
        
    except Exception as e:
        print(f"✗ Error: {e}")
        print()
        print("Troubleshooting:")
        print("1. Install: pip install datasets")
        print("2. Check internet connection")
        print("3. Try MSR-VTT instead (easier): --dataset msrvtt")
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download video dataset metadata")
    parser.add_argument(
        "--dataset",
        type=str,
        default="msrvtt",
        choices=["msrvtt", "webvid"],
        help="Dataset to download (msrvtt is recommended)"
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=10000,
        help="Number of samples (for WebVid only, MSR-VTT downloads all)"
    )
    
    args = parser.parse_args()
    
    print("\n" + "=" * 60)
    print("Video Dataset Metadata Downloader")
    print("=" * 60)
    print()
    
    if args.dataset == "msrvtt":
        print("✅ RECOMMENDED: MSR-VTT is easier and complete!")
        print()
        download_msrvtt()
    else:
        print("⚠️  WebVid downloads metadata only")
        print("   Consider MSR-VTT for easier setup")
        print()
        download_webvid_metadata(args.num_samples)

