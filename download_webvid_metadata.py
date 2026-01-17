"""
Download WebVid-10M Dataset Metadata
This script downloads the first 10,000 entries from WebVid-10M dataset

Usage:
    python download_webvid_metadata.py
"""

import os
from datasets import load_dataset
import pandas as pd


def download_webvid_metadata(num_samples=10000):
    """Download WebVid metadata from Hugging Face."""
    
    print("=" * 60)
    print("WebVid-10M Metadata Downloader")
    print("=" * 60)
    print()
    
    # Create output directory
    output_dir = "./data/webvid"
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")
    print()
    
    # Download dataset
    print(f"Downloading {num_samples} entries from WebVid-10M...")
    print("This may take a few minutes...")
    print()
    
    try:
        # Load dataset from Hugging Face
        dataset = load_dataset(
            "iejMac/video-dataset-gpt-annotations-10M",
            split=f"train[:{num_samples}]"
        )
        
        print(f"✓ Downloaded {len(dataset)} entries")
        print()
        
        # Convert to DataFrame
        df = pd.DataFrame(dataset)
        
        # Check required columns
        print("Dataset columns:", list(df.columns))
        print()
        
        # Rename columns to match expected format
        if 'videoid' in df.columns:
            # Already has correct column names
            pass
        elif 'id' in df.columns:
            df = df.rename(columns={'id': 'videoid'})
        
        # Ensure required columns exist
        required_cols = ['videoid', 'name']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            print(f"⚠️  Warning: Missing columns: {missing_cols}")
            print("   Will use available columns")
        
        # Add page_dir if not present (WebVid uses this for organizing videos)
        if 'page_dir' not in df.columns:
            # Create page directories (e.g., page_000, page_001, etc.)
            df['page_dir'] = df.index // 1000
            df['page_dir'] = df['page_dir'].apply(lambda x: f"page_{x:03d}")
        
        # Add split column for train/val
        df['split'] = 'train'
        # Use last 500 as validation
        df.loc[df.index >= len(df) - 500, 'split'] = 'val'
        
        # Save to CSV
        output_file = os.path.join(output_dir, "results_10M_train.csv")
        df.to_csv(output_file, index=False)
        
        print(f"✓ Saved metadata to: {output_file}")
        print()
        
        # Show sample
        print("Sample entries:")
        print(df.head(3)[['videoid', 'name']].to_string(index=False))
        print()
        
        # Summary
        print("=" * 60)
        print("Download Complete!")
        print("=" * 60)
        print(f"Total entries: {len(df)}")
        print(f"Training samples: {len(df[df['split'] == 'train'])}")
        print(f"Validation samples: {len(df[df['split'] == 'val'])}")
        print()
        print("Next steps:")
        print("1. Download actual videos using:")
        print(f"   python scripts/download_dataset.py \\")
        print(f"     --dataset webvid \\")
        print(f"     --output_dir ./data/webvid \\")
        print(f"     --metadata_csv {output_file} \\")
        print(f"     --num_samples {num_samples}")
        print()
        print("2. Or start training with existing videos:")
        print("   python train.py --dataset webvid --data_root ./data/webvid")
        
        return True
        
    except Exception as e:
        print(f"✗ Error: {e}")
        print()
        print("Troubleshooting:")
        print("1. Make sure 'datasets' is installed: pip install datasets")
        print("2. Check internet connection")
        print("3. Try with fewer samples: python download_webvid_metadata.py")
        return False


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Download WebVid metadata")
    parser.add_argument(
        "--num_samples",
        type=int,
        default=10000,
        help="Number of samples to download (default: 10000)"
    )
    
    args = parser.parse_args()
    
    download_webvid_metadata(args.num_samples)
