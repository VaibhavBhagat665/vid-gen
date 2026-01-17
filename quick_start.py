#!/usr/bin/env python3
"""
Quick Start Script for Text-to-Video Training
This script guides you through the setup process step-by-step

Usage:
    python quick_start.py
"""

import os
import sys
import subprocess
from pathlib import Path

def run_command(cmd, description):
    """Run a command and handle errors."""
    print(f"\n🔧 {description}...")
    print(f"   Command: {cmd}")
    
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    
    if result.returncode == 0:
        print(f"   ✓ Success!")
        if result.stdout:
            print(f"   Output: {result.stdout[:200]}")
        return True
    else:
        print(f"   ✗ Failed!")
        if result.stderr:
            print(f"   Error: {result.stderr[:200]}")
        return False

def check_dependencies():
    """Check if required packages are installed."""
    print("\n" + "=" * 60)
    print("STEP 1: Checking Dependencies")
    print("=" * 60)
    
    try:
        import torch
        print(f"✓ PyTorch {torch.__version__} installed")
        
        if torch.backends.mps.is_available():
            print("✓ MPS backend available (Apple Silicon GPU)")
            return True
        else:
            print("⚠ MPS not available - will use CPU (slow)")
            response = input("Continue anyway? (y/n): ")
            return response.lower() == 'y'
    except ImportError:
        print("✗ PyTorch not installed!")
        print("\nPlease install requirements first:")
        print("  pip install -r requirements.txt")
        return False

def create_test_dataset():
    """Create a test dataset."""
    print("\n" + "=" * 60)
    print("STEP 2: Creating Test Dataset")
    print("=" * 60)
    
    data_dir = Path("./data/test")
    
    if data_dir.exists() and (data_dir / "metadata.csv").exists():
        print(f"✓ Test dataset already exists at {data_dir}")
        response = input("Recreate it? (y/n): ")
        if response.lower() != 'y':
            return True
    
    data_dir.mkdir(parents=True, exist_ok=True)
    
    return run_command(
        "python scripts/download_dataset.py --output_dir ./data/test --create_sample --num_samples 100",
        "Creating 100 sample entries"
    )

def run_verification():
    """Run the verification script."""
    print("\n" + "=" * 60)
    print("STEP 3: Verifying Setup")
    print("=" * 60)
    
    return run_command(
        "python test_training_setup.py",
        "Running verification tests"
    )

def run_quick_test():
    """Run a quick training test."""
    print("\n" + "=" * 60)
    print("STEP 4: Quick Training Test")
    print("=" * 60)
    
    print("\nThis will train on 10 samples for 1 epoch (~5-10 minutes)")
    response = input("Start quick test? (y/n): ")
    
    if response.lower() != 'y':
        print("Skipped. You can run it later with:")
        print("  python train.py --data_root ./data/test --max_samples 10 --num_epochs 1")
        return True
    
    return run_command(
        "python train.py --data_root ./data/test --max_samples 10 --num_epochs 1 --width 256 --height 256 --num_frames 8",
        "Running quick training test"
    )

def show_next_steps():
    """Show what to do next."""
    print("\n" + "=" * 60)
    print("🎉 Setup Complete!")
    print("=" * 60)
    
    print("\n📝 Next Steps:\n")
    
    print("1. Download Real Dataset (HD-VILA-100M)")
    print("   Start with 1,000 videos (~20GB):")
    print("   ")
    print("   a. Download metadata from:")
    print("      https://github.com/microsoft/XPretrain/tree/main/hd-vila-100m")
    print("   ")
    print("   b. Save to ./data/hdvila/metadata.csv")
    print("   ")
    print("   c. Download videos:")
    print("      python scripts/download_dataset.py \\")
    print("        --dataset hdvila \\")
    print("        --output_dir ./data/hdvila \\")
    print("        --metadata_csv ./data/hdvila/metadata.csv \\")
    print("        --num_samples 1000 \\")
    print("        --resolution 720")
    print()
    
    print("2. Start Real Training")
    print("   python train.py \\")
    print("     --data_root ./data/hdvila \\")
    print("     --max_samples 1000 \\")
    print("     --num_epochs 20 \\")
    print("     --batch_size 1 \\")
    print("     --gradient_accumulation_steps 16 \\")
    print("     --width 256 \\")
    print("     --height 256 \\")
    print("     --num_frames 16")
    print()
    
    print("3. Monitor Training")
    print("   tensorboard --logdir ./logs")
    print("   Open: http://localhost:6006")
    print()
    
    print("4. Read the Full Guide")
    print("   See TRAINING_GUIDE.md for detailed instructions")
    print()
    
    print("=" * 60)
    print("Happy Training! 🚀")
    print("=" * 60)

def main():
    print("=" * 60)
    print("Text-to-Video Training - Quick Start")
    print("=" * 60)
    print()
    print("This script will guide you through:")
    print("1. Checking dependencies")
    print("2. Creating a test dataset")
    print("3. Verifying the setup")
    print("4. Running a quick training test")
    print()
    
    response = input("Continue? (y/n): ")
    if response.lower() != 'y':
        print("Exiting. Run this script again when ready.")
        return
    
    # Step 1: Check dependencies
    if not check_dependencies():
        print("\n❌ Please install dependencies first:")
        print("   pip install -r requirements.txt")
        return
    
    # Step 2: Create test dataset
    if not create_test_dataset():
        print("\n❌ Failed to create test dataset")
        return
    
    # Step 3: Run verification
    if not run_verification():
        print("\n⚠ Verification had some issues, but you can continue")
    
    # Step 4: Quick test (optional)
    run_quick_test()
    
    # Show next steps
    show_next_steps()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        print("Run this script again to continue setup")
