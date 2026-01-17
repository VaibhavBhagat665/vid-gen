"""
Quick test script to verify training setup
Run this before starting actual training to catch issues early

Usage:
    python test_training_setup.py
"""

import sys
from pathlib import Path

print("=" * 60)
print("Text-to-Video Training Setup Verification")
print("=" * 60)
print()

# Test 1: Check Python version
print("1. Python Version")
print(f"   Version: {sys.version}")
print(f"   ✓ Python {sys.version_info.major}.{sys.version_info.minor}")
print()

# Test 2: Check PyTorch and MPS
print("2. PyTorch & MPS Backend")
try:
    import torch
    print(f"   PyTorch version: {torch.__version__}")
    
    if torch.backends.mps.is_available():
        print("   ✓ MPS backend available (Apple Silicon GPU)")
        
        # Test MPS
        try:
            x = torch.randn(10, 10).to("mps")
            y = x @ x.T
            print("   ✓ MPS backend working correctly")
        except Exception as e:
            print(f"   ✗ MPS test failed: {e}")
    else:
        print("   ⚠ MPS not available - will use CPU (very slow)")
        print("   Make sure you're on Mac with Apple Silicon")
except ImportError as e:
    print(f"   ✗ PyTorch not installed: {e}")
    sys.exit(1)
print()

# Test 3: Check required packages
print("3. Required Packages")
required_packages = [
    "diffusers",
    "transformers",
    "accelerate",
    "cv2",
    "PIL",
    "numpy",
    "tqdm",
]

all_installed = True
for package in required_packages:
    try:
        if package == "cv2":
            import cv2
            print(f"   ✓ opencv-python ({cv2.__version__})")
        elif package == "PIL":
            from PIL import Image
            print(f"   ✓ pillow")
        else:
            module = __import__(package)
            version = getattr(module, "__version__", "unknown")
            print(f"   ✓ {package} ({version})")
    except ImportError:
        print(f"   ✗ {package} - NOT INSTALLED")
        all_installed = False

if not all_installed:
    print("\n   Run: pip install -r requirements.txt")
    sys.exit(1)
print()

# Test 4: Check project structure
print("4. Project Structure")
required_files = [
    "train.py",
    "src/datasets/__init__.py",
    "src/datasets/hdvila.py",
    "src/datasets/webvid.py",
    "src/trainer.py",
    "config/training_config.py",
    "scripts/download_dataset.py",
]

for file in required_files:
    path = Path(file)
    if path.exists():
        print(f"   ✓ {file}")
    else:
        print(f"   ✗ {file} - MISSING")
print()

# Test 5: Check dataset loaders
print("5. Dataset Loaders")
try:
    from src.datasets import get_dataset, HDVILADataset, WebVidDataset
    print("   ✓ Dataset loaders imported successfully")
    
    # Test instantiation (will fail without data, but checks imports)
    print("   ✓ Dataset classes available")
except Exception as e:
    print(f"   ✗ Dataset import failed: {e}")
print()

# Test 6: Check trainer
print("6. Trainer Module")
try:
    from src.trainer import VideoTrainer
    print("   ✓ VideoTrainer imported successfully")
except Exception as e:
    print(f"   ✗ Trainer import failed: {e}")
print()

# Test 7: Check training config
print("7. Training Configuration")
try:
    from config.training_config import (
        DEVICE, DATASET_CONFIG, MODEL_CONFIG, TRAINING_CONFIG
    )
    print(f"   ✓ Config loaded")
    print(f"   Device: {DEVICE}")
    print(f"   Batch size: {TRAINING_CONFIG['batch_size']}")
    print(f"   Gradient accumulation: {TRAINING_CONFIG['gradient_accumulation_steps']}")
    print(f"   Effective batch size: {TRAINING_CONFIG['batch_size'] * TRAINING_CONFIG['gradient_accumulation_steps']}")
except Exception as e:
    print(f"   ✗ Config import failed: {e}")
print()

# Test 8: Memory check
print("8. System Memory")
try:
    import psutil
    memory = psutil.virtual_memory()
    print(f"   Total: {memory.total / (1024**3):.1f} GB")
    print(f"   Available: {memory.available / (1024**3):.1f} GB")
    
    if memory.available / (1024**3) < 8:
        print("   ⚠ Low memory - close other applications before training")
    else:
        print("   ✓ Sufficient memory available")
except ImportError:
    print("   (psutil not installed - skipping memory check)")
print()

# Summary
print("=" * 60)
print("Verification Complete!")
print("=" * 60)
print()
print("Next steps:")
print("1. Create a test dataset:")
print("   python scripts/download_dataset.py --output_dir ./data/test --create_sample --num_samples 10")
print()
print("2. Test training:")
print("   python train.py --data_root ./data/test --max_samples 10 --num_epochs 1")
print()
print("3. Monitor with TensorBoard:")
print("   tensorboard --logdir ./logs")
print()
