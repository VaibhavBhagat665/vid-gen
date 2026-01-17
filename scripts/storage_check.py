"""
Storage Usage Monitor for Training
Check how much space is being used and how much is available

Usage:
    python scripts/storage_check.py
"""

import os
from pathlib import Path
import shutil


def get_dir_size(path):
    """Calculate total size of a directory."""
    total = 0
    try:
        for entry in os.scandir(path):
            if entry.is_file():
                total += entry.stat().st_size
            elif entry.is_dir():
                total += get_dir_size(entry.path)
    except PermissionError:
        pass
    return total


def format_size(bytes_size):
    """Format bytes to human readable format."""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes_size < 1024.0:
            return f"{bytes_size:.1f} {unit}"
        bytes_size /= 1024.0
    return f"{bytes_size:.1f} PB"


def check_storage():
    """Check storage usage for training."""
    print("=" * 60)
    print("Storage Usage Monitor")
    print("=" * 60)
    print()
    
    # Check overall disk space
    total, used, free = shutil.disk_usage(".")
    print("💾 Disk Space:")
    print(f"   Total: {format_size(total)}")
    print(f"   Used:  {format_size(used)}")
    print(f"   Free:  {format_size(free)}")
    print()
    
    # Calculate storage budget
    storage_budget = 200 * (1024 ** 3)  # 200GB
    print(f"📊 Training Budget: {format_size(storage_budget)} (200GB)")
    print()
    
    # Check dataset size
    data_dir = Path("./data")
    if data_dir.exists():
        data_size = get_dir_size(data_dir)
        print(f"📹 Dataset (./data): {format_size(data_size)}")
        print(f"   Budget remaining: {format_size(storage_budget - data_size)}")
    else:
        print("📹 Dataset (./data): Not found")
        data_size = 0
    print()
    
    # Check checkpoints size
    checkpoint_dir = Path("./checkpoints")
    if checkpoint_dir.exists():
        checkpoint_size = get_dir_size(checkpoint_dir)
        num_checkpoints = len([d for d in checkpoint_dir.iterdir() if d.is_dir()])
        print(f"💾 Checkpoints (./checkpoints): {format_size(checkpoint_size)}")
        print(f"   Number of checkpoints: {num_checkpoints}")
        print(f"   Avg per checkpoint: {format_size(checkpoint_size / max(num_checkpoints, 1))}")
    else:
        print("💾 Checkpoints: Not found")
        checkpoint_size = 0
    print()
    
    # Check logs size
    logs_dir = Path("./logs")
    if logs_dir.exists():
        logs_size = get_dir_size(logs_dir)
        print(f"📝 Logs (./logs): {format_size(logs_size)}")
    else:
        print("📝 Logs: Not found")
        logs_size = 0
    print()
    
    # Total usage
    total_training = data_size + checkpoint_size + logs_size
    print("=" * 60)
    print(f"📊 Total Training Usage: {format_size(total_training)}")
    print(f"📊 Budget: {format_size(storage_budget)}")
    print(f"📊 Remaining: {format_size(storage_budget - total_training)}")
    print()
    
    # Warnings
    usage_percent = (total_training / storage_budget) * 100
    if usage_percent > 90:
        print("⚠️  WARNING: Using >90% of budget!")
        print("   Consider deleting old checkpoints or cleaning logs")
    elif usage_percent > 75:
        print("⚠️  Caution: Using >75% of budget")
    else:
        print(f"✓ Storage OK ({usage_percent:.1f}% of budget used)")
    print()
    
    # Recommendations
    if checkpoint_size > 30 * (1024 ** 3):  # >30GB
        print("💡 Tip: You have many checkpoints. Consider:")
        print("   - Keeping only last 3 checkpoints (auto-cleanup enabled)")
        print("   - Deleting old checkpoints manually")
        print()
    
    if free < 50 * (1024 ** 3):  # <50GB free
        print("⚠️  Low disk space! Free up space before training more.")
        print()
    
    print("=" * 60)


if __name__ == "__main__":
    check_storage()
