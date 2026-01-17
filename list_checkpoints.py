"""
Quick script to list available checkpoints

Usage:
    python list_checkpoints.py
"""

from pathlib import Path

def list_checkpoints():
    """List all available checkpoints."""
    print("=" * 60)
    print("Available Checkpoints")
    print("=" * 60)
    print()
    
    checkpoints_dir = Path("./checkpoints")
    
    if not checkpoints_dir.exists():
        print("❌ No checkpoints directory found!")
        print("   Make sure you've run training first.")
        return
    
    checkpoints = sorted(checkpoints_dir.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True)
    
    if not checkpoints:
        print("❌ No checkpoints found in ./checkpoints/")
        print("   Make sure training completed and saved checkpoints.")
        return
    
    print(f"Found {len(checkpoints)} checkpoint(s):\n")
    
    for i, cp in enumerate(checkpoints, 1):
        if cp.is_dir():
            size = sum(f.stat().st_size for f in cp.rglob('*') if f.is_file())
            size_gb = size / (1024**3)
            mtime = cp.stat().st_mtime
            from datetime import datetime
            mod_time = datetime.fromtimestamp(mtime).strftime('%Y-%m-%d %H:%M:%S')
            
            print(f"{i}. {cp.name}")
            print(f"   Path: {cp}")
            print(f"   Size: {size_gb:.2f} GB")
            print(f"   Modified: {mod_time}")
            print()
    
    print("=" * 60)
    print("To generate videos, use:")
    print(f"  python generate_finetuned.py --checkpoint {checkpoints[0]} --prompt 'your prompt'")
    print("=" * 60)


if __name__ == "__main__":
    list_checkpoints()
