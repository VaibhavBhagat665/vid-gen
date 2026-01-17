import cv2
import numpy as np
from PIL import Image
import os
from pathlib import Path


def frames_to_video(frames, output_path, fps):
    """Convert list of PIL Images to video file"""
    
    if frames is None or len(frames) == 0:
        raise ValueError("No frames provided")
    
    # Convert frames to numpy arrays
    frame_arrays = []
    for i, frame in enumerate(frames):
        try:
            if isinstance(frame, Image.Image):
                frame_array = np.array(frame)
            elif isinstance(frame, np.ndarray):
                frame_array = frame
            else:
                # Try to convert to numpy array
                frame_array = np.array(frame)
            
            # Ensure we have the right shape and data type
            if frame_array.dtype != np.uint8:
                if frame_array.max() <= 1.0:
                    frame_array = (frame_array * 255).astype(np.uint8)
                else:
                    frame_array = frame_array.astype(np.uint8)
            
            # Make sure it's 3D (H, W, C)
            if len(frame_array.shape) == 2:
                frame_array = np.stack([frame_array] * 3, axis=-1)
            elif len(frame_array.shape) == 3 and frame_array.shape[2] == 3:
                # Convert RGB to BGR for OpenCV
                frame_array = cv2.cvtColor(frame_array, cv2.COLOR_RGB2BGR)
            elif len(frame_array.shape) == 3 and frame_array.shape[2] == 4:
                # Handle RGBA by dropping alpha channel then convert to BGR
                frame_array = cv2.cvtColor(frame_array[:, :, :3], cv2.COLOR_RGB2BGR)
            
            frame_arrays.append(frame_array)
            
        except Exception as e:
            print(f"Error processing frame {i}: {e}")
            print(f"Frame type: {type(frame)}, shape: {getattr(frame, 'shape', 'no shape')}")
            raise
    
    if not frame_arrays:
        raise ValueError("No valid frames to process")

    print(f"Each frame shape: {[frame.shape for frame in frame_arrays]}")
    print(f"Frame dtype: {[frame.dtype for frame in frame_arrays]}")

    
    height, width = frame_arrays[0].shape[:2]
    print(f"Creating video with {len(frame_arrays)} frames at {width}x{height}")
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    if not out.isOpened():
        raise RuntimeError(f"Failed to open video writer for {output_path}")
    
    # Write frames
    for i, frame in enumerate(frame_arrays):
        success = out.write(frame)
        if not success:
            print(f"Warning: Failed to write frame {i}")
    
    out.release()
    print(f"Video saved to: {output_path}")
    return output_path


def ensure_output_dir(file_path):
    """Create output directory if it doesn't exist"""
    output_dir = os.path.dirname(file_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)


def get_video_info(video_path):
    """Get basic info about a video file"""
    cap = cv2.VideoCapture(video_path)
    
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    cap.release()
    
    duration = frame_count / fps if fps > 0 else 0
    
    return {
        'duration': duration,
        'fps': fps,
        'width': width,
        'height': height,
        'frame_count': frame_count
    }
