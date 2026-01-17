import torch
import numpy as np
from PIL import Image
import random
from .video_utils import frames_to_video, ensure_output_dir


class VideoGenerator:
    def __init__(self, pipeline, model_name):
        self.pipeline = pipeline
        self.model_name = model_name
        self.device = pipeline.device if hasattr(pipeline, 'device') else "cuda" if torch.cuda.is_available() else "cpu"
    
    def generate_video(self, prompt, output_path, duration=3, width=512, height=512, 
                      fps=8, num_inference_steps=20, seed=None):
        
        ensure_output_dir(output_path)
        
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)
        
        num_frames = duration * fps
        
        print(f"Generating {num_frames} frames at {width}x{height}")
        print(f"Using {num_inference_steps} inference steps")
        
        try:
            # Generate video frames
            if self.model_name == "zeroscope":
                video_frames = self._generate_zeroscope(
                    prompt, num_frames, width, height, num_inference_steps
                )
            elif self.model_name == "modelscope":
                video_frames = self._generate_modelscope(
                    prompt, num_frames, width, height, num_inference_steps
                )
            else:
                raise ValueError(f"Unknown model: {self.model_name}")
            
            # Convert to video file
            video_path = frames_to_video(video_frames, output_path, fps)
            return video_path
            
        except torch.cuda.OutOfMemoryError:
            print("CUDA out of memory! Try reducing resolution or duration.")
            raise
        except Exception as e:
            print(f"Generation failed: {str(e)}")
            raise
    
    def _generate_zeroscope(self, prompt, num_frames, width, height, steps):
        device_type = "cuda" if self.device == "cuda" or str(self.device).startswith("cuda") else "cpu"
        
        if device_type == "cuda":
            with torch.autocast("cuda", dtype=torch.float16):
                video = self.pipeline(
                    prompt,
                    num_frames=num_frames,
                    height=height,
                    width=width,
                    num_inference_steps=steps,
                    guidance_scale=17.5,
                ).frames[0]
        else:
            raw_output = self.pipeline(
                prompt,
                num_frames=num_frames,
                height=height,
                width=width,
                num_inference_steps=steps,
                guidance_scale=17.5,
            )

            video_array = raw_output.frames[0]

            frames = []
            if isinstance(video_array, np.ndarray):
                for i in range(video_array.shape[0]):
                    frame = video_array[i]
                if frame.max() <= 1.0:
                    frame = (frame * 255).astype(np.uint8)
                    img = Image.fromarray(frame)
                    frames.append(img)
                else:
                    frames = video_array  

            return frames

    
    def _generate_modelscope(self, prompt, num_frames, width, height, steps):
        device_type = "cuda" if self.device == "cuda" or str(self.device).startswith("cuda") else "cpu"
        
        if device_type == "cuda":
            with torch.autocast("cuda", dtype=torch.float16):
                video = self.pipeline(
                    prompt,
                    num_frames=num_frames,
                    height=height,
                    width=width,
                    num_inference_steps=steps,
                ).frames[0]
        else:
            video = self.pipeline(
                prompt,
                num_frames=num_frames,
                height=height,
                width=width,
                num_inference_steps=steps,
            ).frames[0]
        
        return video
