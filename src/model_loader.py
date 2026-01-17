import torch
from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler
import os
from huggingface_hub import hf_hub_download
import warnings
warnings.filterwarnings("ignore")


class ModelManager:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.loaded_models = {}
        
        print(f"Using device: {self.device}")
        if self.device == "cpu":
            print("Warning: No CUDA GPU detected. Generation will be very slow.")
    
    def load_model(self, model_name):
        if model_name in self.loaded_models:
            return self.loaded_models[model_name]
        
        print(f"Loading {model_name} model... This may take a while on first run.")
        
        try:
            if model_name == "zeroscope":
                pipe = self._load_zeroscope()
            elif model_name == "modelscope":
                pipe = self._load_modelscope()
            else:
                raise ValueError(f"Unknown model: {model_name}")
            
            self.loaded_models[model_name] = pipe
            print(f"Model {model_name} loaded successfully!")
            return pipe
            
        except Exception as e:
            print(f"Failed to load model {model_name}: {str(e)}")
            raise
    
    def _load_zeroscope(self):
        model_id = "cerspense/zeroscope_v2_576w"
        
        pipe = DiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
        )
        
        # Memory optimization
        if self.device == "cuda":
            pipe = pipe.to(self.device)
            pipe.enable_memory_efficient_attention()
            pipe.enable_vae_slicing()
        
        return pipe
    
    def _load_modelscope(self):
        model_id = "damo-vilab/text-to-video-ms-1.7b"
        
        pipe = DiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            variant="fp16" if self.device == "cuda" else None
        )
        
        if self.device == "cuda":
            pipe = pipe.to(self.device)
            pipe.enable_memory_efficient_attention()
            pipe.enable_vae_slicing()
        
        return pipe
