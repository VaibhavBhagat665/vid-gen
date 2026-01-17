DEFAULT_CONFIG = {
    "models": {
        "zeroscope": {
            "model_id": "cerspense/zeroscope_v2_576w",
            "default_width": 576,
            "default_height": 320,
            "max_frames": 24
        },
        "modelscope": {
            "model_id": "damo-vilab/text-to-video-ms-1.7b", 
            "default_width": 256,
            "default_height": 256,
            "max_frames": 16
        }
    },
    
    "generation": {
        "default_fps": 8,
        "default_duration": 3,
        "default_steps": 20,
        "guidance_scale": 17.5
    },
    
    "output": {
        "format": "mp4",
        "codec": "mp4v",
        "quality": "high"
    },
    
    "memory": {
        "enable_attention_slicing": True,
        "enable_vae_slicing": True,
        "low_vram_mode": False
    }
}