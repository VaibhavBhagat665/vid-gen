from setuptools import setup, find_packages

setup(
    name="localvidgen",
    version="1.0.0",
    description="Local text-to-video generation using open-source models",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0", 
        "diffusers>=0.21.0",
        "transformers>=4.30.0",
        "accelerate>=0.20.0",
        "opencv-python>=4.8.0",
        "pillow>=9.5.0",
        "numpy>=1.24.0",
        "gradio>=3.40.0",
        "huggingface-hub>=0.16.0",
        "safetensors>=0.3.0",
        "scipy>=1.10.0",
    ],
    entry_points={
        "console_scripts": [
            "localvidgen=main:main",
        ],
    },
)