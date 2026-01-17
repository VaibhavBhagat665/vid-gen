# Text-to-Video Generation 🎬

Generate videos from text prompts locally using state-of-the-art diffusion models.

**NEW**: Now with **training support** using HD-VILA-100M dataset (371K hours)! Fine-tune models on Mac with Apple Silicon.

## Features

- ✅ Generate videos from text prompts
- ✅ Multiple model support (Zeroscope, ModelScope)
- ✅ Web UI with Gradio
- ✅ Command-line interface
- ✅ **[NEW] Training/Fine-tuning on custom datasets**
- ✅ **[NEW] Mac Apple Silicon (MPS) optimization**
- ✅ **[NEW] HD-VILA-100M dataset integration**

## Quick Start (Inference)

### Installation

```bash
git clone <your-repo>
cd text-to-vid
pip install -r requirements.txt
```

### Generate Videos

#### Web Interface
```bash
python main.py --ui
```

#### Command Line
```bash
python main.py --prompt "a cat playing with yarn" --model zeroscope
```

## Training (NEW!) 🔥

### For Mac Users

Fine-tune text-to-video models on your friend's Mac with Apple Silicon:

```bash
# 1. Setup
pip install -r requirements.txt

# 2. Create test dataset
python scripts/download_dataset.py \
  --output_dir ./data/hdvila_test \
  --create_sample \
  --num_samples 100

# 3. Start training
python train.py \
  --data_root ./data/hdvila_test \
  --max_samples 100 \
  --num_epochs 5 \
  --batch_size 1 \
  --width 256 \
  --height 256
```

### Full Documentation

See **[TRAINING_GUIDE.md](TRAINING_GUIDE.md)** for complete training instructions.

## Project Structure

```
text-to-vid/
├── main.py                 # Inference script
├── train.py               # Training script (NEW)
├── requirements.txt
├── TRAINING_GUIDE.md      # Training documentation (NEW)
├── config/
│   ├── settings.py        # Inference config
│   └── training_config.py # Training config (NEW)
├── src/
│   ├── datasets/          # Dataset loaders (NEW)
│   │   ├── hdvila.py      # HD-VILA-100M
│   │   └── webvid.py      # WebVid
│   ├── model_loader.py
│   ├── video_generator.py
│   ├── trainer.py         # Training loop (NEW)
│   └── ui_interface.py
├── scripts/
│   └── download_dataset.py # Dataset downloader (NEW)
└── outputs/               # Generated videos
```

## Models

### Pre-trained (Inference)
- **Zeroscope v2**: 576p, high quality
- **ModelScope**: 256p, faster generation

### Custom (Training)
Fine-tune on your own data using:
- **HD-VILA-100M**: 371K hours, 720p, 15 categories (RECOMMENDED)
- **WebVid-10M**: 10.7M clips, general purpose
- **Your own videos**: Custom dataset support

## Requirements

### For Inference
- Python 3.8+
- 8GB+ RAM
- CUDA GPU (recommended) or CPU

### For Training
- Mac with Apple Silicon (M1/M2/M3) **OR**
- NVIDIA GPU with 16GB+ VRAM
- PyTorch 2.0+ with MPS/CUDA support
- 100GB+ free storage (for dataset)

## Training Performance

On Mac M1/M2/M3:
- **Speed**: ~5-20 seconds per training step
- **Memory**: 8-16GB unified memory
- **Dataset**: Start with 1K videos (20GB), scale to 50K+ (1TB+)
- **Quality**: Noticeable improvements after 10K samples

## Examples

### Inference
```python
from src.model_loader import ModelManager
from src.video_generator import VideoGenerator

# Load model
model_manager = ModelManager()
model = model_manager.load_model("zeroscope")

# Generate video
generator = VideoGenerator(model, "zeroscope")
video_path = generator.generate_video(
    prompt="a beautiful sunset over the ocean",
    output_path="outputs/sunset.mp4",
    duration=3,
    width=576,
    height=320
)
```

### Training
```bash
# Train on 10K HD-VILA samples for 20 epochs
python train.py \
  --data_root ./data/hdvila \
  --max_samples 10000 \
  --num_epochs 20 \
  --batch_size 1 \
  --gradient_accumulation_steps 16 \
  --width 320 \
  --height 320 \
  --num_frames 16

# Monitor with TensorBoard
tensorboard --logdir ./logs
```

## Training Datasets

### HD-VILA-100M (Recommended)
- **Size**: 371,500 hours
- **Resolution**: 720p
- **Categories**: 15 YouTube categories
- **Download**: See [TRAINING_GUIDE.md](TRAINING_GUIDE.md)

### WebVid-10M
- **Size**: 52,000 hours
- **Clips**: 10.7M video-caption pairs
- **Note**: Original distribution unavailable, academic use only

## License & Citation

This project uses:
- Zeroscope v2 by cerspense
- ModelScope by damo-vilab
- HD-VILA-100M by Microsoft Research

For academic use of datasets, please cite the original papers.

## Contributing

Contributions welcome! Areas of interest:
- [ ] Additional dataset loaders
- [ ] Multi-GPU training support
- [ ] Better evaluation metrics
- [ ] Prompt engineering tools

## Troubleshooting

See [TRAINING_GUIDE.md](TRAINING_GUIDE.md) for common issues and solutions.

---

**Ready to train?** Check out [TRAINING_GUIDE.md](TRAINING_GUIDE.md) for the complete walkthrough!
