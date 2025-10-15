# Installation Guide

This guide covers different ways to install BehavERT.

## Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU (recommended for training)
- 16GB+ RAM (for large datasets)

## Quick Installation

### Using pip (Recommended)

```bash
pip install behavert
```

### From Source

```bash
git clone https://github.com/yourusername/BehavERT.git
cd BehavERT
pip install -e .
```

## Development Installation

For development and contributing:

```bash
git clone https://github.com/yourusername/BehavERT.git
cd BehavERT
pip install -e ".[dev]"
pre-commit install
```

## Docker Installation

### Using Pre-built Image

```bash
docker pull behavert/behavert:latest
docker run --gpus all -it behavert/behavert:latest
```

### Building from Source

```bash
git clone https://github.com/yourusername/BehavERT.git
cd BehavERT
docker build -t behavert .
docker run --gpus all -it behavert
```

## Conda Installation

```bash
conda create -n behavert python=3.9
conda activate behavert
pip install behavert
```

## GPU Support

BehavERT requires PyTorch with CUDA support for GPU acceleration:

```bash
# For CUDA 11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

## Verification

Test your installation:

```python
import behavert
print(behavert.__version__)

# Test GPU availability
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU count: {torch.cuda.device_count()}")
```

## Troubleshooting

### Common Issues

1. **CUDA out of memory**: Reduce batch size or sequence length
2. **Import errors**: Check Python version and dependencies
3. **Slow training**: Ensure GPU is being used

### Getting Help

- Check the [FAQ](faq.md)
- Search existing [GitHub Issues](https://github.com/yourusername/BehavERT/issues)
- Create a new issue with detailed information
