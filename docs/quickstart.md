# Quick Start Guide

This guide will get you up and running with BehavERT in minutes.

## Installation

```bash
pip install behavert
```

## Basic Usage

### 1. Load a Pre-trained Model

```python
from behavert import BehavERTModel

# Load pre-trained model
model = BehavERTModel.from_pretrained("behavert-calms21-base")
```

### 2. Prepare Your Data

```python
import torch
from behavert.data import normalize_keypoints

# Your keypoint data: [batch_size, sequence_length, num_keypoints * 3]
keypoints = torch.randn(1, 128, 21)  # Example: 7 keypoints × 3 coordinates

# Normalize keypoints
keypoints_norm = normalize_keypoints(keypoints)
```

### 3. Run Inference

```python
# Get behavior predictions
with torch.no_grad():
    outputs = model(keypoints_norm)
    predictions = outputs.logits.argmax(dim=-1)

print(f"Predicted behaviors: {predictions}")
```

## Training a Custom Model

### 1. Prepare Your Dataset

```python
from behavert.data import BehaviorDataset

dataset = BehaviorDataset(
    data_dir="/path/to/keypoints",
    label_dir="/path/to/labels",
    max_seq_length=128
)
```

### 2. Initialize Model and Trainer

```python
from behavert import BehavERTModel, BehavERTTrainer

model = BehavERTModel(
    input_dim=21,  # 7 keypoints × 3 coordinates
    num_classes=5,
    max_seq_length=128
)

trainer = BehavERTTrainer(
    model=model,
    train_dataset=dataset,
    learning_rate=1e-5,
    batch_size=16
)
```

### 3. Train the Model

```python
trainer.train(epochs=50)
```

## Pre-trained Models

BehavERT provides several pre-trained models:

- `behavert-calms21-base`: Trained on CalMS21 dataset
- `behavert-mabe22-social`: Trained on MABe22 for social behaviors
- `behavert-sbea-individual`: Trained on SBeA for individual behaviors

```python
# Load specific pre-trained model
model = BehavERTModel.from_pretrained("behavert-mabe22-social")
```

## Configuration

Use configuration files for complex setups:

```python
from behavert.utils import load_config

config = load_config("configs/calms21_training.yaml")
model = BehavERTModel(**config.model)
```

## Next Steps

- Read the [API Reference](api_reference.md) for detailed documentation
- Check out [example notebooks](../notebooks/) for more examples
- Learn about [dataset formats](datasets.md)
- Explore [advanced training techniques](advanced_training.md)
