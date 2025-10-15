"""
BehavERT: BERT-based Animal Behavior Analysis from Keypoint Data

A transformer-based deep learning framework for automated animal behavior analysis
using 3D keypoint trajectories.
"""

__version__ = "1.0.0"
__author__ = "BehavERT Team"
__email__ = "contact@behavert.org"

from .models import BehavERTModel
from .data import BehaviorDataset
from .training import BehavERTTrainer

__all__ = [
    "BehavERTModel",
    "BehaviorDataset", 
    "BehavERTTrainer",
]
