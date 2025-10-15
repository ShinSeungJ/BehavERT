"""
BehavERT Data Processing

This module contains dataset classes and data processing utilities for various
animal behavior datasets.
"""

from .datasets import BehaviorDataset, CalMS21Dataset, MABe22Dataset, SBeADataset
from .preprocessing import normalize_keypoints, handle_missing_data, detect_outliers
from .augmentation import KeypointAugmentation, TemporalAugmentation

__all__ = [
    "BehaviorDataset",
    "CalMS21Dataset",
    "MABe22Dataset", 
    "SBeADataset",
    "normalize_keypoints",
    "handle_missing_data",
    "detect_outliers",
    "KeypointAugmentation",
    "TemporalAugmentation",
]
