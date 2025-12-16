"""
Garbage Segmentation Models Package

This package provides models and utilities for garbage detection and classification.
"""

from .yolo_segmentation import GarbageSegmentor, segment_garbage
from .mobilenet_classifier import GarbageClassifier
from .inference import GarbageDetectionPipeline, run_inference

__all__ = [
    'GarbageSegmentor',
    'GarbageClassifier', 
    'GarbageDetectionPipeline',
    'segment_garbage',
    'run_inference'
]

__version__ = '1.0.0'
