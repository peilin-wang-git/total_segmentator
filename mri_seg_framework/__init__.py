"""MRI multi-organ segmentation framework based on TotalSegmentator."""

from .config import SegmentationConfig
from .pipeline import SegmentationPipeline

__all__ = ["SegmentationConfig", "SegmentationPipeline"]
