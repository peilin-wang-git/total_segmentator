from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional


def _load_class_map(task: str) -> Dict[int, str]:
    try:
        from totalsegmentator.map_to_binary import class_map  # type: ignore

        mapping = class_map.get(task, {})
        if mapping:
            sample_key = next(iter(mapping.keys()))
            if isinstance(sample_key, int):
                return {int(k): str(v) for k, v in mapping.items()}
            return {int(v): str(k) for k, v in mapping.items()}
    except Exception:
        pass
    return {}


class TotalSegmentatorRunner:
    def __init__(self, task: str = "total_mr", fast: bool = False, ml: bool = True, roi_subset: Optional[list[str]] = None):
        self.task = task
        self.fast = fast
        self.ml = ml
        self.roi_subset = roi_subset

    def run(self, input_nifti: Path, output_seg_path: Path) -> Dict[int, str]:
        output_seg_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            from totalsegmentator.python_api import totalsegmentator  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "TotalSegmentator is not installed. Install requirements.txt before running inference."
            ) from exc

        totalsegmentator(
            input=str(input_nifti),
            output=str(output_seg_path),
            task=self.task,
            fast=self.fast,
            ml=self.ml,
            roi_subset=self.roi_subset,
            nr_thr_resamp=1,
            nr_thr_saving=1,
            verbose=False,
        )
        return _load_class_map(self.task)
