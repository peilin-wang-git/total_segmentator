from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import yaml


SUPPORTED_EXTENSIONS = [".nii", ".nii.gz", ".mha", ".nrrd"]


@dataclass
class SegmentationConfig:
    input_dir: Path
    output_dir: Path
    task: str = "total_mr"
    roi_subset: Optional[List[str]] = None
    fast: bool = False
    ml: bool = True
    preview: bool = True
    num_threads: int = 1
    keep_temp: bool = False
    dry_run: bool = False
    supported_extensions: List[str] = field(default_factory=lambda: SUPPORTED_EXTENSIONS.copy())

    @classmethod
    def from_yaml(cls, path: Path, input_dir: Optional[Path] = None, output_dir: Optional[Path] = None) -> "SegmentationConfig":
        with path.open("r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}

        if input_dir is not None:
            raw["input_dir"] = str(input_dir)
        if output_dir is not None:
            raw["output_dir"] = str(output_dir)

        if "input_dir" not in raw or "output_dir" not in raw:
            raise ValueError("Config must define input_dir and output_dir, or pass them via CLI.")

        raw["input_dir"] = Path(raw["input_dir"])
        raw["output_dir"] = Path(raw["output_dir"])
        return cls(**raw)
