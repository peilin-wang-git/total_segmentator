#!/usr/bin/env python
"""Trigger TotalSegmentator weight download for the selected task."""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

import numpy as np
import SimpleITK as sitk


def create_dummy_nifti(path: Path) -> None:
    arr = np.zeros((16, 16, 16), dtype=np.float32)
    img = sitk.GetImageFromArray(arr)
    sitk.WriteImage(img, str(path))


def main() -> None:
    parser = argparse.ArgumentParser(description="Download TotalSegmentator weights by running a warm-up inference.")
    parser.add_argument("--task", type=str, default="total_mr", help="Task name, e.g. total_mr")
    args = parser.parse_args()

    from totalsegmentator.python_api import totalsegmentator

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        input_path = tmp_path / "dummy.nii.gz"
        output_path = tmp_path / "dummy_seg.nii.gz"
        create_dummy_nifti(input_path)
        try:
            totalsegmentator(
                input=str(input_path),
                output=str(output_path),
                task=args.task,
                ml=True,
                fast=True,
                quiet=True,
            )
        except Exception as exc:
            print(
                "Model download/warmup attempted. If this failed due to tiny dummy input, "
                f"weights may still be cached. Details: {exc}"
            )

    print("Done. Check ~/.totalsegmentator for downloaded weights.")


if __name__ == "__main__":
    main()
