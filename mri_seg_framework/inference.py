from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import SimpleITK as sitk


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
    def __init__(
        self,
        task: str = "total_mr",
        fast: bool = False,
        ml: bool = True,
        roi_subset: Optional[list[str]] = None,
        device: str = "gpu",
        gpu_id: int = 0,
    ):
        self.task = task
        self.fast = fast
        self.ml = ml
        self.roi_subset = roi_subset
        self.device = device
        self.gpu_id = gpu_id

    def run(self, input_nifti: Path, output_seg_path: Path) -> Dict[int, str]:
        output_seg_path.parent.mkdir(parents=True, exist_ok=True)

        if self.device == "gpu":
            os.environ["CUDA_VISIBLE_DEVICES"] = str(self.gpu_id)

        masks_dir = output_seg_path.parent / "_tmp_totalseg_masks"
        if masks_dir.exists():
            shutil.rmtree(masks_dir, ignore_errors=True)
        masks_dir.mkdir(parents=True, exist_ok=True)
        try:
            self._run_totalsegmentator_cli(input_nifti, masks_dir)
            label_map = self._merge_masks_to_multilabel(masks_dir, output_seg_path)
            return label_map
        finally:
            shutil.rmtree(masks_dir, ignore_errors=True)

    def _run_totalsegmentator_cli(self, input_nifti: Path, masks_dir: Path) -> None:
        cmd = [
            "TotalSegmentator",
            "-i",
            str(input_nifti),
            "-o",
            str(masks_dir),
            "--task",
            self.task,
        ]
        if self.fast:
            cmd.append("--fast")
        if self.ml:
            cmd.append("--ml")
        if self.roi_subset:
            cmd.extend(["--roi_subset", *self.roi_subset])
        if self.device == "cpu":
            cmd.append("--device")
            cmd.append("cpu")

        subprocess.run(cmd, check=True)

    def _merge_masks_to_multilabel(self, masks_dir: Path, output_seg_path: Path) -> Dict[int, str]:
        class_map = _load_class_map(self.task)
        name_to_label = {v: k for k, v in class_map.items()}

        mask_files = sorted([p for p in masks_dir.glob("*.nii.gz") if p.is_file()])
        if not mask_files:
            raise RuntimeError(f"No masks produced by TotalSegmentator CLI in: {masks_dir}")

        ref = sitk.ReadImage(str(mask_files[0]))
        out_arr = np.zeros(sitk.GetArrayFromImage(ref).shape, dtype=np.uint16)
        dynamic_map: Dict[int, str] = {}
        next_label = max(class_map.keys(), default=0) + 1

        for mask_file in mask_files:
            organ_name = mask_file.name.replace(".nii.gz", "")
            mask_img = sitk.ReadImage(str(mask_file))
            mask_arr = sitk.GetArrayFromImage(mask_img) > 0
            if not np.any(mask_arr):
                continue

            label_id = name_to_label.get(organ_name)
            if label_id is None:
                label_id = next_label
                next_label += 1
            out_arr[mask_arr] = np.uint16(label_id)
            dynamic_map[int(label_id)] = organ_name

        out_img = sitk.GetImageFromArray(out_arr)
        out_img.CopyInformation(ref)
        sitk.WriteImage(out_img, str(output_seg_path))
        return dynamic_map
