from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import SimpleITK as sitk


def clean_small_components(seg_path: Path, min_voxels: int = 20) -> None:
    image = sitk.ReadImage(str(seg_path))
    arr = sitk.GetArrayFromImage(image)
    if arr.max() == 0:
        return

    output = arr.copy()
    for label in sorted(set(arr.flatten().tolist())):
        if label == 0:
            continue
        binary = (arr == label).astype("uint8")
        cc = sitk.ConnectedComponent(sitk.GetImageFromArray(binary))
        stats = sitk.LabelShapeStatisticsImageFilter()
        stats.Execute(cc)
        for component_id in stats.GetLabels():
            if stats.GetNumberOfPixels(component_id) < min_voxels:
                component_arr = sitk.GetArrayFromImage(cc) == component_id
                output[component_arr] = 0

    out_img = sitk.GetImageFromArray(output)
    out_img.CopyInformation(image)
    sitk.WriteImage(out_img, str(seg_path))


def save_label_map(mapping: Dict[int, str], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump({str(k): v for k, v in sorted(mapping.items())}, f, ensure_ascii=False, indent=2)
