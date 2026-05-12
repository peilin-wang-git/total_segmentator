from __future__ import annotations

from pathlib import Path

import numpy as np
import SimpleITK as sitk


def load_image(path: Path) -> sitk.Image:
    return sitk.ReadImage(str(path))


def orient_to_lps(image: sitk.Image) -> sitk.Image:
    orienter = sitk.DICOMOrientImageFilter()
    orienter.SetDesiredCoordinateOrientation("LPS")
    return orienter.Execute(image)


def save_as_nifti(image: sitk.Image, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(image, str(output_path))
    return output_path


def normalize_intensity(image: sitk.Image, method: str = "none") -> sitk.Image:
    method = (method or "none").lower()
    if method == "none":
        return image
    if method != "zscore":
        raise ValueError(f"Unsupported intensity normalization method: {method}")

    arr = sitk.GetArrayFromImage(image).astype(np.float32)
    if arr.size == 0:
        return image

    p1 = np.percentile(arr, 1)
    p99 = np.percentile(arr, 99)
    if p99 > p1:
        arr = np.clip(arr, p1, p99)

    mask = arr != 0
    region = arr[mask] if np.any(mask) else arr
    mean = float(region.mean())
    std = float(region.std())
    if std > 1e-6:
        arr = (arr - mean) / std
    else:
        arr = arr - mean

    out = sitk.GetImageFromArray(arr)
    out.CopyInformation(image)
    return out


def prepare_for_inference(input_path: Path, work_dir: Path, intensity_norm: str = "none") -> Path:
    image = load_image(input_path)
    image = orient_to_lps(image)
    image = normalize_intensity(image, method=intensity_norm)
    nifti_path = work_dir / (input_path.stem.replace(".nii", "") + ".nii.gz")
    return save_as_nifti(image, nifti_path)
