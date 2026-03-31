from __future__ import annotations

from pathlib import Path

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


def prepare_for_inference(input_path: Path, work_dir: Path) -> Path:
    image = load_image(input_path)
    image = orient_to_lps(image)
    nifti_path = work_dir / (input_path.stem.replace(".nii", "") + ".nii.gz")
    return save_as_nifti(image, nifti_path)
