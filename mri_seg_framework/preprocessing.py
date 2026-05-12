from __future__ import annotations

from pathlib import Path

import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt


def load_image(path: Path) -> sitk.Image:
    return sitk.ReadImage(str(path))


def orient_to_lps(image: sitk.Image) -> sitk.Image:
    orienter = sitk.DICOMOrientImageFilter()
    orienter.SetDesiredCoordinateOrientation("LPS")
    return orienter.Execute(image)


def get_orientation_code(image: sitk.Image) -> str:
    return sitk.DICOMOrientImageFilter.GetOrientationFromDirectionCosines(image.GetDirection())


def orient_to_code(image: sitk.Image, code: str) -> sitk.Image:
    orienter = sitk.DICOMOrientImageFilter()
    orienter.SetDesiredCoordinateOrientation(code)
    return orienter.Execute(image)


def apply_transpose_flip(image: sitk.Image, transpose: list[int], flip: list[int]) -> sitk.Image:
    arr = sitk.GetArrayFromImage(image)
    # SITK array order is [z, y, x], map requested [0,1,2] (x,y,z) to array axes.
    to_array_axis = {0: 2, 1: 1, 2: 0}
    transpose_arr = [to_array_axis[i] for i in transpose[::-1]]
    arr = np.transpose(arr, axes=transpose_arr)

    for axis_xyz, do_flip in enumerate(flip):
        if bool(do_flip):
            arr_axis = to_array_axis[axis_xyz]
            arr = np.flip(arr, axis=arr_axis)

    out = sitk.GetImageFromArray(arr)
    out.SetOrigin(image.GetOrigin())
    out.SetSpacing(image.GetSpacing())
    out.SetDirection(image.GetDirection())
    return out


def invert_transpose_flip(image: sitk.Image, transpose: list[int], flip: list[int]) -> sitk.Image:
    arr = sitk.GetArrayFromImage(image)
    to_array_axis = {0: 2, 1: 1, 2: 0}

    # Undo flip first
    for axis_xyz, do_flip in enumerate(flip):
        if bool(do_flip):
            arr_axis = to_array_axis[axis_xyz]
            arr = np.flip(arr, axis=arr_axis)

    inverse = [0, 0, 0]
    for i, idx in enumerate(transpose):
        inverse[idx] = i
    inverse_arr = [to_array_axis[i] for i in inverse[::-1]]
    arr = np.transpose(arr, axes=inverse_arr)

    out = sitk.GetImageFromArray(arr)
    out.SetOrigin(image.GetOrigin())
    out.SetSpacing(image.GetSpacing())
    out.SetDirection(image.GetDirection())
    return out


def save_as_nifti(image: sitk.Image, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(image, str(output_path))
    return output_path


def normalize_intensity(image: sitk.Image, method: str = "none") -> sitk.Image:
    method = (method or "none").lower()
    if method == "none":
        return image
    if method not in {"zscore", "percentile_minmax", "zscore_robust", "itksnap_window"}:
        raise ValueError(f"Unsupported intensity normalization method: {method}")

    arr = sitk.GetArrayFromImage(image).astype(np.float32)
    if arr.size == 0:
        return image

    p1 = np.percentile(arr, 1)
    p99 = np.percentile(arr, 99)
    if p99 > p1:
        arr = np.clip(arr, p1, p99)

    if method == "percentile_minmax":
        lo = float(arr.min())
        hi = float(arr.max())
        if hi > lo:
            arr = (arr - lo) / (hi - lo)
        else:
            arr = arr - lo
        out = sitk.GetImageFromArray(arr)
        out.CopyInformation(image)
        return out

    if method == "itksnap_window":
        # Case-by-case adaptive windowing using image contrast statistics.
        # 1) robust foreground estimation via Otsu on non-zero voxels
        # 2) derive center from foreground median
        # 3) adapt width from foreground std + robust global range
        nonzero = arr[arr != 0]
        if nonzero.size > 32:
            try:
                otsu = sitk.OtsuThresholdImageFilter()
                otsu.SetInsideValue(0)
                otsu.SetOutsideValue(1)
                mask_img = otsu.Execute(sitk.GetImageFromArray(arr.astype(np.float32)))
                mask = sitk.GetArrayFromImage(mask_img).astype(bool)
                region = arr[mask] if np.any(mask) else nonzero
            except Exception:
                region = nonzero
        else:
            region = arr.reshape(-1)

        center = float(np.median(region))
        fg_std = float(np.std(region))
        rg_low = float(np.percentile(arr, 0.5))
        rg_high = float(np.percentile(arr, 99.5))
        robust_range = max(rg_high - rg_low, 1e-6)

        contrast_ratio = fg_std / robust_range
        if contrast_ratio < 0.08:
            width = 1.4 * robust_range
        elif contrast_ratio < 0.16:
            width = 1.8 * robust_range
        else:
            width = 2.2 * robust_range
        width = max(width, 6.0 * fg_std, 1e-6)

        p_low = center - 0.5 * width
        p_high = center + 0.5 * width
        arr = np.clip(arr, p_low, p_high)
        arr = (arr - p_low) / max(p_high - p_low, 1e-6)
        out = sitk.GetImageFromArray(arr.astype(np.float32))
        out.CopyInformation(image)
        return out

    mask = arr != 0
    region = arr[mask] if np.any(mask) else arr
    if method == "zscore":
        mean = float(region.mean())
        std = float(region.std())
        if std > 1e-6:
            arr = (arr - mean) / std
        else:
            arr = arr - mean
    else:
        median = float(np.median(region))
        mad = float(np.median(np.abs(region - median)))
        robust_std = 1.4826 * mad
        if robust_std > 1e-6:
            arr = (arr - median) / robust_std
        else:
            arr = arr - median

    out = sitk.GetImageFromArray(arr)
    out.CopyInformation(image)
    return out


def prepare_for_inference(input_path: Path, work_dir: Path, intensity_norm: str = "none") -> Path:
    image = load_image(input_path)
    image = orient_to_lps(image)
    image = normalize_intensity(image, method=intensity_norm)
    nifti_path = work_dir / (input_path.stem.replace(".nii", "") + ".nii.gz")
    return save_as_nifti(image, nifti_path)


def save_intensity_colorbar_preview(image_path: Path, output_png: Path) -> Path:
    image = sitk.ReadImage(str(image_path))
    arr = sitk.GetArrayFromImage(image).astype(np.float32)
    if arr.ndim >= 3:
        slice_2d = arr[arr.shape[0] // 2]
    else:
        slice_2d = arr

    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(slice_2d, cmap="gray")
    ax.set_title("Normalized intensity preview")
    ax.axis("off")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Intensity")
    fig.tight_layout()
    fig.savefig(output_png, dpi=150)
    plt.close(fig)
    return output_png
