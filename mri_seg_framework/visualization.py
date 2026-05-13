from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import SimpleITK as sitk


def save_overlay_preview(image_path: Path, seg_path: Path, output_png: Path) -> None:
    img = sitk.GetArrayFromImage(sitk.ReadImage(str(image_path))).astype(np.float32)
    seg = sitk.GetArrayFromImage(sitk.ReadImage(str(seg_path))).astype(np.int16)

    z = img.shape[0] // 2
    img_slice = img[z]
    seg_slice = seg[z]

    # Robust normalization
    p1, p99 = np.percentile(img_slice, [1, 99])
    img_slice = np.clip((img_slice - p1) / max(p99 - p1, 1e-6), 0, 1)

    output_png.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 8))
    plt.imshow(img_slice, cmap="gray")
    if np.any(seg_slice > 0):
        plt.imshow(np.ma.masked_where(seg_slice == 0, seg_slice), cmap="tab20", alpha=0.45)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(output_png, dpi=150)
    plt.close()


def save_overlay_slices_jpg(image_path: Path, seg_path: Path, output_dir: Path, slice_step: int = 10) -> None:
    img = sitk.GetArrayFromImage(sitk.ReadImage(str(image_path))).astype(np.float32)
    seg = sitk.GetArrayFromImage(sitk.ReadImage(str(seg_path))).astype(np.int16)

    output_dir.mkdir(parents=True, exist_ok=True)

    if img.ndim == 3 and seg.ndim == 3:
        _save_3d_overlay_series(img, seg, output_dir, slice_step=slice_step)
        return

    if img.ndim == 4 and seg.ndim == 4:
        frames = min(img.shape[0], seg.shape[0])
        for t in range(frames):
            frame_dir = output_dir / f"frame_{t:04d}"
            _save_3d_overlay_series(img[t], seg[t], frame_dir, slice_step=slice_step)
        return

    # Fallback for shape mismatch or uncommon dims: try first available 3D block.
    if img.ndim >= 3 and seg.ndim >= 3:
        _save_3d_overlay_series(
            np.asarray(img)[0] if img.ndim > 3 else img,
            np.asarray(seg)[0] if seg.ndim > 3 else seg,
            output_dir,
            slice_step=slice_step,
        )


def _save_3d_overlay_series(img_3d: np.ndarray, seg_3d: np.ndarray, output_dir: Path, slice_step: int = 10) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    num_slices = min(img_3d.shape[0], seg_3d.shape[0])
    unique_labels = np.unique(seg_3d)
    max_label = int(unique_labels.max()) if unique_labels.size > 0 else 0
    for z in range(0, num_slices, max(1, slice_step)):
        img_slice = img_3d[z]
        seg_slice = seg_3d[z]

        p1, p99 = np.percentile(img_slice, [1, 99])
        img_slice = np.clip((img_slice - p1) / max(p99 - p1, 1e-6), 0, 1)

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.imshow(img_slice, cmap="gray")
        if np.any(seg_slice > 0):
            ax.imshow(
                np.ma.masked_where(seg_slice == 0, seg_slice),
                cmap="tab20",
                vmin=1,
                vmax=max(20, max_label),
                alpha=0.45,
            )
        ax.axis("off")
        fig.tight_layout()
        fig.savefig(output_dir / f"slice_{z:04d}.jpg", dpi=120)
        plt.close(fig)
