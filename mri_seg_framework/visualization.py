from __future__ import annotations

from pathlib import Path

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
