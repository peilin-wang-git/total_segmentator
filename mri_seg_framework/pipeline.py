from __future__ import annotations

import json
import shutil
import traceback
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import SimpleITK as sitk
import numpy as np

from .config import SegmentationConfig
from .inference import TotalSegmentatorRunner
from .io_utils import case_id_from_path, load_cases_from_csv, scan_mri_files
from .logging_utils import setup_logger
from .postprocessing import clean_small_components, save_label_map
from .preprocessing import (
    apply_transpose_flip,
    get_orientation_code,
    invert_transpose_flip,
    orient_to_code,
    prepare_for_inference,
    save_intensity_colorbar_preview,
)
from .visualization import save_overlay_preview, save_overlay_slices_jpg


class SegmentationPipeline:
    def __init__(self, cfg: SegmentationConfig):
        self.cfg = cfg
        self.cfg.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger = setup_logger(self.cfg.output_dir / "run.log")
        self.runner = TotalSegmentatorRunner(
            task=self.cfg.task,
            fast=self.cfg.fast,
            ml=self.cfg.ml,
            roi_subset=self.cfg.roi_subset,
            device=self.cfg.device,
            gpu_id=self.cfg.gpu_id,
        )
        self.tmp_root = self.cfg.output_dir / "tmp"

    def run(self) -> Dict[str, Any]:
        self.logger.info("Starting MRI segmentation with config: %s", asdict(self.cfg))
        if self.cfg.input_csv:
            csv_cases = load_cases_from_csv(self.cfg.input_csv, self.cfg.supported_extensions)
            files = [x["path"] for x in csv_cases]
            case_meta = {str(x["path"]): x for x in csv_cases}
            case_root = Path("/")
            self.logger.info("Using CSV input list: %s", self.cfg.input_csv)
            self.logger.info("[CSV] Loaded %d cases from: %s", len(csv_cases), self.cfg.input_csv)
            for i, case in enumerate(csv_cases, start=1):
                self.logger.info(
                    "[CSV][%03d] image path=%s | transpose=%s | flip=%s",
                    i,
                    case["path"],
                    case["transpose"],
                    case["flip"],
                )
        else:
            files = scan_mri_files(self.cfg.input_dir, self.cfg.supported_extensions)
            case_meta = {}
            case_root = self.cfg.input_dir
        self.logger.info("Discovered %d candidate MRI files.", len(files))

        summary: List[Dict[str, Any]] = []
        for input_file in files:
            case_id = case_id_from_path(input_file, case_root)
            case_output = self._resolve_case_output_dir(input_file, case_id)
            case_output.mkdir(parents=True, exist_ok=True)

            entry = {
                "case_id": case_id,
                "input_path": str(input_file),
                "status": "pending",
                "message": "",
                "segmentation_path": "",
                "preview_path": "",
                "input_nonzero_ratio": None,
                "input_p01": None,
                "input_p50": None,
                "input_p99": None,
                "seg_nonzero_before_clean": None,
                "seg_nonzero_after_clean": None,
                "normalized_image_path": "",
                "intensity_colorbar_path": "",
                "overlay_slices_dir": "",
                "csv_transpose": "",
                "csv_flip": "",
            }

            try:
                if self.cfg.dry_run:
                    entry["status"] = "dry_run_skipped"
                    entry["message"] = "Dry-run mode: inference skipped."
                    summary.append(entry)
                    continue

                temp_dir = self.tmp_root / case_id
                temp_dir.mkdir(parents=True, exist_ok=True)
                input_stats = self._image_intensity_stats(input_file)
                entry.update(input_stats)

                case_transform = case_meta.get(str(input_file), {})
                transpose = case_transform.get("transpose", [0, 1, 2])
                flip = case_transform.get("flip", [0, 0, 0])
                entry["csv_transpose"] = str(transpose)
                entry["csv_flip"] = str(flip)
                run_input_file = self._prepare_case_input_with_transform(input_file, temp_dir, transpose, flip)

                seg_path = case_output / "segmentation.nii.gz"
                label_map, preview_input, is_4d_case, normalized_qc_path = self._run_single_or_4d(run_input_file, temp_dir, seg_path)
                entry["seg_nonzero_before_clean"] = self._seg_nonzero_voxels(seg_path)
                if not is_4d_case:
                    clean_small_components(seg_path)
                entry["seg_nonzero_after_clean"] = self._seg_nonzero_voxels(seg_path)
                save_label_map(label_map, case_output / "labels.json")

                if normalized_qc_path is not None and normalized_qc_path.exists():
                    qc_nifti = case_output / "normalized_input.nii.gz"
                    sitk.WriteImage(sitk.ReadImage(str(normalized_qc_path)), str(qc_nifti))
                    entry["normalized_image_path"] = str(qc_nifti)
                    colorbar_path = case_output / "normalized_intensity_colorbar.png"
                    save_intensity_colorbar_preview(qc_nifti, colorbar_path)
                    entry["intensity_colorbar_path"] = str(colorbar_path)

                if self.cfg.preview and preview_input is not None:
                    preview_path = case_output / "preview_overlay.png"
                    save_overlay_preview(preview_input, seg_path, preview_path)
                    entry["preview_path"] = str(preview_path)

                overlay_dir = case_output / "overlay_slices_jpg"
                overlay_input = normalized_qc_path if normalized_qc_path is not None else input_file
                save_overlay_slices_jpg(overlay_input, seg_path, overlay_dir)
                entry["overlay_slices_dir"] = str(overlay_dir)

                # Restore segmentation to original transform space only at final output stage.
                self._restore_case_output_transform(seg_path, None, transpose, flip)

                final_seg_path = self._save_seg_to_input_path(input_file, seg_path)
                entry["segmentation_path"] = str(final_seg_path)
                entry["status"] = "success"
                entry["message"] = "Completed"

                if not self.cfg.keep_temp and temp_dir.exists():
                    shutil.rmtree(temp_dir)

            except Exception as exc:
                entry["status"] = "failed"
                entry["message"] = f"{exc}\n{traceback.format_exc(limit=2)}"
                self.logger.error("Case %s failed: %s", case_id, exc)

            summary.append(entry)

        summary_json = self.cfg.output_dir / "summary.json"
        with summary_json.open("w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

        pd.DataFrame(summary).to_csv(self.cfg.output_dir / "summary.csv", index=False)

        ok = sum(1 for x in summary if x["status"] == "success")
        failed = sum(1 for x in summary if x["status"] == "failed")
        self.logger.info("Pipeline finished. success=%d failed=%d total=%d", ok, failed, len(summary))
        return {"success": ok, "failed": failed, "total": len(summary), "summary": summary}

    def _save_seg_to_input_path(self, input_file: Path, generated_seg_path: Path) -> Path:
        suffixes = "".join(input_file.suffixes)
        output_suffix = self.cfg.output_suffix or "_seg"
        if suffixes:
            output_name = input_file.name[: -len(suffixes)] + f"{output_suffix}{suffixes}"
        else:
            output_name = input_file.name + output_suffix
        final_path = input_file.parent / output_name
        sitk.WriteImage(sitk.ReadImage(str(generated_seg_path)), str(final_path))
        return final_path

    def _resolve_case_output_dir(self, input_file: Path, case_id: str) -> Path:
        if not self.cfg.input_csv:
            return self.cfg.output_dir / case_id
        image_stem = input_file.name
        for suffix in (".nii.gz", ".nii", ".mha", ".nrrd"):
            if image_stem.endswith(suffix):
                image_stem = image_stem[: -len(suffix)]
                break
        return input_file.parent / f"{image_stem}_totalseg"

    def _run_single_or_4d(self, input_file: Path, temp_dir: Path, seg_path: Path) -> tuple[Dict[int, str], Optional[Path], bool, Optional[Path]]:
        image = sitk.ReadImage(str(input_file))
        original_orientation = get_orientation_code(image)
        if image.GetDimension() < 4:
            normalized_input = prepare_for_inference(input_file, temp_dir, intensity_norm=self.cfg.intensity_norm)
            label_map = self.runner.run(normalized_input, seg_path)
            seg_img = sitk.ReadImage(str(seg_path))
            seg_img = orient_to_code(seg_img, original_orientation)
            sitk.WriteImage(seg_img, str(seg_path))

            norm_img = sitk.ReadImage(str(normalized_input))
            norm_img = orient_to_code(norm_img, original_orientation)
            norm_out = temp_dir / "normalized_input_original_orientation.nii.gz"
            sitk.WriteImage(norm_img, str(norm_out))
            return label_map, norm_out, False, norm_out

        frame_count = image.GetSize()[3]
        frame_outputs: List[sitk.Image] = []
        normalized_frames: List[sitk.Image] = []
        label_map: Dict[int, str] = {}

        for i in range(frame_count):
            frame = image[:, :, :, i]
            frame_path = temp_dir / f"frame_{i:04d}.nii.gz"
            sitk.WriteImage(frame, str(frame_path))

            frame_norm = prepare_for_inference(frame_path, temp_dir, intensity_norm=self.cfg.intensity_norm)
            normalized_frames.append(sitk.ReadImage(str(frame_norm)))
            frame_seg = temp_dir / f"frame_{i:04d}_seg.nii.gz"
            label_map = self.runner.run(frame_norm, frame_seg)
            clean_small_components(frame_seg)
            frame_outputs.append(sitk.ReadImage(str(frame_seg)))

        seg_4d = sitk.JoinSeries(frame_outputs)
        seg_4d.CopyInformation(image)
        sitk.WriteImage(seg_4d, str(seg_path))
        norm_4d = sitk.JoinSeries(normalized_frames)
        norm_4d.CopyInformation(image)
        norm_qc_path = temp_dir / "normalized_4d_input.nii.gz"
        sitk.WriteImage(norm_4d, str(norm_qc_path))
        return label_map, None, True, norm_qc_path

    def _seg_nonzero_voxels(self, seg_path: Path) -> int:
        arr = sitk.GetArrayFromImage(sitk.ReadImage(str(seg_path)))
        return int(np.count_nonzero(arr))

    def _image_intensity_stats(self, image_path: Path) -> Dict[str, float]:
        arr = sitk.GetArrayFromImage(sitk.ReadImage(str(image_path))).astype(np.float32)
        flat = arr.reshape(-1)
        if flat.size == 0:
            return {
                "input_nonzero_ratio": 0.0,
                "input_p01": 0.0,
                "input_p50": 0.0,
                "input_p99": 0.0,
            }
        return {
            "input_nonzero_ratio": float(np.count_nonzero(flat) / flat.size),
            "input_p01": float(np.percentile(flat, 1)),
            "input_p50": float(np.percentile(flat, 50)),
            "input_p99": float(np.percentile(flat, 99)),
        }

    def _prepare_case_input_with_transform(self, input_file: Path, temp_dir: Path, transpose: list[int], flip: list[int]) -> Path:
        if transpose == [0, 1, 2] and flip == [0, 0, 0]:
            return input_file
        image = sitk.ReadImage(str(input_file))
        if image.GetDimension() == 3:
            transformed = apply_transpose_flip(image, transpose, flip)
        elif image.GetDimension() == 4:
            transformed_frames: List[sitk.Image] = []
            frame_count = image.GetSize()[3]
            for i in range(frame_count):
                frame = image[:, :, :, i]
                transformed_frames.append(apply_transpose_flip(frame, transpose, flip))
            transformed = sitk.JoinSeries(transformed_frames)
        else:
            return input_file
        transformed_path = temp_dir / f"{input_file.stem}_csv_transformed.nii.gz"
        sitk.WriteImage(transformed, str(transformed_path))
        return transformed_path

    def _restore_case_output_transform(self, seg_path: Path, normalized_qc_path: Optional[Path], transpose: list[int], flip: list[int]) -> None:
        if transpose == [0, 1, 2] and flip == [0, 0, 0]:
            return
        seg = sitk.ReadImage(str(seg_path))
        if seg.GetDimension() == 3:
            seg = invert_transpose_flip(seg, transpose, flip)
            sitk.WriteImage(seg, str(seg_path))
        elif seg.GetDimension() == 4:
            restored_frames: List[sitk.Image] = []
            frame_count = seg.GetSize()[3]
            for i in range(frame_count):
                frame = seg[:, :, :, i]
                restored_frames.append(invert_transpose_flip(frame, transpose, flip))
            seg_restored = sitk.JoinSeries(restored_frames)
            sitk.WriteImage(seg_restored, str(seg_path))
        if normalized_qc_path is not None and normalized_qc_path.exists():
            qc = sitk.ReadImage(str(normalized_qc_path))
            if qc.GetDimension() == 3:
                qc = invert_transpose_flip(qc, transpose, flip)
                sitk.WriteImage(qc, str(normalized_qc_path))
            elif qc.GetDimension() == 4:
                restored_qc_frames: List[sitk.Image] = []
                frame_count = qc.GetSize()[3]
                for i in range(frame_count):
                    frame = qc[:, :, :, i]
                    restored_qc_frames.append(invert_transpose_flip(frame, transpose, flip))
                qc_restored = sitk.JoinSeries(restored_qc_frames)
                sitk.WriteImage(qc_restored, str(normalized_qc_path))
