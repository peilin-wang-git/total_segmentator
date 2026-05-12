from __future__ import annotations

import json
import shutil
import traceback
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import SimpleITK as sitk

from .config import SegmentationConfig
from .inference import TotalSegmentatorRunner
from .io_utils import case_id_from_path, load_mri_files_from_csv, scan_mri_files
from .logging_utils import setup_logger
from .postprocessing import clean_small_components, save_label_map
from .preprocessing import prepare_for_inference
from .visualization import save_overlay_preview


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
            files = load_mri_files_from_csv(self.cfg.input_csv, self.cfg.supported_extensions)
            case_root = Path("/")
            self.logger.info("Using CSV input list: %s", self.cfg.input_csv)
        else:
            files = scan_mri_files(self.cfg.input_dir, self.cfg.supported_extensions)
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
            }

            try:
                if self.cfg.dry_run:
                    entry["status"] = "dry_run_skipped"
                    entry["message"] = "Dry-run mode: inference skipped."
                    summary.append(entry)
                    continue

                temp_dir = self.tmp_root / case_id
                temp_dir.mkdir(parents=True, exist_ok=True)
                seg_path = case_output / "segmentation.nii.gz"
                label_map, preview_input = self._run_single_or_4d(input_file, temp_dir, seg_path)

                clean_small_components(seg_path)
                save_label_map(label_map, case_output / "labels.json")

                if self.cfg.preview and preview_input is not None:
                    preview_path = case_output / "preview_overlay.png"
                    save_overlay_preview(preview_input, seg_path, preview_path)
                    entry["preview_path"] = str(preview_path)

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

    def _run_single_or_4d(self, input_file: Path, temp_dir: Path, seg_path: Path) -> tuple[Dict[int, str], Optional[Path]]:
        image = sitk.ReadImage(str(input_file))
        if image.GetDimension() < 4:
            normalized_input = prepare_for_inference(input_file, temp_dir)
            return self.runner.run(normalized_input, seg_path), normalized_input

        frame_count = image.GetSize()[3]
        frame_outputs: List[sitk.Image] = []
        label_map: Dict[int, str] = {}

        for i in range(frame_count):
            frame = image[:, :, :, i]
            frame_path = temp_dir / f"frame_{i:04d}.nii.gz"
            sitk.WriteImage(frame, str(frame_path))

            frame_norm = prepare_for_inference(frame_path, temp_dir)
            frame_seg = temp_dir / f"frame_{i:04d}_seg.nii.gz"
            label_map = self.runner.run(frame_norm, frame_seg)
            frame_outputs.append(sitk.ReadImage(str(frame_seg)))

        seg_4d = sitk.JoinSeries(frame_outputs)
        seg_4d.CopyInformation(image)
        sitk.WriteImage(seg_4d, str(seg_path))
        return label_map, None
