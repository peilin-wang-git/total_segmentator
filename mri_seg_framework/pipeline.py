from __future__ import annotations

import json
import shutil
import traceback
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from .config import SegmentationConfig
from .inference import TotalSegmentatorRunner
from .io_utils import case_id_from_path, scan_mri_files
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
        )
        self.tmp_root = self.cfg.output_dir / "tmp"

    def run(self) -> Dict[str, Any]:
        self.logger.info("Starting MRI segmentation with config: %s", asdict(self.cfg))
        files = scan_mri_files(self.cfg.input_dir, self.cfg.supported_extensions)
        self.logger.info("Discovered %d candidate MRI files.", len(files))

        summary: List[Dict[str, Any]] = []
        for input_file in files:
            case_id = case_id_from_path(input_file, self.cfg.input_dir)
            case_output = self.cfg.output_dir / case_id
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
                normalized_input = prepare_for_inference(input_file, temp_dir)

                seg_path = case_output / "segmentation.nii.gz"
                label_map = self.runner.run(normalized_input, seg_path)

                clean_small_components(seg_path)
                save_label_map(label_map, case_output / "labels.json")

                if self.cfg.preview:
                    preview_path = case_output / "preview_overlay.png"
                    save_overlay_preview(normalized_input, seg_path, preview_path)
                    entry["preview_path"] = str(preview_path)

                entry["segmentation_path"] = str(seg_path)
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
