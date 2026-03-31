from __future__ import annotations

import argparse
from pathlib import Path

from .config import SegmentationConfig
from .pipeline import SegmentationPipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="MRI multi-organ segmentation framework")
    parser.add_argument("--input-dir", type=Path, help="Directory containing MRI images (recursive scan).")
    parser.add_argument("--output-dir", type=Path, help="Directory to save segmentation outputs.")
    parser.add_argument("--config", type=Path, default=None, help="Optional YAML config file.")
    parser.add_argument("--task", type=str, default="total_mr", help="TotalSegmentator task, default: total_mr")
    parser.add_argument("--fast", action="store_true", help="Enable fast mode if supported by model.")
    parser.add_argument("--no-preview", action="store_true", help="Disable preview PNG generation.")
    parser.add_argument("--keep-temp", action="store_true", help="Keep temporary normalized files.")
    parser.add_argument("--dry-run", action="store_true", help="Only scan and validate files, skip inference.")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.config:
        cfg = SegmentationConfig.from_yaml(args.config, input_dir=args.input_dir, output_dir=args.output_dir)
    else:
        if args.input_dir is None or args.output_dir is None:
            parser.error("--input-dir and --output-dir are required when --config is not provided.")
        cfg = SegmentationConfig(input_dir=args.input_dir, output_dir=args.output_dir)

    cfg.task = args.task
    cfg.fast = args.fast
    cfg.preview = not args.no_preview
    cfg.keep_temp = args.keep_temp
    cfg.dry_run = args.dry_run

    pipeline = SegmentationPipeline(cfg)
    pipeline.run()


if __name__ == "__main__":
    main()
