from __future__ import annotations

from pathlib import Path
import ast
import csv
from typing import Dict, Iterable, List

import pandas as pd


SUPPORTED_EXTENSIONS = (".nii", ".nii.gz", ".mha", ".nrrd")


def has_medical_suffix(path: Path, extensions: Iterable[str] = SUPPORTED_EXTENSIONS) -> bool:
    name = path.name.lower()
    return any(name.endswith(ext.lower()) for ext in extensions)


def scan_mri_files(input_dir: Path, extensions: Iterable[str] = SUPPORTED_EXTENSIONS) -> List[Path]:
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")
    if not input_dir.is_dir():
        raise NotADirectoryError(f"Input path is not a directory: {input_dir}")

    return sorted([p for p in input_dir.rglob("*") if p.is_file() and has_medical_suffix(p, extensions)])


def case_id_from_path(path: Path, root: Path) -> str:
    relative = path.relative_to(root)
    safe = "__".join(relative.parts)
    for suffix in (".nii.gz", ".nii", ".mha", ".nrrd"):
        if safe.endswith(suffix):
            return safe[: -len(suffix)]
    return path.stem


def load_mri_files_from_csv(csv_path: Path, extensions: Iterable[str] = SUPPORTED_EXTENSIONS) -> List[Path]:
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file does not exist: {csv_path}")
    table = pd.read_csv(csv_path)
    if table.empty:
        return []

    first_col = table.columns[0]
    files: List[Path] = []
    for value in table[first_col].dropna().tolist():
        p = Path(str(value))
        if not p.is_absolute():
            raise ValueError(f"CSV path must be absolute: {p}")
        if p.is_file() and has_medical_suffix(p, extensions):
            files.append(p)
    return sorted(files)


def load_cases_from_csv(csv_path: Path, extensions: Iterable[str] = SUPPORTED_EXTENSIONS) -> List[Dict[str, object]]:
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file does not exist: {csv_path}")

    cases: List[Dict[str, object]] = []
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, skipinitialspace=True)
        if reader.fieldnames is None:
            return []

        fields = [str(x).strip().lower() for x in reader.fieldnames]
        image_col = "image path" if "image path" in fields else fields[0]

        for raw_row in reader:
            row = {str(k).strip().lower(): v for k, v in raw_row.items() if k is not None}
            value = row.get(image_col)
            if value is None or str(value).strip() == "":
                continue

            p = Path(str(value).strip().strip('"').strip("'"))
            if not p.is_absolute():
                raise ValueError(f"CSV path must be absolute: {p}")
            if not (p.is_file() and has_medical_suffix(p, extensions)):
                continue

            transpose = _parse_triplet(row.get("transpose", "[0,1,2]"), default=[0, 1, 2])
            flip = _parse_triplet(row.get("flip", "[0,0,0]"), default=[0, 0, 0])
            cases.append({"path": p, "transpose": transpose, "flip": flip})
    print(f"[CSV] Loaded {len(cases)} cases from: {csv_path}")
    for i, case in enumerate(cases, start=1):
        print(
            f"[CSV][{i:03d}] image path={case['path']} | transpose={case['transpose']} | flip={case['flip']}"
        )
    return cases


def _parse_triplet(raw: object, default: List[int]) -> List[int]:
    if raw is None:
        return default
    s0 = str(raw).strip()
    if s0 == "":
        return default
    if isinstance(raw, float) and pd.isna(raw):
        return default
    if isinstance(raw, list):
        vals = raw
    else:
        s = s0.strip('"').strip("'")
        if not s:
            return default
        if s.startswith("[") or s.startswith("("):
            vals = ast.literal_eval(s)
        else:
            vals = [x.strip() for x in s.split(",")]
    if not (isinstance(vals, (list, tuple)) and len(vals) == 3):
        raise ValueError(f"Expected list of 3 elements, got: {raw}")
    return [int(x) for x in vals]
