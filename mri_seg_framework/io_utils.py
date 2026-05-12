from __future__ import annotations

from pathlib import Path
import ast
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
    table = pd.read_csv(csv_path)
    if table.empty:
        return []

    image_col = "image path" if "image path" in table.columns else table.columns[0]
    transpose_col = "transpose"
    flip_col = "flip"

    cases: List[Dict[str, object]] = []
    for _, row in table.iterrows():
        value = row.get(image_col)
        if pd.isna(value):
            continue
        p = Path(str(value))
        if not p.is_absolute():
            raise ValueError(f"CSV path must be absolute: {p}")
        if not (p.is_file() and has_medical_suffix(p, extensions)):
            continue

        transpose = _parse_triplet(row.get(transpose_col, "[0,1,2]"), default=[0, 1, 2])
        flip = _parse_triplet(row.get(flip_col, "[0,0,0]"), default=[0, 0, 0])
        cases.append({"path": p, "transpose": transpose, "flip": flip})
    return cases


def _parse_triplet(raw: object, default: List[int]) -> List[int]:
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return default
    if isinstance(raw, list):
        vals = raw
    else:
        vals = ast.literal_eval(str(raw))
    if not (isinstance(vals, (list, tuple)) and len(vals) == 3):
        raise ValueError(f"Expected list of 3 elements, got: {raw}")
    return [int(x) for x in vals]
