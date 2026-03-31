from __future__ import annotations

from pathlib import Path
from typing import Iterable, List


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
