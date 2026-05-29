from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict

import numpy as np

from .inference import normalize_and_resample, project_root


def resolve_project_path(path: str | os.PathLike[str] | None) -> str:
    text = str(path or "").strip()
    if not text:
        raise FileNotFoundError("empty ECG path")
    raw = Path(text)
    if raw.is_absolute():
        return str(raw)
    return str(Path(project_root()) / raw)


def load_cpsc_mat_tensor(path: str | os.PathLike[str], mat_key: str = "val") -> np.ndarray:
    from scipy.io import loadmat

    resolved = resolve_project_path(path)
    payload = loadmat(resolved)
    key = mat_key if mat_key and mat_key in payload else "val"
    if key not in payload:
        candidates = [name for name, value in payload.items() if not name.startswith("__") and hasattr(value, "shape")]
        if not candidates:
            raise ValueError(f"No signal matrix found in CPSC MAT file: {path}")
        key = candidates[0]
    arr = np.asarray(payload[key], dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"CPSC MAT signal must be 2D, got {arr.shape}: {path}")
    if arr.shape[0] != 12 and arr.shape[1] == 12:
        arr = arr.T
    return normalize_and_resample(arr)


def load_npy_case_tensor(path: str | os.PathLike[str]) -> np.ndarray:
    resolved = resolve_project_path(path)
    x = np.load(resolved).astype(np.float32, copy=False)
    if x.ndim != 2:
        raise ValueError(f"ECG npy file must be 2D, got {x.shape}: {path}")
    if x.shape[0] != 12 and x.shape[1] == 12:
        x = x.T
    return normalize_and_resample(x)


def load_wfdb_case_tensor(path: str | os.PathLike[str]) -> np.ndarray:
    import wfdb

    resolved = resolve_project_path(path)
    base = Path(resolved)
    if base.suffix.lower() in {".hea", ".dat", ".mat"}:
        base = base.with_suffix("")
    signal, _info = wfdb.rdsamp(str(base))
    return normalize_and_resample(np.asarray(signal, dtype=np.float32).T)


def load_case_tensor(record_path: str, signal_quality: Dict[str, Any] | None = None) -> np.ndarray:
    q = signal_quality or {}
    fmt = str(q.get("record_format", "")).strip().lower()
    if fmt == "cpsc_mat":
        return load_cpsc_mat_tensor(record_path, mat_key=str(q.get("mat_key", "val")))
    if fmt == "wfdb":
        return load_wfdb_case_tensor(record_path)
    if fmt == "npy":
        return load_npy_case_tensor(record_path)

    suffix = Path(str(record_path)).suffix.lower()
    if suffix == ".npy":
        return load_npy_case_tensor(record_path)
    if suffix == ".mat" and q.get("source") == "CPSC2018":
        return load_cpsc_mat_tensor(record_path, mat_key=str(q.get("mat_key", "val")))
    if suffix in {"", ".hea", ".dat", ".mat"}:
        return load_wfdb_case_tensor(record_path)
    raise ValueError(f"Unsupported ECG record format for CaseBank: {record_path}")
