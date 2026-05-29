from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import onnxruntime as ort

from .io import resolve_classes_thresholds, resolve_onnx_path


TARGET_LEN = 5000


@dataclass
class ModelRuntime:
    model_dir: str
    onnx_path: str
    classes: List[str]
    thresholds: Dict[str, float]
    classes_file: str
    thresholds_file: str
    session: ort.InferenceSession
    input_name: str

    @property
    def threshold_array(self) -> np.ndarray:
        return np.array([self.thresholds[c] for c in self.classes], dtype=np.float32)

    def predict_batch(self, x: np.ndarray) -> np.ndarray:
        xb = np.asarray(x, dtype=np.float32)
        logits = self.session.run(None, {self.input_name: xb})[0]
        if logits.ndim != 2 or logits.shape[1] != len(self.classes):
            raise ValueError(
                f"ONNX output dimension {logits.shape} does not match classes length {len(self.classes)}"
            )
        probs = 1.0 / (1.0 + np.exp(-logits))
        return probs.astype(np.float32, copy=False)

    def predict_one(self, x: np.ndarray) -> np.ndarray:
        return self.predict_batch(np.asarray(x, dtype=np.float32)[None, ...])[0]


def project_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def default_model_dir(root: Optional[str] = None) -> str:
    root = root or project_root()
    for preferred in [
        os.path.join(root, "models_fine_chapman_ft"),
    ]:
        if os.path.isdir(preferred):
            return preferred
    return os.path.join(root, "models_fine")


def load_runtime(model_dir: Optional[str] = None, label_mode: int = 10) -> ModelRuntime:
    model_dir = os.path.abspath(model_dir or default_model_dir())
    classes, thresholds, classes_file, thresholds_file = resolve_classes_thresholds(model_dir, label_mode=label_mode)
    onnx_path = resolve_onnx_path(model_dir)
    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    return ModelRuntime(
        model_dir=model_dir,
        onnx_path=onnx_path,
        classes=classes,
        thresholds=thresholds,
        classes_file=classes_file,
        thresholds_file=thresholds_file,
        session=session,
        input_name=input_name,
    )


def normalize_and_resample(x: np.ndarray, target_len: int = TARGET_LEN) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if x.ndim != 2:
        raise ValueError(f"ECG tensor must be 2D [12, T], got {x.shape}")
    if x.shape[0] != 12:
        raise ValueError(f"ECG tensor must have 12 leads, got {x.shape[0]}")
    if not np.isfinite(x).any():
        raise ValueError("ECG tensor contains no finite values")
    if not np.isfinite(x).all():
        cleaned = x.copy()
        for lead_idx in range(cleaned.shape[0]):
            lead = cleaned[lead_idx]
            finite = np.isfinite(lead)
            if finite.any():
                lead[~finite] = float(np.nanmedian(lead[finite]))
            else:
                lead[:] = 0.0
        x = cleaned

    if x.shape[1] != target_len:
        base = np.arange(x.shape[1], dtype=np.float32)
        idx = np.linspace(0, x.shape[1] - 1, target_len, dtype=np.float32)
        out = np.zeros((x.shape[0], target_len), dtype=np.float32)
        for lead_idx in range(x.shape[0]):
            out[lead_idx] = np.interp(idx, base, x[lead_idx]).astype(np.float32)
        x = out

    mean = x.mean(axis=1, keepdims=True)
    std = x.std(axis=1, keepdims=True) + 1e-6
    return ((x - mean) / std).astype(np.float32, copy=False)


def _record_base_exists(base: Path) -> bool:
    return base.exists() or base.with_suffix(".hea").exists() or base.with_suffix(".dat").exists()


def _strip_record_suffix(path: Path) -> Path:
    if path.suffix.lower() in {".hea", ".dat"}:
        return path.with_suffix("")
    return path


def resolve_record_base(record: str, ptbxl_root: Optional[str] = None) -> str:
    raw = Path(record)
    candidates = []
    if raw.is_absolute():
        candidates.append(raw)
    else:
        cwd = Path.cwd()
        candidates.append(cwd / raw)
        if ptbxl_root:
            ptb = Path(ptbxl_root)
            candidates.append(ptb / raw)
            candidates.append(ptb / "wfdb" / raw)
            parts = raw.parts
            lower_parts = [p.lower() for p in parts]
            if "ptbxl" in lower_parts:
                idx = lower_parts.index("ptbxl")
                suffix = Path(*parts[idx + 1 :]) if idx + 1 < len(parts) else Path()
                if str(suffix):
                    candidates.append(ptb / suffix)
                    candidates.append(ptb / "wfdb" / suffix)

    for candidate in candidates:
        base = _strip_record_suffix(candidate)
        if _record_base_exists(base):
            return str(base)
    raise FileNotFoundError(
        f"Cannot find WFDB record {record}. Tried direct path and ptbxl_root/wfdb candidates."
    )


def load_wfdb_tensor(record: str, ptbxl_root: Optional[str] = None, target_len: int = TARGET_LEN) -> Tuple[np.ndarray, str]:
    import wfdb

    base = resolve_record_base(record, ptbxl_root=ptbxl_root)
    signal, _info = wfdb.rdsamp(base)
    x = signal.astype(np.float32).T
    return normalize_and_resample(x, target_len=target_len), base


def load_npy_tensor(path: str, target_len: int = TARGET_LEN) -> Tuple[np.ndarray, str]:
    resolved = os.path.abspath(path)
    if not os.path.exists(resolved):
        raise FileNotFoundError(f"Cannot find ECG npy file: {path}")
    x = np.load(resolved).astype(np.float32, copy=False)
    if x.ndim != 2:
        raise ValueError(f"ECG npy file must be 2D, got shape {x.shape}: {path}")
    if x.shape[0] != 12 and x.shape[1] == 12:
        x = x.T
    return normalize_and_resample(x, target_len=target_len), resolved


def predicted_labels(classes: List[str], probabilities: np.ndarray, thresholds: Dict[str, float]) -> List[str]:
    labels = [
        cls
        for cls, prob in zip(classes, np.asarray(probabilities, dtype=np.float32))
        if float(prob) >= float(thresholds[cls])
    ]
    if "NORM" in labels and any(label != "NORM" for label in labels):
        labels = [label for label in labels if label != "NORM"]
    return labels
