# -*- coding: utf-8 -*-
"""
Image ECG digitization adapters.

The service imports this module even when no image digitizer is configured.
Actual third-party dependencies are checked only when a backend is selected and
called. Every backend must return a float32 numpy array with shape (12, L).
"""
from __future__ import annotations

import glob
import os
import shutil
import subprocess
import tempfile
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image

from .config import (
    AHUS_MODEL_DIR,
    AHUS_LAYOUT_CONFIG,
    AHUS_PY,
    AHUS_ROOT,
    AHUS_WEIGHTS_DIR,
    DIGITIZER_BACKEND,
    FELIX_MODEL_DIR,
    FELIX_PY,
    FELIX_ROOT,
)


LEADS = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
SUPPORTED_BACKENDS = ("none", "felix", "ahus")


class DigitiseError(RuntimeError):
    pass


def _exists(path: str) -> bool:
    return bool(path) and os.path.exists(path)


def backend_status() -> dict[str, Any]:
    """Return configuration status without importing or running third-party code."""
    backend = DIGITIZER_BACKEND if DIGITIZER_BACKEND in SUPPORTED_BACKENDS else "invalid"
    return {
        "configured_backend": DIGITIZER_BACKEND,
        "backend": backend,
        "supported_backends": list(SUPPORTED_BACKENDS),
        "felix": {
            "root": FELIX_ROOT,
            "python": FELIX_PY,
            "model_dir": FELIX_MODEL_DIR,
            "root_exists": _exists(FELIX_ROOT),
            "python_exists": _exists(FELIX_PY),
            "model_dir_exists": _exists(FELIX_MODEL_DIR),
            "ready": _exists(FELIX_ROOT) and _exists(FELIX_PY) and _exists(FELIX_MODEL_DIR),
        },
        "ahus": {
            "root": AHUS_ROOT,
            "python": AHUS_PY,
            "model_dir": AHUS_MODEL_DIR,
            "weights_dir": AHUS_WEIGHTS_DIR,
            "layout_config": AHUS_LAYOUT_CONFIG,
            "root_exists": _exists(AHUS_ROOT),
            "python_exists": _exists(AHUS_PY),
            "model_dir_exists": _exists(AHUS_MODEL_DIR),
            "weights_dir_exists": _exists(AHUS_WEIGHTS_DIR),
            "ready": _exists(AHUS_ROOT) and _exists(AHUS_PY) and _exists(AHUS_WEIGHTS_DIR),
        },
    }


def _coerce_wfdb_record(fp: str) -> np.ndarray:
    """Parse a WFDB record into a 12-lead matrix, preserving missing leads as NaN."""
    try:
        import wfdb
    except ImportError as exc:
        raise DigitiseError("WFDB output found, but Python package 'wfdb' is not installed") from exc

    record_path = os.path.splitext(fp)[0] if fp.lower().endswith((".hea", ".dat")) else fp
    signal, fields = wfdb.rdsamp(record_path)
    sig_names = list(fields.get("sig_name") or [])
    if signal.ndim != 2 or not sig_names:
        raise DigitiseError(f"WFDB output is not a valid multi-lead signal: {fp}")

    x = np.full((12, signal.shape[0]), np.nan, dtype="float32")
    normalized = {str(name).strip(): idx for idx, name in enumerate(sig_names)}
    for row, lead in enumerate(LEADS):
        idx = normalized.get(lead)
        if idx is not None:
            x[row, :] = signal[:, idx].astype("float32")
    if not np.isfinite(x).any():
        raise DigitiseError(f"WFDB output contains no recognized standard ECG leads: {fp}")
    return x


def _coerce_12lead_matrix(fp: str) -> np.ndarray:
    """Parse digitizer output into (12, L)."""
    lower = fp.lower()
    if lower.endswith(".csv"):
        df = pd.read_csv(fp)
        cols = [c for c in LEADS if c in df.columns]
        if len(cols) == 12:
            x = df[cols].to_numpy("float32").T
        else:
            numeric = df.select_dtypes(include=["number"])
            if numeric.shape[1] < 12:
                raise DigitiseError(f"CSV numeric columns < 12: {fp}")
            x = numeric.iloc[:, :12].to_numpy("float32").T
    elif lower.endswith(".npy"):
        x = np.load(fp, allow_pickle=True).astype("float32")
    elif lower.endswith(".npz"):
        data = np.load(fp)
        x = None
        for key in data.files:
            arr = data[key]
            if arr.ndim == 2 and 12 in arr.shape:
                x = arr.astype("float32")
                break
        if x is None:
            raise DigitiseError(f"NPZ contains no 12-lead matrix: {fp}")
    elif lower.endswith((".hea", ".dat")):
        return _coerce_wfdb_record(fp)
    else:
        raise DigitiseError(f"Unsupported digitizer output type: {fp}")

    if x.ndim != 2:
        raise DigitiseError(f"Digitizer output is not 2D: shape={tuple(x.shape)}")
    if x.shape[0] != 12 and x.shape[1] == 12:
        x = x.T
    if x.shape[0] != 12:
        raise DigitiseError(f"Digitizer output shape {tuple(x.shape)} is not (12, L)")
    return np.asarray(x, dtype="float32")


def _find_digitized_output(out_dir: str, stem: str) -> str:
    candidates: list[str] = []
    for ext in ("*.csv", "*.npy", "*.npz", "*.hea"):
        candidates.extend(glob.glob(os.path.join(out_dir, f"{stem}*{ext}")))
    if not candidates:
        candidates = [
            fp
            for ext in ("*.csv", "*.npy", "*.npz", "*.hea")
            for fp in glob.glob(os.path.join(out_dir, "**", ext), recursive=True)
        ]
    if not candidates:
        raise DigitiseError("Digitizer produced no CSV/NPY/NPZ/WFDB output")
    return candidates[0]


def _ensure_felix_ready() -> None:
    if not _exists(FELIX_ROOT):
        raise DigitiseError(f"Felix ECG-Digitiser root not found: {FELIX_ROOT}")
    if not _exists(FELIX_PY):
        raise DigitiseError(f"Felix Python environment not found: {FELIX_PY}")
    if not _exists(FELIX_MODEL_DIR):
        raise DigitiseError(f"Felix model directory not found: {FELIX_MODEL_DIR}")


def _digitize_felix(img_path: str) -> np.ndarray:
    _ensure_felix_ready()
    work_dir = tempfile.mkdtemp(prefix="ecgdig_felix_")
    in_dir = os.path.join(work_dir, "in")
    out_dir = os.path.join(work_dir, "out")
    os.makedirs(in_dir, exist_ok=True)
    os.makedirs(out_dir, exist_ok=True)
    isolated_img = os.path.join(in_dir, os.path.basename(img_path))
    shutil.copy2(img_path, isolated_img)

    cmd = [
        FELIX_PY,
        "-m",
        "src.run.digitize",
        "-d",
        in_dir,
        "-o",
        out_dir,
        "-m",
        FELIX_MODEL_DIR,
        "-v",
    ]
    env = os.environ.copy()
    felix_scripts = os.path.dirname(FELIX_PY)
    env["PATH"] = felix_scripts + os.pathsep + env.get("PATH", "")
    try:
        proc = subprocess.run(
            cmd,
            cwd=FELIX_ROOT,
            check=True,
            capture_output=True,
            text=True,
            timeout=300,
            env=env,
        )
    except subprocess.CalledProcessError as exc:
        detail = (exc.stderr or exc.stdout or str(exc)).strip()
        raise DigitiseError(f"Felix digitizer failed: {detail}") from exc
    except subprocess.TimeoutExpired as exc:
        raise DigitiseError("Felix digitizer timed out after 300 seconds") from exc

    stem = os.path.splitext(os.path.basename(isolated_img))[0]
    fp = _find_digitized_output(out_dir, stem)
    try:
        return _coerce_12lead_matrix(fp)
    except DigitiseError as exc:
        raise DigitiseError(f"Felix output could not be parsed: {exc}") from exc


def _digitize_ahus(img_path: str) -> np.ndarray:
    if not _exists(AHUS_ROOT):
        raise DigitiseError(
            "Ahus-AIM Open-ECG-Digitizer is not installed. "
            f"Expected isolated root: {AHUS_ROOT}"
        )
    if not _exists(AHUS_PY):
        raise DigitiseError(
            "Ahus backend root exists, but its Python environment is not configured. "
            f"Expected: {AHUS_PY}"
        )
    if not _exists(AHUS_WEIGHTS_DIR):
        raise DigitiseError(f"Ahus weights directory not found: {AHUS_WEIGHTS_DIR}")

    work_dir = tempfile.mkdtemp(prefix="ecgdig_ahus_")
    in_dir = os.path.join(work_dir, "in")
    out_dir = os.path.join(work_dir, "out")
    os.makedirs(in_dir, exist_ok=True)
    os.makedirs(out_dir, exist_ok=True)
    # Keep the Ahus path conservative: only normalize the file format to PNG.
    # Layout correction/reflow adapters were intentionally removed.
    isolated_img = os.path.join(in_dir, "input.png")
    with Image.open(img_path) as im:
        im.convert("RGB").save(isolated_img)

    config_path = os.path.join(work_dir, "ahus_inference.yml")
    cfg = f"""MODEL:
  class_path: 'src.model.inference_wrapper.InferenceWrapper'
  KWARGS:
    config:
      SIGNAL_EXTRACTOR:
        class_path: 'src.model.signal_extractor.SignalExtractor'
        KWARGS: {{}}
      PERSPECTIVE_DETECTOR:
        class_path: 'src.model.perspective_detector.PerspectiveDetector'
        KWARGS:
          num_thetas: 250
      DEWARPER:
        class_path: 'src.model.dewarper.Dewarper'
        KWARGS:
          abs_peak_threshold: 0.1
      SEGMENTATION_MODEL:
        class_path: 'src.model.unet.UNet'
        weight_path: './weights/unet_weights_07072025.pt'
        KWARGS:
          num_in_channels: 3
          num_out_channels: 4
          dims: [32, 64, 128, 256, 320, 320, 320, 320]
          depth: 2
      CROPPER:
        class_path: 'src.model.cropper.Cropper'
        KWARGS:
          granularity: 80
          percentiles: [0.02, 0.98]
          alpha: 0.85
      PIXEL_SIZE_FINDER:
        class_path: 'src.model.pixel_size_finder.PixelSizeFinder'
        KWARGS:
          min_number_of_grid_lines: 30
          max_number_of_grid_lines: 70
          lower_grid_line_factor: 0.3
      LAYOUT_IDENTIFIER:
        class_path: 'src.model.lead_identifier.LeadIdentifier'
        config_path: '{AHUS_LAYOUT_CONFIG}'
        unet_config_path: 'src/config/lead_name_unet.yml'
        unet_weight_path: './weights/lead_name_unet_weights_07072025.pt'
        KWARGS:
          debug: false
          device: 'cpu'
          possibly_flipped: false
    device: 'cpu'
    resample_size: 3000
    rotate_on_resample: true
    enable_timing: false
    apply_dewarping: false
DATA:
  images_path: '{in_dir.replace(os.sep, "/")}'
  image_extensions: ['.png', '.PNG']
  output_path: '{out_dir.replace(os.sep, "/")}'
  save_mode: 'timeseries_only'
  layout_should_include_substring: false
  clear_output_dir_if_exists: true
"""
    with open(config_path, "w", encoding="utf-8") as f:
        f.write(cfg)

    cmd = [AHUS_PY, "-m", "src.digitize", "--config", config_path]
    try:
        proc = subprocess.run(
            cmd,
            cwd=AHUS_ROOT,
            check=True,
            capture_output=True,
            text=True,
            timeout=300,
        )
    except subprocess.CalledProcessError as exc:
        detail = (exc.stderr or exc.stdout or str(exc)).strip()
        raise DigitiseError(f"Ahus digitizer failed: {detail}") from exc
    except subprocess.TimeoutExpired as exc:
        raise DigitiseError("Ahus digitizer timed out after 300 seconds") from exc

    stem = os.path.splitext(os.path.basename(isolated_img))[0]
    fp = _find_digitized_output(out_dir, stem)
    try:
        x = _coerce_12lead_matrix(fp)
    except DigitiseError as exc:
        raise DigitiseError(f"Ahus output could not be parsed: {exc}") from exc
    if not np.isfinite(x).any():
        raise DigitiseError("Ahus output contains no finite signal values")
    return x


def digitize_image(img_path: str, backend: str | None = None) -> np.ndarray:
    selected = (backend or DIGITIZER_BACKEND or "none").strip().lower()
    if selected == "none":
        raise DigitiseError(
            "Image digitization backend is disabled. Set ECG_DIGITIZER_BACKEND=felix or ahus."
        )
    if selected == "felix":
        return _digitize_felix(img_path)
    if selected == "ahus":
        return _digitize_ahus(img_path)
    raise DigitiseError(f"Unsupported DIGITIZER_BACKEND={selected!r}; use none, felix, or ahus")


def digitize_3x4(img_path: str) -> np.ndarray:
    """Backward-compatible Felix 3x4 entrypoint."""
    return digitize_image(img_path, backend="felix")
