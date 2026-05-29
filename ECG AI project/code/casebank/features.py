from __future__ import annotations

from typing import Any, Dict

import numpy as np


EPS = 1e-6


def _clean_signal(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if x.ndim != 2:
        raise ValueError(f"ECG tensor must be 2D [12, T], got shape {x.shape}")
    if x.shape[0] != 12:
        raise ValueError(f"ECG tensor must have 12 leads, got {x.shape[0]}")
    if not np.isfinite(x).any():
        raise ValueError("ECG tensor contains no finite values")
    if np.isfinite(x).all():
        return x

    cleaned = x.copy()
    for lead_idx in range(cleaned.shape[0]):
        lead = cleaned[lead_idx]
        finite = np.isfinite(lead)
        if finite.any():
            fill = float(np.nanmedian(lead[finite]))
            lead[~finite] = fill
        else:
            lead[:] = 0.0
    return cleaned


def extract_basic_wave_features(x: np.ndarray) -> np.ndarray:
    """Extract 65 lightweight features from a normalized ECG tensor [12, T]."""
    x = _clean_signal(x)
    per_lead = [
        x.mean(axis=1),
        x.std(axis=1),
        x.min(axis=1),
        x.max(axis=1),
        np.mean(x * x, axis=1),
    ]
    global_features = np.array(
        [
            x.mean(),
            x.std(),
            x.min(),
            x.max(),
            np.mean(x * x),
        ],
        dtype=np.float32,
    )
    out = np.concatenate([np.asarray(v, dtype=np.float32) for v in per_lead] + [global_features])
    if out.shape != (65,):
        raise ValueError(f"Expected 65 wave features, got {out.shape}")
    return out.astype(np.float32, copy=False)


def _fit_mean_std(x: np.ndarray) -> Dict[str, Any]:
    arr = np.asarray(x, dtype=np.float32)
    mean = arr.mean(axis=0)
    std = arr.std(axis=0)
    std = np.where(std < EPS, 1.0, std)
    return {"mean": mean.astype(float).tolist(), "std": std.astype(float).tolist()}


def fit_vector_stats(
    probabilities: np.ndarray,
    margins: np.ndarray,
    wave_features: np.ndarray,
) -> Dict[str, Any]:
    return {
        "probabilities": _fit_mean_std(probabilities),
        "margins": _fit_mean_std(margins),
        "wave_features": _fit_mean_std(wave_features),
    }


def zscore(values: np.ndarray, stats: Dict[str, Any]) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    mean = np.asarray(stats["mean"], dtype=np.float32)
    std = np.asarray(stats["std"], dtype=np.float32)
    std = np.where(std < EPS, 1.0, std)
    return ((values - mean) / std).astype(np.float32, copy=False)


def build_retrieval_vectors(
    probabilities: np.ndarray,
    margins: np.ndarray,
    wave_features: np.ndarray,
    vector_stats: Dict[str, Any],
) -> np.ndarray:
    parts = [
        zscore(probabilities, vector_stats["probabilities"]),
        zscore(margins, vector_stats["margins"]),
        zscore(wave_features, vector_stats["wave_features"]),
    ]
    return np.concatenate(parts, axis=-1).astype(np.float32, copy=False)


def build_single_retrieval_vector(
    probabilities: np.ndarray,
    margins: np.ndarray,
    wave_features: np.ndarray,
    vector_stats: Dict[str, Any],
) -> np.ndarray:
    return build_retrieval_vectors(
        np.asarray(probabilities, dtype=np.float32)[None, :],
        np.asarray(margins, dtype=np.float32)[None, :],
        np.asarray(wave_features, dtype=np.float32)[None, :],
        vector_stats,
    )[0]
