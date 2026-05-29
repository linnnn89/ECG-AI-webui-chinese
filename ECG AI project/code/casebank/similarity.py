from __future__ import annotations

from typing import Iterable, Set

import numpy as np


EPS = 1e-8


def cosine_similarity_01(a: np.ndarray, b: np.ndarray) -> float:
    """Return cosine similarity mapped to [0, 1]."""
    va = np.asarray(a, dtype=np.float32).reshape(-1)
    vb = np.asarray(b, dtype=np.float32).reshape(-1)
    denom = float(np.linalg.norm(va) * np.linalg.norm(vb))
    if denom <= EPS:
        return 0.5
    cos = float(np.dot(va, vb) / denom)
    sim = (cos + 1.0) / 2.0
    return float(np.clip(sim, 0.0, 1.0))


def batch_cosine_similarity_01(query: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    q = np.asarray(query, dtype=np.float32).reshape(1, -1)
    m = np.asarray(matrix, dtype=np.float32)
    q_norm = np.linalg.norm(q, axis=1, keepdims=True).reshape(-1)[0]
    m_norm = np.linalg.norm(m, axis=1)
    denom = m_norm * q_norm
    cos = np.zeros(m.shape[0], dtype=np.float32)
    valid = denom > EPS
    cos[valid] = (m[valid] @ q.reshape(-1)) / denom[valid]
    return np.clip((cos + 1.0) / 2.0, 0.0, 1.0).astype(np.float32)


def jaccard_labels(a: Iterable[str] | Set[str], b: Iterable[str] | Set[str]) -> float:
    sa = set(a)
    sb = set(b)
    if not sa and not sb:
        return 0.0
    return float(len(sa & sb) / len(sa | sb))


def wave_similarity_01(a: np.ndarray, b: np.ndarray) -> float:
    va = np.asarray(a, dtype=np.float32).reshape(-1)
    vb = np.asarray(b, dtype=np.float32).reshape(-1)
    dist = float(np.linalg.norm(va - vb))
    return float(1.0 / (1.0 + dist))


def score_level(score: float) -> str:
    if score >= 0.80:
        return "high"
    if score >= 0.65:
        return "medium"
    if score >= 0.55:
        return "weak"
    return "hidden"
