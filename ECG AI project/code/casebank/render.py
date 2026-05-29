from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable

import numpy as np

from .schema import CaseRecord
from .waveform_io import load_case_tensor, resolve_project_path


LEAD_NAMES = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]


def safe_source_dir(source: str) -> str:
    return "".join(ch.lower() if ch.isalnum() else "_" for ch in str(source)).strip("_") or "unknown"


def render_cache_path(cache_dir: str | os.PathLike[str], source: str, case_id: str) -> str:
    return str(Path(cache_dir) / "rendered_ecg_png" / safe_source_dir(source) / f"{case_id}.png")


def _diagnosis_line(labels: Iterable[str], diagnosis: str | None) -> str:
    label_text = "|".join([str(x) for x in labels if str(x).strip()])
    if diagnosis and diagnosis != label_text:
        return f"{label_text} ; {diagnosis}" if label_text else diagnosis
    return label_text or "unmapped"


def render_case_png(case: CaseRecord, cache_dir: str | os.PathLike[str], overwrite: bool = False) -> str:
    out_path = Path(render_cache_path(cache_dir, case.source, case.case_id))
    if out_path.exists() and not overwrite:
        return str(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not case.record_path:
        raise FileNotFoundError(f"Case has no record_path: {case.case_id}")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x = load_case_tensor(case.record_path, {**(case.signal_quality or {}), "source": case.source})
    x = np.asarray(x, dtype=np.float32)
    t = np.linspace(0.0, 10.0, x.shape[1], dtype=np.float32)
    step = max(1, int(np.ceil(x.shape[1] / 2500)))

    fig, axes = plt.subplots(12, 1, figsize=(12, 9), sharex=True)
    diagnosis = _diagnosis_line(case.labels, (case.signal_quality or {}).get("true_diagnosis"))
    fig.suptitle(f"{case.case_id} | {case.source} | {diagnosis}", fontsize=10)
    for idx, ax in enumerate(axes):
        y = x[idx]
        ax.plot(t[::step], y[::step], color="#111827", linewidth=0.75)
        ax.axhline(0, color="#d1d5db", linewidth=0.4)
        ax.set_ylabel(LEAD_NAMES[idx], rotation=0, labelpad=18, fontsize=8, va="center")
        ax.set_yticks([])
        ax.grid(True, axis="x", color="#eef2f7", linewidth=0.4)
        for spine in ax.spines.values():
            spine.set_visible(False)
    axes[-1].set_xlabel("Time (s)", fontsize=8)
    fig.tight_layout(rect=[0.02, 0.02, 1, 0.96])
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return str(out_path)


def resolve_rendered_png(path: str | os.PathLike[str]) -> str:
    resolved = resolve_project_path(path)
    if not os.path.exists(resolved):
        raise FileNotFoundError(f"Rendered ECG PNG does not exist: {path}")
    return resolved
