from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import sqlite3
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import numpy as np

from code.casebank.features import build_retrieval_vectors, extract_basic_wave_features, fit_vector_stats
from code.casebank.inference import load_runtime, predicted_labels, project_root
from code.casebank.io import write_case_rows, write_json
from code.casebank.render import render_cache_path
from code.casebank.waveform_io import load_case_tensor


BUILD_VERSION = "casebank_full_vector_index_2026-05-29"
CLASSES_10 = ["NORM", "MI", "STTC", "LVH", "LBBB", "RBBB", "1AVB", "2AVB", "3AVB", "WPW"]
CPSC_OUT_OF_SCOPE_NAMES = {
    "164889003": "AF",
    "284470004": "PAC",
    "164884008": "ventricular_ectopics",
}


def _project_rel(path: str | os.PathLike[str]) -> str:
    p = Path(path)
    try:
        return p.resolve().relative_to(Path(project_root()).resolve()).as_posix()
    except ValueError:
        return str(p)


def _pipe(value: str | None) -> List[str]:
    out: List[str] = []
    for item in str(value or "").replace("｜", "|").split("|"):
        text = item.strip()
        if text and text not in out:
            out.append(text)
    return out


def _diagnosis_text(labels: Sequence[str], out_scope: Sequence[str], raw: str | None, label_scope: str) -> str:
    parts = []
    if labels:
        parts.append("|".join(labels))
    if out_scope:
        parts.append("out_of_scope=" + "|".join(out_scope))
    if raw and raw not in parts:
        parts.append(str(raw))
    return "; ".join(parts) if parts else label_scope


def _read_sqlite_rows(path: Path) -> List[Dict[str, str]]:
    with sqlite3.connect(path) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute("SELECT * FROM cases ORDER BY source, case_id").fetchall()
    return [dict(row) for row in rows]


def _read_csv_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def _display_records(display_sqlite: Path, cache_dir: Path) -> Iterable[Dict[str, Any]]:
    for row in _read_sqlite_rows(display_sqlite):
        labels_10 = _pipe(row.get("labels_10class"))
        out_scope = _pipe(row.get("out_of_scope_labels"))
        true_labels = labels_10 + [label for label in out_scope if label not in labels_10]
        source = row.get("source", "")
        case_id = row["case_id"]
        record_format = row.get("record_format", "")
        record_path = row.get("record_path") or row.get("npy_path") or row.get("wfdb_record")
        diagnosis = _diagnosis_text(labels_10, out_scope, row.get("diagnosis_text"), row.get("label_scope", ""))
        yield {
            "case_id": case_id,
            "source": source,
            "source_record_id": row.get("source_record_id", ""),
            "split": row.get("split", ""),
            "record_path": record_path,
            "header_path": row.get("header_path", ""),
            "patient_id_hash": row.get("patient_id_hash", "") or None,
            "image_path": _project_rel(render_cache_path(cache_dir, source, case_id)),
            "labels": true_labels,
            "signal_quality": {
                "source": source,
                "record_format": record_format,
                "label_scope": row.get("label_scope", ""),
                "labels_10class": labels_10,
                "out_of_scope_labels": out_scope,
                "source_labels_raw": row.get("source_labels_raw", ""),
                "true_diagnosis": diagnosis,
                "diagnosis_text": row.get("diagnosis_text", ""),
                "metadata_source": row.get("metadata_source", ""),
                "display_index_case_id": case_id,
            },
        }


def _cpsc_records(cpsc_manifest_csv: Path, cache_dir: Path) -> Iterable[Dict[str, Any]]:
    for row in _read_csv_rows(cpsc_manifest_csv):
        record_id = row["record_id"]
        case_id = f"cpsc2018_{record_id}"
        labels_10 = _pipe(row.get("labels_10class"))
        out_codes = _pipe(row.get("out_of_scope_codes"))
        out_scope = [CPSC_OUT_OF_SCOPE_NAMES.get(code, code) for code in out_codes]
        unknown = _pipe(row.get("unknown_dx_codes"))
        true_labels = labels_10 + [label for label in out_scope + unknown if label not in labels_10]
        label_scope = row.get("label_scope", "")
        diagnosis = _diagnosis_text(labels_10, out_scope + unknown, row.get("dx_codes"), label_scope)
        yield {
            "case_id": case_id,
            "source": "CPSC2018",
            "source_record_id": record_id,
            "split": row.get("group", ""),
            "record_path": row.get("mat_path", ""),
            "header_path": row.get("hea_path", ""),
            "patient_id_hash": None,
            "image_path": _project_rel(render_cache_path(cache_dir, "CPSC2018", case_id)),
            "labels": true_labels,
            "signal_quality": {
                "source": "CPSC2018",
                "record_format": "cpsc_mat",
                "mat_key": row.get("mat_key", "val"),
                "label_scope": label_scope,
                "labels_10class": labels_10,
                "out_of_scope_labels": out_scope,
                "out_of_scope_codes": out_codes,
                "unknown_dx_codes": unknown,
                "dx_codes": row.get("dx_codes", ""),
                "true_diagnosis": diagnosis,
                "diagnosis_text": diagnosis,
                "original_fs_hz": row.get("fs_hz", ""),
                "original_num_samples": row.get("num_samples", ""),
                "seconds": row.get("seconds", ""),
                "centered_10s_policy": row.get("centered_10s_policy", ""),
                "shorter_than_10s": float(row.get("seconds", "0") or 0) < 10.0,
            },
        }


def _safe_prepare_out_dir(out_dir: Path, overwrite: bool) -> None:
    root = Path(project_root()).resolve()
    resolved = out_dir.resolve()
    if root not in [resolved, *resolved.parents]:
        raise ValueError(f"Refusing to write CaseBank outside project root: {resolved}")
    generated = {
        "case_index.sqlite",
        "case_probs.npy",
        "case_margins.npy",
        "case_wave_features.npy",
        "case_retrieval_vectors.npy",
        "vector_stats.json",
        "build_config.json",
    }
    if out_dir.exists() and generated.intersection({p.name for p in out_dir.iterdir()}):
        if not overwrite:
            raise FileExistsError(f"{out_dir} already contains CaseBank files. Pass --overwrite 1.")
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)


def _flush_batch(
    *,
    runtime,
    batch_meta: List[Dict[str, Any]],
    batch_x: List[np.ndarray],
    rows: List[Dict[str, Any]],
    probs_list: List[np.ndarray],
    margins_list: List[np.ndarray],
    wave_list: List[np.ndarray],
    created_at: str,
    row_start: int,
) -> int:
    if not batch_meta:
        return row_start
    xb = np.stack(batch_x, axis=0).astype(np.float32, copy=False)
    probs = runtime.predict_batch(xb)
    row_id = row_start
    for meta, x, prob in zip(batch_meta, xb, probs):
        margins = prob - runtime.threshold_array
        pred = predicted_labels(runtime.classes, prob, runtime.thresholds)
        rows.append(
            {
                "row_id": row_id,
                "case_id": meta["case_id"],
                "source": meta["source"],
                "source_record_id": meta.get("source_record_id"),
                "record_path": meta.get("record_path"),
                "header_path": meta.get("header_path"),
                "image_path": meta.get("image_path"),
                "patient_id_hash": meta.get("patient_id_hash"),
                "split": meta.get("split"),
                "labels": meta.get("labels", []),
                "predicted_labels": pred,
                "probabilities": {cls: float(value) for cls, value in zip(runtime.classes, prob)},
                "margins": {cls: float(value) for cls, value in zip(runtime.classes, margins)},
                "signal_quality": meta.get("signal_quality", {}),
                "has_embedding": False,
                "build_version": BUILD_VERSION,
                "created_at": created_at,
            }
        )
        probs_list.append(prob.astype(np.float32, copy=False))
        margins_list.append(margins.astype(np.float32, copy=False))
        wave_list.append(extract_basic_wave_features(x))
        row_id += 1
    return row_id


def build(args: argparse.Namespace) -> Dict[str, Any]:
    root = Path(project_root())
    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = root / out_dir
    cache_dir = Path(args.cache_dir)
    if not cache_dir.is_absolute():
        cache_dir = root / cache_dir
    cache_dir.mkdir(parents=True, exist_ok=True)
    for source in ["Hospital", "PTBXL_500", "Chapman", "CPSC2018"]:
        Path(render_cache_path(cache_dir, source, "_placeholder")).parent.mkdir(parents=True, exist_ok=True)

    display_sqlite = Path(args.display_sqlite)
    if not display_sqlite.is_absolute():
        display_sqlite = root / display_sqlite
    cpsc_manifest = Path(args.cpsc_manifest_csv)
    if not cpsc_manifest.is_absolute():
        cpsc_manifest = root / cpsc_manifest
    if not display_sqlite.exists():
        raise FileNotFoundError(display_sqlite)
    if not cpsc_manifest.exists():
        raise FileNotFoundError(cpsc_manifest)

    runtime = load_runtime(model_dir=args.model_dir, label_mode=10)
    _safe_prepare_out_dir(out_dir, bool(args.overwrite))

    records = list(_display_records(display_sqlite, cache_dir))
    if args.include_cpsc:
        records.extend(_cpsc_records(cpsc_manifest, cache_dir))
    records = sorted(records, key=lambda r: (str(r["source"]), str(r["case_id"])))
    if args.limit > 0:
        records = records[: args.limit]

    rows: List[Dict[str, Any]] = []
    probs_list: List[np.ndarray] = []
    margins_list: List[np.ndarray] = []
    wave_list: List[np.ndarray] = []
    skipped: List[Dict[str, str]] = []
    source_seen: Counter[str] = Counter()
    source_kept: Counter[str] = Counter()
    label_scope_kept: Counter[str] = Counter()
    created_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    batch_meta: List[Dict[str, Any]] = []
    batch_x: List[np.ndarray] = []
    row_id = 0

    for idx, meta in enumerate(records, start=1):
        source_seen[str(meta["source"])] += 1
        try:
            x = load_case_tensor(meta["record_path"], meta.get("signal_quality"))
        except Exception as exc:  # noqa: BLE001 - keep build going and report skipped records.
            skipped.append({"case_id": meta["case_id"], "source": meta["source"], "error": f"{type(exc).__name__}: {exc}"})
            continue
        batch_meta.append(meta)
        batch_x.append(x)
        source_kept[str(meta["source"])] += 1
        label_scope_kept[str((meta.get("signal_quality") or {}).get("label_scope", ""))] += 1
        if len(batch_meta) >= args.batch_size:
            row_id = _flush_batch(
                runtime=runtime,
                batch_meta=batch_meta,
                batch_x=batch_x,
                rows=rows,
                probs_list=probs_list,
                margins_list=margins_list,
                wave_list=wave_list,
                created_at=created_at,
                row_start=row_id,
            )
            batch_meta.clear()
            batch_x.clear()
            print(f"processed={idx} kept={row_id} skipped={len(skipped)}", flush=True)
    row_id = _flush_batch(
        runtime=runtime,
        batch_meta=batch_meta,
        batch_x=batch_x,
        rows=rows,
        probs_list=probs_list,
        margins_list=margins_list,
        wave_list=wave_list,
        created_at=created_at,
        row_start=row_id,
    )

    if not rows:
        raise ValueError("No CaseBank records were built.")

    probs = np.stack(probs_list, axis=0).astype(np.float32)
    margins = np.stack(margins_list, axis=0).astype(np.float32)
    wave_features = np.stack(wave_list, axis=0).astype(np.float32)
    vector_stats = fit_vector_stats(probs, margins, wave_features)
    retrieval_vectors = build_retrieval_vectors(probs, margins, wave_features, vector_stats)

    np.save(out_dir / "case_probs.npy", probs)
    np.save(out_dir / "case_margins.npy", margins)
    np.save(out_dir / "case_wave_features.npy", wave_features)
    np.save(out_dir / "case_retrieval_vectors.npy", retrieval_vectors)
    write_case_rows(out_dir / "case_index.sqlite", rows)
    write_json(out_dir / "vector_stats.json", vector_stats)

    build_config = {
        "build_version": BUILD_VERSION,
        "created_at_utc": created_at,
        "classes": runtime.classes,
        "classes_file": runtime.classes_file,
        "thresholds_file": runtime.thresholds_file,
        "model_file": runtime.onnx_path,
        "model_dir": runtime.model_dir,
        "out_dir": str(out_dir),
        "cache_dir": str(cache_dir),
        "display_sqlite": str(display_sqlite),
        "cpsc_manifest_csv": str(cpsc_manifest),
        "include_cpsc": bool(args.include_cpsc),
        "num_cases_seen": len(records),
        "num_cases": len(rows),
        "skipped_count": len(skipped),
        "skipped_first50": skipped[:50],
        "source_seen_counts": dict(source_seen),
        "source_counts": dict(source_kept),
        "label_scope_counts": dict(label_scope_kept),
        "retrieval_vector_dim": int(retrieval_vectors.shape[1]),
        "retrieval_features": "zscore(probabilities)+zscore(threshold_margins)+zscore(basic_wave_features)",
        "has_embedding": False,
        "score_policy": "service/search default returns nearest top 10 with score_threshold=0.0",
        "cpsc_extra_policy": "data/challenge_2020/cpsc_2018_extra is not included because it has no separate QC manifest in this build.",
    }
    write_json(out_dir / "build_config.json", build_config)
    if skipped:
        write_json(out_dir / "skipped_records.json", skipped)
    return build_config


def parse_args() -> argparse.Namespace:
    root = project_root()
    parser = argparse.ArgumentParser(description="Build full ECG CaseBank vector index from display metadata plus CPSC2018.")
    parser.add_argument("--display_sqlite", default=os.path.join(root, "data", "casebank_display_assets", "case_display_index.sqlite"))
    parser.add_argument("--cpsc_manifest_csv", default=os.path.join(root, "data", "challenge_2020", "cpsc_2018_manifest", "cpsc2018_manifest.csv"))
    parser.add_argument("--model_dir", default=None)
    parser.add_argument("--out_dir", default=os.path.join(root, "data", "casebank_vector_index"))
    parser.add_argument("--cache_dir", default=os.path.join(root, "casebank_cache"))
    parser.add_argument("--include_cpsc", type=int, choices=[0, 1], default=1)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--overwrite", type=int, choices=[0, 1], default=0)
    return parser.parse_args()


def main() -> None:
    print(json.dumps(build(parse_args()), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
