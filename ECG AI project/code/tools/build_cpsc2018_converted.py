from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np
from scipy.io import loadmat


CLASSES_10 = ["NORM", "MI", "STTC", "LVH", "LBBB", "RBBB", "1AVB", "2AVB", "3AVB", "WPW"]
BUILD_VERSION = "cpsc2018_current_10class_converted_2026-05-29"
TARGET_SAMPLES = 5000


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: Iterable[Dict[str, object]], fieldnames: Sequence[str]) -> int:
    count = 0
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
            count += 1
    return count


def rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(project_root()).as_posix()
    except ValueError:
        return str(path)


def labels_from_pipe(value: str) -> List[str]:
    labels = []
    for item in (value or "").replace("｜", "|").split("|"):
        label = item.strip().upper()
        if label and label not in labels:
            labels.append(label)
    return labels


def validate_labels(record_id: str, labels: Sequence[str]) -> None:
    unknown = [label for label in labels if label not in CLASSES_10]
    if unknown:
        raise ValueError(f"{record_id}: unknown labels {unknown}")
    if not labels:
        raise ValueError(f"{record_id}: empty current 10-class labels")
    if "NORM" in labels and len(labels) > 1:
        raise ValueError(f"{record_id}: NORM cannot coexist with abnormal labels")


def waveform_sha256(x: np.ndarray) -> str:
    arr = np.ascontiguousarray(x)
    return hashlib.sha256(arr.tobytes()).hexdigest()


def centered_crop(x: np.ndarray, target_samples: int) -> tuple[np.ndarray, int, int]:
    n = int(x.shape[1])
    if n < target_samples:
        raise ValueError(f"cannot center-crop {n} samples to {target_samples}")
    start = (n - target_samples) // 2
    end = start + target_samples
    return x[:, start:end], start, end


def load_signal(mat_path: Path) -> np.ndarray:
    data = loadmat(mat_path)
    if "val" not in data:
        raise ValueError(f"{mat_path}: missing key 'val'")
    x = np.asarray(data["val"])
    if x.ndim != 2:
        raise ValueError(f"{mat_path}: expected 2D array, got {x.shape}")
    if x.shape[0] != 12 and x.shape[1] == 12:
        x = x.T
    if x.shape[0] != 12:
        raise ValueError(f"{mat_path}: expected 12 leads, got {x.shape}")
    x = np.nan_to_num(x.astype(np.float32, copy=False), nan=0.0, posinf=0.0, neginf=0.0)
    return x


def conversion_exclusion_reason(row: Dict[str, str]) -> str:
    if row.get("header_error", ""):
        return "header_parse_error"
    if row.get("label_scope", "") != "current_10class":
        return f"label_scope_{row.get('label_scope', 'missing')}"
    if row.get("centered_10s_policy", "") != "direct_center_crop":
        return "shorter_than_10s_or_no_direct_crop"
    if row.get("fs_hz", "") != "500":
        return "non_500hz"
    if row.get("num_leads", "") != "12":
        return "non_12_lead"
    if row.get("mat_key", "") != "val":
        return "missing_val_key"
    return ""


def attach_label_columns(row: Dict[str, object], labels: Sequence[str]) -> None:
    for label in CLASSES_10:
        row[f"label_{label}"] = 1 if label in labels else 0


def build_converted(args: argparse.Namespace) -> Dict[str, object]:
    root = project_root()
    manifest_path = Path(args.manifest_csv)
    out_dir = Path(args.out_dir)
    signals_dir = out_dir / "signals_npy"
    out_dir.mkdir(parents=True, exist_ok=True)
    signals_dir.mkdir(parents=True, exist_ok=True)

    manifest_rows = read_csv(manifest_path)
    converted_rows: List[Dict[str, object]] = []
    excluded_rows: List[Dict[str, object]] = []
    label_counts: Counter[str] = Counter()
    group_counts: Counter[str] = Counter()
    crop_start_values: List[int] = []
    source_length_values: List[int] = []
    conversion_errors: List[str] = []

    for row in manifest_rows:
        record_id = row["record_id"].strip()
        labels = labels_from_pipe(row.get("labels_10class", ""))
        reason = conversion_exclusion_reason(row)
        if not reason:
            try:
                validate_labels(record_id, labels)
            except ValueError as exc:
                reason = f"label_validation_error:{exc}"

        if reason:
            excluded_rows.append(
                {
                    "record_id": record_id,
                    "group": row.get("group", ""),
                    "reason": reason,
                    "label_scope": row.get("label_scope", ""),
                    "labels_10class": row.get("labels_10class", ""),
                    "out_of_scope_codes": row.get("out_of_scope_codes", ""),
                    "unknown_dx_codes": row.get("unknown_dx_codes", ""),
                    "num_samples": row.get("num_samples", ""),
                    "seconds": row.get("seconds", ""),
                    "dx_codes": row.get("dx_codes", ""),
                    "source_manifest": rel(manifest_path),
                    "build_version": BUILD_VERSION,
                }
            )
            continue

        try:
            mat_path = root / row["mat_path"]
            signal = load_signal(mat_path)
            cropped, crop_start, crop_end = centered_crop(signal, TARGET_SAMPLES)
            if cropped.shape != (12, TARGET_SAMPLES):
                raise ValueError(f"cropped shape is {cropped.shape}")
            out_path = signals_dir / f"{record_id}.npy"
            np.save(out_path, cropped.astype(np.float32, copy=False))
            sha256 = waveform_sha256(cropped.astype(np.float32, copy=False))
        except Exception as exc:
            error = repr(exc)
            conversion_errors.append(f"{record_id}:{error}")
            excluded_rows.append(
                {
                    "record_id": record_id,
                    "group": row.get("group", ""),
                    "reason": f"conversion_error:{error}",
                    "label_scope": row.get("label_scope", ""),
                    "labels_10class": row.get("labels_10class", ""),
                    "out_of_scope_codes": row.get("out_of_scope_codes", ""),
                    "unknown_dx_codes": row.get("unknown_dx_codes", ""),
                    "num_samples": row.get("num_samples", ""),
                    "seconds": row.get("seconds", ""),
                    "dx_codes": row.get("dx_codes", ""),
                    "source_manifest": rel(manifest_path),
                    "build_version": BUILD_VERSION,
                }
            )
            continue

        out_row: Dict[str, object] = {
            "record_name": record_id,
            "record_id": record_id,
            "source_dataset": "CPSC2018",
            "group": row.get("group", ""),
            "npy_path": rel(out_path),
            "labels": "|".join(labels),
            "labels_10class": "|".join(labels),
            "label_scope": row.get("label_scope", ""),
            "dx_codes": row.get("dx_codes", ""),
            "source_hea_path": row.get("hea_path", ""),
            "source_mat_path": row.get("mat_path", ""),
            "original_fs_hz": row.get("fs_hz", ""),
            "original_num_samples": row.get("num_samples", ""),
            "converted_fs_hz": 500,
            "converted_num_samples": TARGET_SAMPLES,
            "window_policy": "center_crop_10s_from_500hz_signal",
            "crop_start_sample": crop_start,
            "crop_end_sample_exclusive": crop_end,
            "waveform_sha256": sha256,
            "include_in_current_10class_conversion": 1,
            "build_version": BUILD_VERSION,
        }
        attach_label_columns(out_row, labels)
        converted_rows.append(out_row)
        label_counts.update(labels)
        group_counts.update([str(row.get("group", ""))])
        crop_start_values.append(crop_start)
        source_length_values.append(int(row.get("num_samples", 0)))

    converted_rows = sorted(converted_rows, key=lambda r: str(r["record_id"]))
    excluded_rows = sorted(excluded_rows, key=lambda r: str(r["record_id"]))

    converted_fields = [
        "record_name",
        "record_id",
        "source_dataset",
        "group",
        "npy_path",
        "labels",
        "labels_10class",
        "label_scope",
        "dx_codes",
        "source_hea_path",
        "source_mat_path",
        "original_fs_hz",
        "original_num_samples",
        "converted_fs_hz",
        "converted_num_samples",
        "window_policy",
        "crop_start_sample",
        "crop_end_sample_exclusive",
        "waveform_sha256",
        "include_in_current_10class_conversion",
        *[f"label_{label}" for label in CLASSES_10],
        "build_version",
    ]
    exclusion_fields = [
        "record_id",
        "group",
        "reason",
        "label_scope",
        "labels_10class",
        "out_of_scope_codes",
        "unknown_dx_codes",
        "num_samples",
        "seconds",
        "dx_codes",
        "source_manifest",
        "build_version",
    ]

    ground_truth_path = out_dir / "ground_truth.csv"
    converted_manifest_path = out_dir / "cpsc2018_converted_manifest.csv"
    excluded_path = out_dir / "cpsc2018_conversion_excluded.csv"
    write_csv(ground_truth_path, converted_rows, converted_fields)
    write_csv(converted_manifest_path, converted_rows, converted_fields)
    write_csv(excluded_path, excluded_rows, exclusion_fields)

    exclusion_reason_counts = Counter(str(row["reason"]) for row in excluded_rows)
    label_count_rows = [{"label": label, "positive_count": label_counts[label]} for label in CLASSES_10]
    write_csv(out_dir / "cpsc2018_converted_label_counts.csv", label_count_rows, ["label", "positive_count"])

    expected_convertible = sum(
        1
        for row in manifest_rows
        if row.get("label_scope", "") == "current_10class"
        and row.get("centered_10s_policy", "") == "direct_center_crop"
    )
    summary: Dict[str, object] = {
        "build_version": BUILD_VERSION,
        "built_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_manifest": str(manifest_path),
        "out_dir": str(out_dir),
        "signals_dir": str(signals_dir),
        "ground_truth_csv": str(ground_truth_path),
        "converted_manifest_csv": str(converted_manifest_path),
        "excluded_csv": str(excluded_path),
        "manifest_rows": len(manifest_rows),
        "expected_convertible_current_10class_direct_crop": expected_convertible,
        "converted_count": len(converted_rows),
        "excluded_count": len(excluded_rows),
        "conversion_error_count": len(conversion_errors),
        "conversion_errors_first20": conversion_errors[:20],
        "label_counts": {label: label_counts[label] for label in CLASSES_10},
        "group_counts_converted": dict(sorted(group_counts.items())),
        "exclusion_reason_counts": dict(sorted(exclusion_reason_counts.items())),
        "source_num_samples_min": min(source_length_values) if source_length_values else None,
        "source_num_samples_max": max(source_length_values) if source_length_values else None,
        "crop_start_sample_min": min(crop_start_values) if crop_start_values else None,
        "crop_start_sample_max": max(crop_start_values) if crop_start_values else None,
        "converted_shape": [12, TARGET_SAMPLES],
        "dtype": "float32",
        "qc_pass": len(converted_rows) == expected_convertible and len(conversion_errors) == 0,
        "policy": [
            "Converted only label_scope=current_10class records.",
            "Excluded current_10class_plus_out_of_scope records for now.",
            "Excluded out_of_scope_only records for now.",
            "Excluded records shorter than 10 seconds; no padding applied in this build.",
            "No final model probabilities, embeddings, or retrieval vectors were produced.",
        ],
    }
    (out_dir / "cpsc2018_converted_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_readme(out_dir, summary)
    return summary


def write_readme(out_dir: Path, summary: Dict[str, object]) -> None:
    text = f"""# CPSC2018 Converted Current 10-Class Records

Date: 2026-05-29

This folder contains CPSC2018 records converted for the current ECG AI 10-class label space.

Converted policy:

- Include only `label_scope=current_10class`.
- Exclude mixed current-class plus out-of-scope records for now.
- Exclude out-of-scope-only records for now.
- Exclude records shorter than 10 seconds; no padding is applied.
- Convert waveforms to centered 10-second `[12, 5000]` NumPy arrays at 500 Hz.

Main outputs:

- `signals_npy/`
- `ground_truth.csv`
- `cpsc2018_converted_manifest.csv`
- `cpsc2018_conversion_excluded.csv`
- `cpsc2018_converted_label_counts.csv`
- `cpsc2018_converted_summary.json`

Current build summary:

- Converted records: {summary["converted_count"]}
- Excluded records: {summary["excluded_count"]}
- QC pass: {summary["qc_pass"]}

This build is a converted candidate dataset only. It has not been merged into PTB-XL, Chapman, hospital data, CaseBank display assets, or final retrieval vectors.
"""
    (out_dir / "README.md").write_text(text, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert CPSC2018 current 10-class records to fixed 10-second NPY files.")
    parser.add_argument(
        "--manifest-csv",
        default=str(project_root() / "data/challenge_2020/cpsc_2018_manifest/cpsc2018_manifest.csv"),
    )
    parser.add_argument(
        "--out-dir",
        default=str(project_root() / "data/challenge_2020/cpsc_2018_converted_current10"),
    )
    return parser.parse_args()


def main() -> None:
    print(json.dumps(build_converted(parse_args()), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
