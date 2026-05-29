from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np
from scipy.io import loadmat, whosmat


CLASSES_10 = ["NORM", "MI", "STTC", "LVH", "LBBB", "RBBB", "1AVB", "2AVB", "3AVB", "WPW"]
BUILD_VERSION = "cpsc2018_candidate_manifest_2026-05-29"
EXPECTED_RECORD_COUNT = 6877
CENTERED_10S_SAMPLES = 5000

CURRENT_CODE_TO_LABEL = {
    "426783006": "NORM",
    "270492004": "1AVB",
    "164909002": "LBBB",
    "59118001": "RBBB",
    "429622005": "STTC",
    "164931005": "STTC",
}

OUT_OF_SCOPE_CODES = {
    "164889003": "AF",
    "284470004": "PAC",
    "164884008": "ventricular_ectopics",
}


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(project_root()).as_posix()
    except ValueError:
        return str(path)


def write_csv(path: Path, rows: Iterable[Dict[str, object]], fieldnames: Sequence[str]) -> int:
    count = 0
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
            count += 1
    return count


def parse_header(path: Path) -> Dict[str, object]:
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    if not lines:
        return {
            "record_id": path.stem,
            "num_leads": 0,
            "fs_hz": 0,
            "num_samples_header": 0,
            "dx_codes": [],
            "header_error": "empty_header",
        }
    first = lines[0].split()
    if len(first) < 4:
        return {
            "record_id": path.stem,
            "num_leads": 0,
            "fs_hz": 0,
            "num_samples_header": 0,
            "dx_codes": [],
            "header_error": "malformed_first_header_line",
        }
    try:
        num_leads = int(first[1])
        fs_hz = int(first[2])
        num_samples = int(first[3])
    except ValueError:
        return {
            "record_id": path.stem,
            "num_leads": 0,
            "fs_hz": 0,
            "num_samples_header": 0,
            "dx_codes": [],
            "header_error": "non_numeric_header_fields",
        }
    dx_codes: List[str] = []
    for line in lines:
        if line.startswith("# Dx:"):
            dx_codes = [code.strip() for code in line.split(":", 1)[1].split(",") if code.strip()]
            break
    return {
        "record_id": first[0],
        "num_leads": num_leads,
        "fs_hz": fs_hz,
        "num_samples_header": num_samples,
        "dx_codes": dx_codes,
        "header_error": "",
    }


def mat_shape(path: Path) -> Dict[str, object]:
    try:
        entries = whosmat(path)
    except Exception as exc:  # pragma: no cover - defensive report path
        return {"mat_key": "", "mat_shape": "", "mat_dtype": "", "mat_error": repr(exc)}
    val_entries = [entry for entry in entries if entry[0] == "val"]
    entry = val_entries[0] if val_entries else (entries[0] if entries else ("", (), ""))
    shape = tuple(entry[1])
    return {
        "mat_key": entry[0],
        "mat_shape": "x".join(str(x) for x in shape),
        "mat_dtype": entry[2],
        "mat_error": "",
    }


def map_codes(dx_codes: Sequence[str]) -> Dict[str, object]:
    current: List[str] = []
    out_codes: List[str] = []
    unknown: List[str] = []
    has_non_norm = any(code != "426783006" for code in dx_codes)

    for code in dx_codes:
        if code == "426783006" and has_non_norm:
            continue
        if code in CURRENT_CODE_TO_LABEL:
            label = CURRENT_CODE_TO_LABEL[code]
            if label not in current:
                current.append(label)
        elif code in OUT_OF_SCOPE_CODES:
            out_codes.append(code)
        else:
            unknown.append(code)

    if current and out_codes:
        scope = "current_10class_plus_out_of_scope"
    elif current:
        scope = "current_10class"
    elif out_codes:
        scope = "out_of_scope_only"
    else:
        scope = "unmapped"

    notes = []
    if "426783006" in dx_codes and has_non_norm:
        notes.append("norm_code_suppressed_due_to_co_label")
    if unknown:
        notes.append("unknown_dx_code")

    return {
        "labels_10class": current,
        "out_of_scope_codes": out_codes,
        "unknown_dx_codes": unknown,
        "label_scope": scope,
        "mapping_notes": notes,
    }


def candidate_training_status(label_scope: str) -> str:
    if label_scope == "current_10class":
        return "candidate_current_10class_not_integrated"
    if label_scope == "current_10class_plus_out_of_scope":
        return "review_required_current_plus_out_of_scope_not_integrated"
    if label_scope == "out_of_scope_only":
        return "exclude_from_current_10class_training"
    return "exclude_until_mapping_resolved"


def build_manifest(args: argparse.Namespace) -> Dict[str, object]:
    root = Path(args.cpsc_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    hea_paths = sorted(root.rglob("*.hea"))
    mat_paths = sorted(root.rglob("*.mat"))
    hea_by_stem = {path.stem: path for path in hea_paths}
    mat_by_stem = {path.stem: path for path in mat_paths}
    all_stems = sorted(set(hea_by_stem) | set(mat_by_stem))

    rows: List[Dict[str, object]] = []
    group_counts: Counter[str] = Counter()
    code_counts: Counter[str] = Counter()
    label_counts: Counter[str] = Counter()
    out_code_counts: Counter[str] = Counter()
    unknown_code_counts: Counter[str] = Counter()
    scope_counts: Counter[str] = Counter()
    fs_counts: Counter[str] = Counter()
    lead_counts: Counter[str] = Counter()
    mat_key_counts: Counter[str] = Counter()
    mat_dtype_counts: Counter[str] = Counter()
    sample_lengths: List[int] = []
    valid_header_count = 0
    missing_hea: List[str] = []
    missing_mat: List[str] = []
    mat_shape_mismatches: List[str] = []
    header_parse_errors: List[str] = []
    qc_issues: List[str] = []

    for stem in all_stems:
        hea_path = hea_by_stem.get(stem)
        mat_path = mat_by_stem.get(stem)
        if hea_path is None:
            missing_hea.append(stem)
            continue
        if mat_path is None:
            missing_mat.append(stem)

        header = parse_header(hea_path)
        mapped = map_codes(header["dx_codes"])
        mat_info = mat_shape(mat_path) if mat_path else {"mat_key": "", "mat_shape": "", "mat_dtype": "", "mat_error": "missing_mat"}
        seconds = float(header["num_samples_header"]) / float(header["fs_hz"]) if header["fs_hz"] else 0.0
        group = hea_path.parent.name

        dx_codes = list(header["dx_codes"])
        labels = list(mapped["labels_10class"])
        out_codes = list(mapped["out_of_scope_codes"])
        unknown_codes = list(mapped["unknown_dx_codes"])
        notes = list(mapped["mapping_notes"])
        header_error = str(header.get("header_error", ""))
        if header_error:
            header_parse_errors.append(stem)
            qc_issues.append(f"{stem}:header_parse_error={header_error}")
            notes.append("header_parse_error")
        else:
            if header["num_leads"] != 12:
                qc_issues.append(f"{stem}:num_leads={header['num_leads']}")
                notes.append("non_12_lead")
            if header["fs_hz"] != 500:
                qc_issues.append(f"{stem}:fs_hz={header['fs_hz']}")
                notes.append("non_500hz")
            if header["num_samples_header"] < CENTERED_10S_SAMPLES:
                qc_issues.append(f"{stem}:shorter_than_10s")
                notes.append("shorter_than_centered_10s")
        if mat_info["mat_key"] != "val":
            qc_issues.append(f"{stem}:mat_key={mat_info['mat_key']}")
        mat_shape_value = str(mat_info["mat_shape"])
        expected_shape = f"{header['num_leads']}x{header['num_samples_header']}"
        if not header_error and mat_shape_value and mat_shape_value != expected_shape:
            mat_shape_mismatches.append(stem)
            notes.append("mat_shape_header_mismatch")

        row = {
            "record_id": stem,
            "group": group,
            "hea_path": rel(hea_path),
            "mat_path": rel(mat_path) if mat_path else "",
            "num_leads": header["num_leads"],
            "fs_hz": header["fs_hz"],
            "num_samples": header["num_samples_header"],
            "seconds": f"{seconds:.3f}",
            "mat_key": mat_info["mat_key"],
            "mat_shape": mat_info["mat_shape"],
            "mat_dtype": mat_info["mat_dtype"],
            "header_error": header_error,
            "dx_codes": "|".join(dx_codes),
            "labels_10class": "|".join(labels),
            "out_of_scope_codes": "|".join(out_codes),
            "unknown_dx_codes": "|".join(unknown_codes),
            "label_scope": mapped["label_scope"],
            "centered_10s_policy": "direct_center_crop" if header["num_samples_header"] >= CENTERED_10S_SAMPLES else "requires_padding_or_exclusion",
            "candidate_casebank_status": "candidate_after_adapter_qc",
            "candidate_training_status": candidate_training_status(str(mapped["label_scope"])),
            "mapping_notes": "|".join(notes),
            "build_version": BUILD_VERSION,
        }
        rows.append(row)

        group_counts[group] += 1
        if not header_error:
            valid_header_count += 1
            fs_counts[str(header["fs_hz"])] += 1
            lead_counts[str(header["num_leads"])] += 1
            sample_lengths.append(int(header["num_samples_header"]))
        mat_key_counts[str(mat_info["mat_key"])] += 1
        mat_dtype_counts[str(mat_info["mat_dtype"])] += 1
        scope_counts[str(mapped["label_scope"])] += 1
        code_counts.update(dx_codes)
        label_counts.update(labels)
        out_code_counts.update(out_codes)
        unknown_code_counts.update(unknown_codes)

    manifest_path = out_dir / "cpsc2018_manifest.csv"
    fieldnames = [
        "record_id",
        "group",
        "hea_path",
        "mat_path",
        "num_leads",
        "fs_hz",
        "num_samples",
        "seconds",
        "mat_key",
        "mat_shape",
        "mat_dtype",
        "header_error",
        "dx_codes",
        "labels_10class",
        "out_of_scope_codes",
        "unknown_dx_codes",
        "label_scope",
        "centered_10s_policy",
        "candidate_casebank_status",
        "candidate_training_status",
        "mapping_notes",
        "build_version",
    ]
    write_csv(manifest_path, rows, fieldnames)

    label_summary_rows = []
    for label in CLASSES_10:
        label_summary_rows.append({"kind": "current_10class", "code_or_label": label, "count": label_counts[label]})
    for code, count in sorted(out_code_counts.items()):
        label_summary_rows.append({"kind": "out_of_scope_code", "code_or_label": code, "count": count})
    for code, count in sorted(unknown_code_counts.items()):
        label_summary_rows.append({"kind": "unknown_dx_code", "code_or_label": code, "count": count})
    for code, count in sorted(code_counts.items()):
        label_summary_rows.append({"kind": "raw_dx_code", "code_or_label": code, "count": count})
    write_csv(out_dir / "cpsc2018_label_summary.csv", label_summary_rows, ["kind", "code_or_label", "count"])

    sample_rows = sample_waveforms(rows, args.sample_size, args.seed)
    write_csv(
        out_dir / "cpsc2018_sample_waveform_qc.csv",
        sample_rows,
        ["record_id", "group", "loaded_key", "loaded_shape", "finite_fraction", "matches_header_shape", "load_error"],
    )

    summary = {
        "build_version": BUILD_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_root": str(root),
        "manifest_path": str(manifest_path),
        "expected_record_count": EXPECTED_RECORD_COUNT,
        "hea_count": len(hea_paths),
        "mat_count": len(mat_paths),
        "matched_record_count": len(rows),
        "valid_header_count": valid_header_count,
        "complete_expected_count": len(hea_paths) == EXPECTED_RECORD_COUNT and len(mat_paths) == EXPECTED_RECORD_COUNT,
        "missing_hea_count": len(missing_hea),
        "missing_mat_count": len(missing_mat),
        "missing_hea_first20": missing_hea[:20],
        "missing_mat_first20": missing_mat[:20],
        "group_counts": dict(sorted(group_counts.items())),
        "fs_counts": dict(sorted(fs_counts.items())),
        "lead_counts": dict(sorted(lead_counts.items())),
        "mat_key_counts": dict(sorted(mat_key_counts.items())),
        "mat_dtype_counts": dict(sorted(mat_dtype_counts.items())),
        "mat_shape_mismatch_count": len(mat_shape_mismatches),
        "mat_shape_mismatch_first20": mat_shape_mismatches[:20],
        "header_parse_error_count": len(header_parse_errors),
        "header_parse_error_first20": header_parse_errors[:20],
        "min_samples": min(sample_lengths) if sample_lengths else None,
        "max_samples": max(sample_lengths) if sample_lengths else None,
        "records_shorter_than_10s": sum(1 for n in sample_lengths if n < CENTERED_10S_SAMPLES),
        "label_scope_counts": dict(sorted(scope_counts.items())),
        "current_10class_positive_counts": {label: label_counts[label] for label in CLASSES_10},
        "out_of_scope_code_counts": dict(sorted(out_code_counts.items())),
        "unknown_dx_code_counts": dict(sorted(unknown_code_counts.items())),
        "raw_dx_code_counts": dict(sorted(code_counts.items())),
        "qc_issue_count": len(qc_issues),
        "qc_issues_first50": qc_issues[:50],
        "sample_waveform_qc_path": str(out_dir / "cpsc2018_sample_waveform_qc.csv"),
        "sample_waveform_qc_pass": all(row["load_error"] == "" and row["matches_header_shape"] == "1" for row in sample_rows),
    }
    (out_dir / "cpsc2018_qc_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def sample_waveforms(rows: Sequence[Dict[str, object]], sample_size: int, seed: int) -> List[Dict[str, object]]:
    rng = random.Random(seed)
    sample = rng.sample(list(rows), min(sample_size, len(rows))) if rows else []
    sample_rows: List[Dict[str, object]] = []
    root = project_root()
    for row in sample:
        load_error = ""
        loaded_key = ""
        loaded_shape = ""
        finite_fraction = ""
        matches_header_shape = "0"
        try:
            mat_path = root / str(row["mat_path"])
            data = loadmat(mat_path)
            arr = data.get("val")
            loaded_key = "val" if arr is not None else ""
            if arr is None:
                load_error = "missing_val_key"
            else:
                loaded_shape = "x".join(str(x) for x in arr.shape)
                total = arr.size
                finite = int(np.isfinite(arr).sum())
                finite_fraction = f"{finite / total:.6f}" if total else "0.000000"
                expected = f"{row['num_leads']}x{row['num_samples']}"
                matches_header_shape = "1" if loaded_shape == expected else "0"
        except Exception as exc:  # pragma: no cover - defensive report path
            load_error = repr(exc)
        sample_rows.append(
            {
                "record_id": row["record_id"],
                "group": row["group"],
                "loaded_key": loaded_key,
                "loaded_shape": loaded_shape,
                "finite_fraction": finite_fraction,
                "matches_header_shape": matches_header_shape,
                "load_error": load_error,
            }
        )
    return sample_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build CPSC2018 candidate manifest and QC tables.")
    parser.add_argument("--cpsc-root", default=str(project_root() / "data/challenge_2020/cpsc_2018"))
    parser.add_argument("--out-dir", default=str(project_root() / "data/challenge_2020/cpsc_2018_manifest"))
    parser.add_argument("--sample-size", type=int, default=24)
    parser.add_argument("--seed", type=int, default=20260529)
    return parser.parse_args()


def main() -> None:
    summary = build_manifest(parse_args())
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
