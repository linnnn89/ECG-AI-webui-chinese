from __future__ import annotations

import argparse
import csv
import hashlib
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
from torch.utils.data import DataLoader

from code.casebank.features import (
    build_retrieval_vectors,
    extract_basic_wave_features,
    fit_vector_stats,
)
from code.casebank.inference import (
    TARGET_LEN,
    load_npy_tensor,
    load_runtime,
    normalize_and_resample,
    predicted_labels,
    project_root,
)
from code.casebank.io import write_case_rows, write_json
from code.datasets.ptbxl import PTBXLWaveform


BUILD_VERSION = "casebank_mvp0_2026-05-28_reference"


def _case_id_from_record(record: str) -> str:
    stem = Path(str(record).replace("\\", "/")).stem
    return f"ptbxl_{stem}"


def _patient_hash(value: Any) -> str | None:
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return None
    return hashlib.sha256(f"PTBXL:{text}".encode("utf-8")).hexdigest()


def _stable_split(record_name: str, val_ratio: float, seed: int) -> str:
    key = f"{seed}:{record_name}".encode("utf-8")
    value = int(hashlib.md5(key).hexdigest()[:8], 16) / 0xFFFFFFFF
    return "val" if value < val_ratio else "train"


def _iter_splits(split: str) -> List[str]:
    if split in {"all", "reference"}:
        return ["train", "val", "test"]
    return [split]


def _chapman_split_matches(requested_split: str, stable_split: str) -> bool:
    if requested_split in {"all", "reference"}:
        return True
    if requested_split == "test":
        return False
    return requested_split == stable_split


def _prepare_out_dir(out_dir: str, overwrite: bool) -> None:
    out_path = Path(out_dir)
    generated = [
        "case_index.sqlite",
        "case_probs.npy",
        "case_margins.npy",
        "case_wave_features.npy",
        "case_retrieval_vectors.npy",
        "vector_stats.json",
        "build_config.json",
    ]
    if out_path.exists() and any((out_path / name).exists() for name in generated):
        if not overwrite:
            raise FileExistsError(
                f"{out_dir} already contains CaseBank files. Pass --overwrite 1 to rebuild."
            )
        shutil.rmtree(out_path)
    out_path.mkdir(parents=True, exist_ok=True)


def _ensure_dataset_classes(ds: PTBXLWaveform, runtime_classes: List[str]) -> None:
    if list(ds.classes) != list(runtime_classes):
        raise ValueError(
            "PTB-XL label columns do not match model classes.\n"
            f"dataset: {list(ds.classes)}\nmodel:   {list(runtime_classes)}"
        )


def _labels_from_target(classes: List[str], y: np.ndarray) -> List[str]:
    return [cls for cls, flag in zip(classes, y.astype(float)) if flag >= 0.5]


def _relative_to_project(path: str) -> str:
    try:
        return os.path.relpath(os.path.abspath(path), project_root())
    except ValueError:
        return os.path.abspath(path)


def _append_case(
    *,
    rows: List[Dict[str, Any]],
    probs_list: List[np.ndarray],
    margins_list: List[np.ndarray],
    wave_list: List[np.ndarray],
    row_id: int,
    runtime,
    x: np.ndarray,
    y: np.ndarray,
    case_id: str,
    source: str,
    source_record_id: str,
    record_path: str,
    header_path: str | None,
    patient_id_hash: str | None,
    split: str,
    created_at: str,
    signal_quality: Dict[str, Any] | None = None,
    probabilities: np.ndarray | None = None,
) -> int:
    prob = runtime.predict_one(x) if probabilities is None else np.asarray(probabilities, dtype=np.float32)
    margins = prob - runtime.threshold_array
    wave_features = extract_basic_wave_features(x)
    true_labels = _labels_from_target(runtime.classes, y)
    pred_labels = predicted_labels(runtime.classes, prob, runtime.thresholds)
    rows.append(
        {
            "row_id": row_id,
            "case_id": case_id,
            "source": source,
            "source_record_id": source_record_id,
            "record_path": record_path,
            "header_path": header_path,
            "image_path": None,
            "patient_id_hash": patient_id_hash,
            "split": split,
            "labels": true_labels,
            "predicted_labels": pred_labels,
            "probabilities": {cls: float(v) for cls, v in zip(runtime.classes, prob)},
            "margins": {cls: float(v) for cls, v in zip(runtime.classes, margins)},
            "signal_quality": signal_quality or {},
            "has_embedding": False,
            "build_version": BUILD_VERSION,
            "created_at": created_at,
        }
    )
    probs_list.append(prob.astype(np.float32, copy=False))
    margins_list.append(margins.astype(np.float32, copy=False))
    wave_list.append(wave_features)
    return row_id + 1


def _selected_sources(raw_sources: str) -> List[str]:
    aliases = {
        "ptbxl_chapman": ["ptbxl", "chapman"],
        "all": ["ptbxl", "chapman"],
    }
    sources: List[str] = []
    for item in raw_sources.split(","):
        key = item.strip().lower()
        if not key:
            continue
        expanded = aliases.get(key, [key])
        for source in expanded:
            if source not in {"ptbxl", "chapman"}:
                raise ValueError(f"Unsupported source: {source}")
            if source not in sources:
                sources.append(source)
    if not sources:
        raise ValueError("--sources must include at least one source")
    return sources


def _resolve_chapman_npy(chapman_root: str, record_name: str, raw_path: str) -> str | None:
    fallback = os.path.join(chapman_root, "signals_npy", f"{record_name}.npy")
    if os.path.exists(fallback):
        return fallback
    if raw_path and os.path.exists(raw_path):
        return raw_path
    return None


def _build_ptbxl_cases(
    *,
    args: argparse.Namespace,
    runtime,
    rows: List[Dict[str, Any]],
    probs_list: List[np.ndarray],
    margins_list: List[np.ndarray],
    wave_list: List[np.ndarray],
    row_id: int,
    created_at: str,
) -> Tuple[int, Dict[str, Any]]:
    ptbxl_root = os.path.abspath(args.ptbxl_root)
    stats = {"source": "PTBXL", "collected": 0, "missing": 0, "skipped_unmapped": 0}
    for split in _iter_splits(args.split):
        ds = PTBXLWaveform(ptbxl_root, split=split, cache=True)
        _ensure_dataset_classes(ds, runtime.classes)
        limit = len(ds) if args.limit <= 0 else min(args.limit - stats["collected"], len(ds))
        if limit <= 0:
            break
        loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
        seen = 0
        for xb, yb in loader:
            if seen >= limit:
                break
            batch_take = min(len(xb), limit - seen)
            x_np = xb[:batch_take].numpy().astype(np.float32, copy=False)
            y_np = yb[:batch_take].numpy().astype(np.float32, copy=False)
            if x_np.shape[-1] != TARGET_LEN:
                x_np = np.stack([normalize_and_resample(x) for x in x_np], axis=0)
            prob_batch = runtime.predict_batch(x_np)
            for j in range(batch_take):
                meta = ds.meta.iloc[seen + j]
                record = str(meta["record"])
                row_id = _append_case(
                    rows=rows,
                    probs_list=probs_list,
                    margins_list=margins_list,
                    wave_list=wave_list,
                    row_id=row_id,
                    runtime=runtime,
                    x=x_np[j],
                    y=y_np[j],
                    case_id=_case_id_from_record(record),
                    source="PTBXL",
                    source_record_id=str(meta.get("ecg_id", "")),
                    record_path=record,
                    header_path=f"{record}.hea",
                    patient_id_hash=_patient_hash(meta.get("patient_id")),
                    split=str(meta.get("split", split)),
                    created_at=created_at,
                    signal_quality={"label_scope": "current_10class"},
                    probabilities=prob_batch[j],
                )
                stats["collected"] += 1
            seen += batch_take
    return row_id, stats


def _build_chapman_cases(
    *,
    args: argparse.Namespace,
    runtime,
    rows: List[Dict[str, Any]],
    probs_list: List[np.ndarray],
    margins_list: List[np.ndarray],
    wave_list: List[np.ndarray],
    row_id: int,
    created_at: str,
) -> Tuple[int, Dict[str, Any]]:
    chapman_root = os.path.abspath(args.chapman_root)
    csv_path = os.path.join(chapman_root, "ground_truth.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Chapman ground_truth.csv not found: {csv_path}")
    class_to_i = {cls: i for i, cls in enumerate(runtime.classes)}
    stats = {
        "source": "CHAPMAN",
        "collected": 0,
        "missing": 0,
        "skipped_unmapped": 0,
        "collected_current_10class": 0,
        "collected_out_of_scope_for_current_10class": 0,
        "collected_current_10class_plus_out_of_scope": 0,
    }
    with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for raw in reader:
            if args.limit > 0 and stats["collected"] >= args.limit:
                break
            record_name = raw["record_name"]
            stable = _stable_split(record_name, args.chapman_val_ratio, args.seed)
            if not _chapman_split_matches(args.split, stable):
                continue
            source_labels = [label.strip() for label in raw["labels"].split("|") if label.strip()]
            y = np.zeros(len(runtime.classes), dtype=np.float32)
            mapped = []
            out_of_scope = []
            for label in source_labels:
                if label in class_to_i:
                    y[class_to_i[label]] = 1.0
                    mapped.append(label)
                else:
                    out_of_scope.append(label)
            if not mapped and not args.chapman_include_empty_negative:
                stats["skipped_unmapped"] += 1
                continue
            if mapped and out_of_scope:
                label_scope = "current_10class_plus_out_of_scope"
            elif mapped:
                label_scope = "current_10class"
            else:
                label_scope = "out_of_scope_for_current_10class"
            npy_path = _resolve_chapman_npy(chapman_root, record_name, raw.get("npy_path", ""))
            if not npy_path:
                stats["missing"] += 1
                continue
            x, resolved_path = load_npy_tensor(npy_path)
            row_id = _append_case(
                rows=rows,
                probs_list=probs_list,
                margins_list=margins_list,
                wave_list=wave_list,
                row_id=row_id,
                runtime=runtime,
                x=x,
                y=y,
                case_id=f"chapman_{record_name}",
                source="CHAPMAN",
                source_record_id=record_name,
                record_path=_relative_to_project(resolved_path),
                header_path=None,
                patient_id_hash=None,
                split=stable,
                created_at=created_at,
                signal_quality={
                    "label_scope": label_scope,
                    "source_labels": source_labels,
                    "mapped_labels": mapped,
                    "out_of_scope_labels": out_of_scope,
                },
            )
            stats["collected"] += 1
            if label_scope == "current_10class":
                stats["collected_current_10class"] += 1
            elif label_scope == "current_10class_plus_out_of_scope":
                stats["collected_current_10class_plus_out_of_scope"] += 1
            else:
                stats["collected_out_of_scope_for_current_10class"] += 1
    return row_id, stats


def build_casebank(args: argparse.Namespace) -> Dict[str, Any]:
    root = project_root()
    ptbxl_root = os.path.abspath(args.ptbxl_root)
    model_dir = os.path.abspath(args.model_dir) if args.model_dir else None
    out_dir = os.path.abspath(args.out_dir)
    if not os.path.isdir(ptbxl_root):
        raise FileNotFoundError(f"PTB-XL path does not exist: {ptbxl_root}")

    sources = _selected_sources(args.sources)
    runtime = load_runtime(model_dir=model_dir, label_mode=args.label_mode)
    _prepare_out_dir(out_dir, overwrite=bool(args.overwrite))

    rows: List[Dict[str, Any]] = []
    probs_list: List[np.ndarray] = []
    margins_list: List[np.ndarray] = []
    wave_list: List[np.ndarray] = []
    created_at = datetime.now().isoformat(timespec="seconds")
    row_id = 0
    source_stats = []

    if "ptbxl" in sources:
        row_id, stats = _build_ptbxl_cases(
            args=args,
            runtime=runtime,
            rows=rows,
            probs_list=probs_list,
            margins_list=margins_list,
            wave_list=wave_list,
            row_id=row_id,
            created_at=created_at,
        )
        source_stats.append(stats)

    if "chapman" in sources:
        row_id, stats = _build_chapman_cases(
            args=args,
            runtime=runtime,
            rows=rows,
            probs_list=probs_list,
            margins_list=margins_list,
            wave_list=wave_list,
            row_id=row_id,
            created_at=created_at,
        )
        source_stats.append(stats)

    if not rows:
        raise ValueError("No cases were collected. Check --split and --limit.")

    probs = np.stack(probs_list, axis=0).astype(np.float32)
    margins = np.stack(margins_list, axis=0).astype(np.float32)
    wave_features = np.stack(wave_list, axis=0).astype(np.float32)
    vector_stats = fit_vector_stats(probs, margins, wave_features)
    retrieval_vectors = build_retrieval_vectors(probs, margins, wave_features, vector_stats)

    np.save(os.path.join(out_dir, "case_probs.npy"), probs)
    np.save(os.path.join(out_dir, "case_margins.npy"), margins)
    np.save(os.path.join(out_dir, "case_wave_features.npy"), wave_features)
    np.save(os.path.join(out_dir, "case_retrieval_vectors.npy"), retrieval_vectors)
    write_case_rows(os.path.join(out_dir, "case_index.sqlite"), rows)
    write_json(os.path.join(out_dir, "vector_stats.json"), vector_stats)
    build_config = {
        "label_mode": args.label_mode,
        "classes": runtime.classes,
        "classes_file": runtime.classes_file,
        "thresholds_file": runtime.thresholds_file,
        "model_file": runtime.onnx_path,
        "model_dir": runtime.model_dir,
        "out_dir": out_dir,
        "num_cases": len(rows),
        "has_embedding": False,
        "split": args.split,
        "sources": sources,
        "source_stats": source_stats,
        "ptbxl_root": ptbxl_root,
        "chapman_root": os.path.abspath(args.chapman_root),
        "chapman_val_ratio": args.chapman_val_ratio,
        "seed": args.seed,
        "generate_images": bool(args.generate_images),
        "created_at": created_at,
        "build_version": BUILD_VERSION,
        "git_commit": None,
    }
    write_json(os.path.join(out_dir, "build_config.json"), build_config)
    Path(os.path.join(out_dir, "images")).mkdir(exist_ok=True)
    Path(os.path.join(out_dir, "preview")).mkdir(exist_ok=True)
    return build_config


def parse_args() -> argparse.Namespace:
    root = project_root()
    parser = argparse.ArgumentParser(description="Build ECG CaseBank MVP-0 from PTB-XL and Chapman.")
    parser.add_argument("--sources", default="ptbxl", help="Comma-separated: ptbxl, chapman, or ptbxl,chapman")
    parser.add_argument("--ptbxl_root", default=os.path.join(root, "data", "ptbxl"))
    parser.add_argument("--chapman_root", default=os.path.join(root, "data", "chapman_converted"))
    parser.add_argument("--model_dir", default=None)
    parser.add_argument("--out_dir", default=os.path.join(root, "data", "casebank"))
    parser.add_argument("--label_mode", type=int, default=10)
    parser.add_argument(
        "--split",
        choices=["train", "val", "test", "all", "reference"],
        default="train",
        help="train/val/test for split-specific banks; all/reference builds the public reference bank.",
    )
    parser.add_argument("--limit", type=int, default=0, help="0 means all; for multiple sources this is applied per source")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--chapman_val_ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--chapman_include_empty_negative", type=int, choices=[0, 1], default=0)
    parser.add_argument("--generate_images", type=int, choices=[0, 1], default=0)
    parser.add_argument("--overwrite", type=int, choices=[0, 1], default=0)
    return parser.parse_args()


def main() -> None:
    config = build_casebank(parse_args())
    print(f"Built CaseBank: {config['num_cases']} cases")
    print(f"Model: {config['model_dir']}")
    print(f"Output: {config['out_dir']}")


if __name__ == "__main__":
    main()
