from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np
import onnxruntime as ort
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from torch.utils.data import DataLoader, Dataset


CLASSES_10 = ["NORM", "MI", "STTC", "LVH", "LBBB", "RBBB", "1AVB", "2AVB", "3AVB", "WPW"]
BUILD_VERSION = "cpsc2018_external_sanity_eval_2026-05-29"


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


def fmt4(x: object) -> str:
    if x is None:
        return ""
    try:
        value = float(x)
    except (TypeError, ValueError):
        return ""
    if not np.isfinite(value):
        return ""
    return f"{value:.4f}"


def load_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, -80, 80)
    return 1.0 / (1.0 + np.exp(-x))


def label_text(row: Sequence[int], classes: Sequence[str]) -> str:
    labels = [label for label, value in zip(classes, row) if int(value) == 1]
    return "|".join(labels) if labels else "NONE"


def labels_from_row(row: Dict[str, str], classes: Sequence[str]) -> np.ndarray:
    return np.array([1.0 if str(row.get(f"label_{label}", "0")).strip() == "1" else 0.0 for label in classes], dtype=np.float32)


class CPSCConvertedDataset(Dataset):
    def __init__(self, csv_path: Path, classes: Sequence[str], normalize: str = "z"):
        self.csv_path = Path(csv_path)
        self.root = project_root()
        self.classes = list(classes)
        self.normalize = normalize
        self.rows = read_csv(self.csv_path)
        self.targets = np.stack([labels_from_row(row, self.classes) for row in self.rows], axis=0).astype(np.float32)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int):
        row = self.rows[idx]
        path = self.root / row["npy_path"]
        x = np.load(path).astype(np.float32)
        if x.shape != (12, 5000):
            raise ValueError(f"{path}: expected shape (12, 5000), got {x.shape}")
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        if self.normalize == "z":
            mean = x.mean(axis=1, keepdims=True)
            std = x.std(axis=1, keepdims=True) + 1e-6
            x = (x - mean) / std
        return x.astype(np.float32, copy=False), self.targets[idx]


def load_model_info(model_dir: Path) -> Dict[str, object]:
    model_dir = Path(model_dir)
    classes = load_json(model_dir / "classes_5cls.json")
    thresholds = load_json(model_dir / "thresholds_5cls.json")
    onnx_path = model_dir / "inception_5cls.onnx"
    if not isinstance(classes, list) or classes != CLASSES_10:
        raise ValueError(f"Unexpected classes in {model_dir}: {classes}")
    if not onnx_path.exists():
        raise FileNotFoundError(f"Missing ONNX model: {onnx_path}")
    return {
        "model_dir": str(model_dir),
        "onnx_path": str(onnx_path),
        "classes": classes,
        "thresholds": {label: float(thresholds.get(label, 0.5)) for label in classes},
    }


def run_onnx(model_info: Dict[str, object], loader: DataLoader, providers: Sequence[str]) -> tuple[np.ndarray, np.ndarray]:
    sess = ort.InferenceSession(str(model_info["onnx_path"]), providers=list(providers))
    input_name = sess.get_inputs()[0].name
    output_name = sess.get_outputs()[0].name
    probs = []
    y_true = []
    for xb, yb in loader:
        x = xb.numpy().astype(np.float32)
        logits = sess.run([output_name], {input_name: x})[0]
        probs.append(sigmoid(logits).astype(np.float32))
        y_true.append(yb.numpy().astype(np.int32))
    return np.concatenate(probs, axis=0), np.concatenate(y_true, axis=0)


def apply_thresholds(prob: np.ndarray, classes: Sequence[str], thresholds: Dict[str, float]) -> np.ndarray:
    thr = np.array([thresholds.get(label, 0.5) for label in classes], dtype=np.float32)
    return (prob >= thr[None, :]).astype(np.int32)


def sample_jaccard(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    scores = []
    for yt, yp in zip(y_true, y_pred):
        union = np.logical_or(yt == 1, yp == 1).sum()
        inter = np.logical_and(yt == 1, yp == 1).sum()
        scores.append(1.0 if union == 0 else inter / union)
    return np.array(scores, dtype=np.float32)


def safe_auc(y: np.ndarray, p: np.ndarray) -> float | None:
    if len(np.unique(y)) < 2:
        return None
    try:
        return float(roc_auc_score(y, p))
    except ValueError:
        return None


def class_metrics(y_true: np.ndarray, y_pred: np.ndarray, prob: np.ndarray, classes: Sequence[str], thresholds: Dict[str, float]) -> List[Dict[str, object]]:
    rows = []
    for i, label in enumerate(classes):
        yt = y_true[:, i]
        yp = y_pred[:, i]
        tp = int(((yt == 1) & (yp == 1)).sum())
        fp = int(((yt == 0) & (yp == 1)).sum())
        fn = int(((yt == 1) & (yp == 0)).sum())
        tn = int(((yt == 0) & (yp == 0)).sum())
        pos_prob = prob[yt == 1, i]
        neg_prob = prob[yt == 0, i]
        rows.append(
            {
                "class": label,
                "threshold": thresholds.get(label, 0.5),
                "support_positive": int(yt.sum()),
                "support_negative": int((yt == 0).sum()),
                "predicted_positive": int(yp.sum()),
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "tn": tn,
                "accuracy": float((yt == yp).mean()),
                "precision": float(precision_score(yt, yp, zero_division=0)),
                "recall": float(recall_score(yt, yp, zero_division=0)),
                "specificity": None if (tn + fp) == 0 else float(tn / (tn + fp)),
                "f1": float(f1_score(yt, yp, zero_division=0)),
                "auc": safe_auc(yt, prob[:, i]),
                "mean_prob_positive_cases": float(pos_prob.mean()) if len(pos_prob) else None,
                "mean_prob_negative_cases": float(neg_prob.mean()) if len(neg_prob) else None,
            }
        )
    return rows


def overall_metrics(y_true: np.ndarray, y_pred: np.ndarray, prob: np.ndarray, class_rows: Sequence[Dict[str, object]]) -> Dict[str, object]:
    aucs = [row["auc"] for row in class_rows if row["auc"] is not None]
    supported = [row for row in class_rows if int(row["support_positive"]) > 0]
    supported_aucs = [row["auc"] for row in supported if row["auc"] is not None]
    jaccard = sample_jaccard(y_true, y_pred)
    return {
        "n_records": int(y_true.shape[0]),
        "n_classes": int(y_true.shape[1]),
        "exact_match_accuracy": float(np.all(y_true == y_pred, axis=1).mean()),
        "label_accuracy": float((y_true == y_pred).mean()),
        "sample_jaccard_accuracy": float(jaccard.mean()),
        "micro_precision": float(precision_score(y_true.ravel(), y_pred.ravel(), zero_division=0)),
        "micro_recall": float(recall_score(y_true.ravel(), y_pred.ravel(), zero_division=0)),
        "micro_f1": float(f1_score(y_true.ravel(), y_pred.ravel(), zero_division=0)),
        "macro_f1_all_classes": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_f1_supported_classes": float(f1_score(y_true[:, [i for i, row in enumerate(class_rows) if int(row["support_positive"]) > 0]], y_pred[:, [i for i, row in enumerate(class_rows) if int(row["support_positive"]) > 0]], average="macro", zero_division=0)),
        "macro_auc_auc_defined_classes": float(np.mean(aucs)) if aucs else None,
        "macro_auc_supported_classes": float(np.mean(supported_aucs)) if supported_aucs else None,
        "mean_predicted_labels_per_record": float(y_pred.sum(axis=1).mean()),
        "mean_true_labels_per_record": float(y_true.sum(axis=1).mean()),
        "records_with_no_predicted_label": int((y_pred.sum(axis=1) == 0).sum()),
        "records_with_more_than_one_predicted_label": int((y_pred.sum(axis=1) > 1).sum()),
    }


def prediction_label_counts(y_pred: np.ndarray, classes: Sequence[str]) -> Dict[str, int]:
    return {label: int(y_pred[:, i].sum()) for i, label in enumerate(classes)}


def truth_label_counts(y_true: np.ndarray, classes: Sequence[str]) -> Dict[str, int]:
    return {label: int(y_true[:, i].sum()) for i, label in enumerate(classes)}


def threshold_scan(prob: np.ndarray, y_true: np.ndarray, classes: Sequence[str], existing_thresholds: Dict[str, float]) -> List[Dict[str, object]]:
    rows = []
    grid = np.linspace(0.05, 0.95, 19)
    for i, label in enumerate(classes):
        yt = y_true[:, i]
        if len(np.unique(yt)) < 2:
            rows.append(
                {
                    "class": label,
                    "support_positive": int(yt.sum()),
                    "existing_threshold": existing_thresholds.get(label, 0.5),
                    "existing_f1": None,
                    "best_threshold_by_f1": None,
                    "best_f1": None,
                    "best_precision": None,
                    "best_recall": None,
                    "note": "undefined_no_positive_or_no_negative_support",
                }
            )
            continue
        existing_pred = (prob[:, i] >= existing_thresholds.get(label, 0.5)).astype(np.int32)
        existing_f1 = float(f1_score(yt, existing_pred, zero_division=0))
        best = None
        for threshold in grid:
            pred = (prob[:, i] >= threshold).astype(np.int32)
            item = {
                "threshold": float(threshold),
                "f1": float(f1_score(yt, pred, zero_division=0)),
                "precision": float(precision_score(yt, pred, zero_division=0)),
                "recall": float(recall_score(yt, pred, zero_division=0)),
            }
            if best is None or item["f1"] > best["f1"]:
                best = item
        rows.append(
            {
                "class": label,
                "support_positive": int(yt.sum()),
                "existing_threshold": existing_thresholds.get(label, 0.5),
                "existing_f1": existing_f1,
                "best_threshold_by_f1": best["threshold"],
                "best_f1": best["f1"],
                "best_precision": best["precision"],
                "best_recall": best["recall"],
                "note": "diagnostic_scan_only_not_retuned",
            }
        )
    return rows


def prediction_pattern_rows(y_true: np.ndarray, y_pred: np.ndarray, classes: Sequence[str], limit: int = 30) -> List[Dict[str, object]]:
    counts: Dict[tuple[str, str], int] = {}
    exact_counts: Dict[tuple[str, str], int] = {}
    for yt, yp in zip(y_true, y_pred):
        key = (label_text(yt, classes), label_text(yp, classes))
        counts[key] = counts.get(key, 0) + 1
        if np.all(yt == yp):
            exact_counts[key] = exact_counts.get(key, 0) + 1
    rows = []
    for (true_labels, predicted_labels), count in sorted(counts.items(), key=lambda item: item[1], reverse=True)[:limit]:
        rows.append(
            {
                "true_labels": true_labels,
                "predicted_labels": predicted_labels,
                "count": count,
                "exact_count": exact_counts.get((true_labels, predicted_labels), 0),
            }
        )
    return rows


def write_overall_csv(path: Path, metrics: Dict[str, object]) -> None:
    rows = [{"metric": key, "value": fmt4(value) if isinstance(value, float) else value} for key, value in metrics.items()]
    write_csv(path, rows, ["metric", "value"])


def write_per_class_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    fields = [
        "class",
        "threshold",
        "support_positive",
        "support_negative",
        "predicted_positive",
        "tp",
        "fp",
        "fn",
        "tn",
        "accuracy",
        "precision",
        "recall",
        "specificity",
        "f1",
        "auc",
        "mean_prob_positive_cases",
        "mean_prob_negative_cases",
    ]
    formatted = []
    for row in rows:
        formatted.append({field: fmt4(row[field]) if isinstance(row.get(field), float) else row.get(field, "") for field in fields})
    write_csv(path, formatted, fields)


def write_threshold_scan_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    fields = [
        "class",
        "support_positive",
        "existing_threshold",
        "existing_f1",
        "best_threshold_by_f1",
        "best_f1",
        "best_precision",
        "best_recall",
        "note",
    ]
    formatted = []
    for row in rows:
        formatted.append({field: fmt4(row[field]) if isinstance(row.get(field), float) else row.get(field, "") for field in fields})
    write_csv(path, formatted, fields)


def write_per_record_csv(path: Path, ds: CPSCConvertedDataset, classes: Sequence[str], y_true: np.ndarray, y_pred: np.ndarray, prob: np.ndarray) -> None:
    fields = [
        "record_id",
        "group",
        "true_labels",
        "predicted_labels",
        "exact_match",
        "sample_jaccard",
        "n_true_labels",
        "n_predicted_labels",
    ]
    for label in classes:
        fields.extend([f"true_{label}", f"pred_{label}", f"prob_{label}"])

    rows = []
    jaccard = sample_jaccard(y_true, y_pred)
    for i, source_row in enumerate(ds.rows):
        row: Dict[str, object] = {
            "record_id": source_row["record_id"],
            "group": source_row.get("group", ""),
            "true_labels": label_text(y_true[i], classes),
            "predicted_labels": label_text(y_pred[i], classes),
            "exact_match": int(np.all(y_true[i] == y_pred[i])),
            "sample_jaccard": fmt4(jaccard[i]),
            "n_true_labels": int(y_true[i].sum()),
            "n_predicted_labels": int(y_pred[i].sum()),
        }
        for j, label in enumerate(classes):
            row[f"true_{label}"] = int(y_true[i, j])
            row[f"pred_{label}"] = int(y_pred[i, j])
            row[f"prob_{label}"] = fmt4(prob[i, j])
        rows.append(row)
    write_csv(path, rows, fields)


def build_report(
    path: Path,
    summary: Dict[str, object],
    overall: Dict[str, object],
    class_rows: Sequence[Dict[str, object]],
    threshold_rows: Sequence[Dict[str, object]],
) -> None:
    supported_rows = [row for row in class_rows if int(row["support_positive"]) > 0]
    lines = [
        "# CPSC2018 External Sanity Evaluation",
        "",
        "Date: 2026-05-29",
        "",
        "This is a sanity evaluation of the existing `models_fine_chapman_ft` model on the conservative CPSC2018 current-10-class converted subset. It is not a final validation result and does not change model weights.",
        "",
        "## Inputs",
        "",
        f"- Model directory: `{summary['model_dir']}`",
        f"- Dataset CSV: `{summary['dataset_csv']}`",
        f"- Records: `{overall['n_records']}`",
        "- Normalization: per-lead z-score, matching the existing PTB-XL dataset loader.",
        "- Thresholds: existing `models_fine_chapman_ft/thresholds_5cls.json`; no retuning on CPSC.",
        "",
        "## Overall Metrics",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Exact match accuracy | {fmt4(overall['exact_match_accuracy'])} |",
        f"| Label accuracy | {fmt4(overall['label_accuracy'])} |",
        f"| Sample Jaccard accuracy | {fmt4(overall['sample_jaccard_accuracy'])} |",
        f"| Micro precision | {fmt4(overall['micro_precision'])} |",
        f"| Micro recall | {fmt4(overall['micro_recall'])} |",
        f"| Micro F1 | {fmt4(overall['micro_f1'])} |",
        f"| Macro F1, supported classes | {fmt4(overall['macro_f1_supported_classes'])} |",
        f"| Macro AUROC, supported classes | {fmt4(overall['macro_auc_supported_classes'])} |",
        f"| Mean predicted labels per record | {fmt4(overall['mean_predicted_labels_per_record'])} |",
        f"| Records with no predicted label | {overall['records_with_no_predicted_label']} |",
        f"| Records with >1 predicted labels | {overall['records_with_more_than_one_predicted_label']} |",
        "",
        "## Per-Class Metrics",
        "",
        "| Class | Truth + | Pred + | Precision | Recall | F1 | AUROC |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in supported_rows:
        lines.append(
            f"| {row['class']} | {row['support_positive']} | {row['predicted_positive']} | "
            f"{fmt4(row['precision'])} | {fmt4(row['recall'])} | {fmt4(row['f1'])} | {fmt4(row['auc'])} |"
        )

    lines.extend(
        [
            "",
            "## Threshold Scan",
            "",
            "This grid scan is diagnostic only. It shows whether fixed PTB-XL/Chapman thresholds transfer to CPSC; it does not retune the model.",
            "",
            "| Class | Existing threshold | Existing F1 | Best scan threshold | Best scan F1 | Best scan precision | Best scan recall |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in threshold_rows:
        if row.get("best_threshold_by_f1") is None:
            continue
        lines.append(
            f"| {row['class']} | {fmt4(row['existing_threshold'])} | {fmt4(row['existing_f1'])} | "
            f"{fmt4(row['best_threshold_by_f1'])} | {fmt4(row['best_f1'])} | "
            f"{fmt4(row['best_precision'])} | {fmt4(row['best_recall'])} |"
        )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- This run checks compatibility and coarse external behavior only. CPSC labels, class mix, and thresholds differ from the PTB-XL/Chapman training context.",
            "- AUROC is the better threshold-independent signal for whether the model ranks positive cases above negatives.",
            "- Fixed-threshold precision/recall shows whether the existing thresholds transfer to CPSC without calibration.",
            "- Classes with zero positive CPSC support in this converted subset are not interpretable as CPSC performance endpoints.",
            "",
            "## Outputs",
            "",
            f"- `{summary['overall_metrics_csv']}`",
            f"- `{summary['per_class_metrics_csv']}`",
            f"- `{summary['threshold_scan_csv']}`",
            f"- `{summary['prediction_patterns_csv']}`",
            f"- `{summary['per_record_predictions_csv']}`",
            f"- `{summary['summary_json']}`",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def evaluate(args: argparse.Namespace) -> Dict[str, object]:
    model_info = load_model_info(Path(args.model_dir))
    classes = list(model_info["classes"])
    ds = CPSCConvertedDataset(Path(args.dataset_csv), classes, normalize=args.normalize)
    loader = DataLoader(ds, batch_size=args.batch, shuffle=False, num_workers=args.num_workers)
    providers = ["CPUExecutionProvider"] if args.cpu else ["CUDAExecutionProvider", "CPUExecutionProvider"]

    prob, y_true = run_onnx(model_info, loader, providers)
    y_pred = apply_thresholds(prob, classes, model_info["thresholds"])
    class_rows = class_metrics(y_true, y_pred, prob, classes, model_info["thresholds"])
    overall = overall_metrics(y_true, y_pred, prob, class_rows)
    threshold_rows = threshold_scan(prob, y_true, classes, model_info["thresholds"])
    pattern_rows = prediction_pattern_rows(y_true, y_pred, classes)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    overall_csv = out_dir / "cpsc2018_external_overall_metrics.csv"
    per_class_csv = out_dir / "cpsc2018_external_per_class_metrics.csv"
    threshold_scan_csv = out_dir / "cpsc2018_external_threshold_scan.csv"
    prediction_patterns_csv = out_dir / "cpsc2018_external_prediction_patterns.csv"
    per_record_csv = out_dir / "cpsc2018_external_per_record_predictions.csv"
    summary_json = out_dir / "cpsc2018_external_eval_summary.json"
    report_md = out_dir / "cpsc2018_external_sanity_report.md"

    write_overall_csv(overall_csv, overall)
    write_per_class_csv(per_class_csv, class_rows)
    write_threshold_scan_csv(threshold_scan_csv, threshold_rows)
    write_csv(prediction_patterns_csv, pattern_rows, ["true_labels", "predicted_labels", "count", "exact_count"])
    write_per_record_csv(per_record_csv, ds, classes, y_true, y_pred, prob)

    summary: Dict[str, object] = {
        "build_version": BUILD_VERSION,
        "model_dir": str(Path(args.model_dir)),
        "onnx_path": str(model_info["onnx_path"]),
        "dataset_csv": str(Path(args.dataset_csv)),
        "out_dir": str(out_dir),
        "overall_metrics_csv": str(overall_csv),
        "per_class_metrics_csv": str(per_class_csv),
        "threshold_scan_csv": str(threshold_scan_csv),
        "prediction_patterns_csv": str(prediction_patterns_csv),
        "per_record_predictions_csv": str(per_record_csv),
        "summary_json": str(summary_json),
        "report_md": str(report_md),
        "classes": classes,
        "thresholds": model_info["thresholds"],
        "truth_label_counts": truth_label_counts(y_true, classes),
        "prediction_label_counts": prediction_label_counts(y_pred, classes),
        "overall_metrics": overall,
        "per_class_metrics": class_rows,
        "threshold_scan": threshold_rows,
        "prediction_patterns_top30": pattern_rows,
        "notes": [
            "Sanity evaluation only; model weights were not changed.",
            "Dataset is the conservative CPSC2018 current-10-class converted subset.",
            "Inputs were normalized per lead with z-score to match PTB-XL loader behavior.",
            "Thresholds were loaded from the existing model directory without retuning.",
        ],
    }
    summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    build_report(report_md, summary, overall, class_rows, threshold_rows)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def parse_args() -> argparse.Namespace:
    root = project_root()
    parser = argparse.ArgumentParser(description="Evaluate existing model on converted CPSC2018 current-10-class subset.")
    parser.add_argument("--model-dir", default=str(root / "models_fine_chapman_ft"))
    parser.add_argument("--dataset-csv", default=str(root / "data/challenge_2020/cpsc_2018_converted_current10/ground_truth.csv"))
    parser.add_argument("--out-dir", default=str(root / "outputs/cpsc2018_external_eval"))
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--normalize", default="z", choices=["z", "none"])
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


def main() -> None:
    evaluate(parse_args())


if __name__ == "__main__":
    main()
