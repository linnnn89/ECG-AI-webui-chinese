from __future__ import annotations

import glob
import json
import os
import sqlite3
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

from .schema import CaseRecord, SearchResult


CASEBANK_REQUIRED_FILES = [
    "case_index.sqlite",
    "case_probs.npy",
    "case_margins.npy",
    "case_wave_features.npy",
    "case_retrieval_vectors.npy",
    "vector_stats.json",
    "build_config.json",
]


CREATE_CASES_SQL = """
CREATE TABLE IF NOT EXISTS cases (
    row_id INTEGER PRIMARY KEY,
    case_id TEXT UNIQUE NOT NULL,
    source TEXT NOT NULL,
    source_record_id TEXT,
    record_path TEXT,
    header_path TEXT,
    image_path TEXT,
    patient_id_hash TEXT,
    split TEXT,
    labels_json TEXT,
    predicted_labels_json TEXT,
    probabilities_json TEXT,
    margins_json TEXT,
    signal_quality_json TEXT,
    has_embedding INTEGER DEFAULT 0,
    build_version TEXT,
    created_at TEXT
);
"""

CREATE_INDEXES_SQL = [
    "CREATE INDEX IF NOT EXISTS idx_cases_case_id ON cases(case_id);",
    "CREATE INDEX IF NOT EXISTS idx_cases_source ON cases(source);",
    "CREATE INDEX IF NOT EXISTS idx_cases_split ON cases(split);",
    "CREATE INDEX IF NOT EXISTS idx_cases_patient ON cases(patient_id_hash);",
]


def read_json(path: str | os.PathLike[str]) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: str | os.PathLike[str], payload: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def _json_dumps(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def _json_loads(value: str | None, default: Any) -> Any:
    if value is None or value == "":
        return default
    return json.loads(value)


def validate_casebank_dir(casebank_dir: str | os.PathLike[str]) -> None:
    base = Path(casebank_dir)
    missing = [name for name in CASEBANK_REQUIRED_FILES if not (base / name).exists()]
    if missing:
        raise FileNotFoundError(
            "CaseBank is incomplete or missing. Build it first with "
            f"python -m code.tools.build_casebank. Missing files: {missing}"
        )


def init_case_index(db_path: str | os.PathLike[str]) -> None:
    with sqlite3.connect(db_path) as conn:
        conn.execute(CREATE_CASES_SQL)
        for sql in CREATE_INDEXES_SQL:
            conn.execute(sql)
        conn.commit()


def write_case_rows(db_path: str | os.PathLike[str], rows: Sequence[Dict[str, Any]]) -> None:
    init_case_index(db_path)
    sql = """
    INSERT INTO cases (
        row_id, case_id, source, source_record_id, record_path, header_path,
        image_path, patient_id_hash, split, labels_json, predicted_labels_json,
        probabilities_json, margins_json, signal_quality_json, has_embedding,
        build_version, created_at
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """
    values = []
    for row in rows:
        values.append(
            (
                int(row["row_id"]),
                row["case_id"],
                row["source"],
                row.get("source_record_id"),
                row.get("record_path"),
                row.get("header_path"),
                row.get("image_path"),
                row.get("patient_id_hash"),
                row.get("split"),
                _json_dumps(row.get("labels", [])),
                _json_dumps(row.get("predicted_labels", [])),
                _json_dumps(row.get("probabilities", {})),
                _json_dumps(row.get("margins", {})),
                _json_dumps(row.get("signal_quality", {})),
                1 if row.get("has_embedding") else 0,
                row.get("build_version"),
                row.get("created_at"),
            )
        )
    with sqlite3.connect(db_path) as conn:
        conn.execute("DELETE FROM cases")
        conn.executemany(sql, values)
        conn.commit()


def load_case_rows(db_path: str | os.PathLike[str]) -> List[CaseRecord]:
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute("SELECT * FROM cases ORDER BY row_id").fetchall()
    return [
        CaseRecord(
            row_id=int(row["row_id"]),
            case_id=row["case_id"],
            source=row["source"],
            source_record_id=row["source_record_id"],
            record_path=row["record_path"],
            header_path=row["header_path"],
            image_path=row["image_path"],
            patient_id_hash=row["patient_id_hash"],
            split=row["split"],
            labels=list(_json_loads(row["labels_json"], [])),
            predicted_labels=list(_json_loads(row["predicted_labels_json"], [])),
            probabilities=dict(_json_loads(row["probabilities_json"], {})),
            margins=dict(_json_loads(row["margins_json"], {})),
            signal_quality=dict(_json_loads(row["signal_quality_json"], {})),
            has_embedding=bool(row["has_embedding"]),
        )
        for row in rows
    ]


def _load_classes_file(path: str) -> List[str]:
    classes = read_json(path)
    if not isinstance(classes, list) or not all(isinstance(c, str) for c in classes):
        raise ValueError(f"Invalid classes file: {path}")
    return classes


def _load_thresholds_file(path: str) -> Dict[str, float]:
    thresholds = read_json(path)
    if not isinstance(thresholds, dict):
        raise ValueError(f"Invalid thresholds file: {path}")
    return {str(k): float(v) for k, v in thresholds.items()}


def resolve_classes_thresholds(
    model_dir: str | os.PathLike[str],
    label_mode: int = 10,
) -> Tuple[List[str], Dict[str, float], str, str]:
    model_dir = str(model_dir)
    class_files = sorted(glob.glob(os.path.join(model_dir, "classes_*.json")))
    threshold_files = sorted(glob.glob(os.path.join(model_dir, "thresholds_*.json")))
    if not class_files:
        raise FileNotFoundError(f"Cannot determine current {label_mode}-class labels: no classes_*.json in {model_dir}")
    if not threshold_files:
        raise FileNotFoundError(f"Thresholds are missing: no thresholds_*.json in {model_dir}")

    valid_pairs: List[Tuple[List[str], Dict[str, float], str, str]] = []
    for class_file in class_files:
        classes = _load_classes_file(class_file)
        if len(classes) != label_mode:
            continue
        suffix = os.path.basename(class_file).replace("classes_", "", 1)
        exact_threshold = os.path.join(model_dir, f"thresholds_{suffix}")
        candidate_thresholds = [exact_threshold] if os.path.exists(exact_threshold) else threshold_files
        for threshold_file in candidate_thresholds:
            thresholds = _load_thresholds_file(threshold_file)
            class_set = set(classes)
            threshold_set = set(thresholds)
            if class_set == threshold_set:
                valid_pairs.append((classes, thresholds, class_file, threshold_file))

    if not valid_pairs:
        raise ValueError(
            f"No matching {label_mode}-class classes/thresholds pair in {model_dir}. "
            "Check that classes and thresholds contain the same labels."
        )
    if len(valid_pairs) > 1:
        pairs = [(os.path.basename(c), os.path.basename(t)) for _, _, c, t in valid_pairs]
        raise ValueError(f"Multiple valid {label_mode}-class classes/thresholds pairs found: {pairs}")
    return valid_pairs[0]


def resolve_onnx_path(model_dir: str | os.PathLike[str]) -> str:
    preferred = os.path.join(str(model_dir), "inception_5cls.onnx")
    if os.path.exists(preferred):
        return preferred
    onnx_files = sorted(glob.glob(os.path.join(str(model_dir), "*.onnx")))
    if not onnx_files:
        raise FileNotFoundError(f"No ONNX model found in {model_dir}")
    if len(onnx_files) > 1:
        raise ValueError(f"Multiple ONNX files found in {model_dir}; specify the active model directory explicitly.")
    return onnx_files[0]


def search_result_to_dict(result: SearchResult) -> Dict[str, Any]:
    return {
        "query": result.query,
        "similar_cases": [
            {
                "rank": item.rank,
                "case_id": item.case_id,
                "score": item.score,
                "score_level": item.score_level,
                "components": item.components,
                "source": item.source,
                "record_path": item.record_path,
                "image_path": item.image_path,
                "image_url": item.image_url,
                "true_diagnosis": item.true_diagnosis,
                "label_scope": item.label_scope,
                "labels": item.labels,
                "predicted_labels": item.predicted_labels,
                "probabilities": item.probabilities,
                "threshold_margins": item.margins,
            }
            for item in result.similar_cases
        ],
        "warnings": result.warnings,
    }
