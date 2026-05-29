from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np

from code.casebank.features import extract_basic_wave_features
from code.casebank.inference import (
    load_npy_tensor,
    load_runtime,
    load_wfdb_tensor,
    project_root,
)
from code.casebank.io import search_result_to_dict, write_json
from code.casebank.search import CaseBankStore, SearchEngine


def _case_id_from_record(record_base: str) -> str:
    return f"ptbxl_{Path(record_base).stem}"


def search_casebank(args: argparse.Namespace) -> dict:
    runtime = load_runtime(model_dir=args.model_dir, label_mode=10)
    store = CaseBankStore.load(os.path.abspath(args.casebank))
    if list(store.classes) != list(runtime.classes):
        raise ValueError(
            "CaseBank classes do not match model classes.\n"
            f"casebank: {store.classes}\nmodel:    {runtime.classes}"
        )
    if str(args.record).lower().endswith(".npy"):
        x, record_base = load_npy_tensor(args.record)
        query_case_id = f"chapman_{Path(record_base).stem}"
        query_record_path = record_base
    else:
        x, record_base = load_wfdb_tensor(args.record, ptbxl_root=args.ptbxl_root)
        query_case_id = _case_id_from_record(record_base)
        query_record_path = os.path.relpath(record_base, os.path.join(args.ptbxl_root, "wfdb"))
    probabilities = runtime.predict_one(x)
    margins = probabilities - runtime.threshold_array
    wave_features = extract_basic_wave_features(x)
    retrieval_vector = store.make_query_vector(probabilities, margins, wave_features)
    engine = SearchEngine(store)
    result = engine.search(
        probabilities=probabilities,
        margins=margins,
        wave_features=wave_features,
        retrieval_vector=retrieval_vector,
        predicted_labels=[],
        query_case_id=query_case_id,
        query_record_path=query_record_path,
        top_k=args.top_k,
        prefetch_k=args.prefetch_k,
        min_candidates=args.min_candidates,
        score_threshold=args.score_threshold,
        ablation="retrieval_only",
    )
    payload = search_result_to_dict(result)
    payload["query"]["record_path"] = args.record
    payload["query"]["resolved_record_path"] = record_base
    if args.out_json:
        os.makedirs(os.path.dirname(os.path.abspath(args.out_json)), exist_ok=True)
        write_json(args.out_json, payload)
    return payload


def parse_args() -> argparse.Namespace:
    root = project_root()
    parser = argparse.ArgumentParser(description="Search similar ECG cases from CaseBank.")
    parser.add_argument("--record", required=True)
    parser.add_argument("--casebank", default=os.path.join(root, "data", "casebank_vector_index"))
    parser.add_argument("--model_dir", default=None)
    parser.add_argument("--ptbxl_root", default=os.path.join(root, "data", "ptbxl"))
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--prefetch_k", type=int, default=200)
    parser.add_argument("--min_candidates", type=int, default=50)
    parser.add_argument("--score_threshold", type=float, default=0.0)
    parser.add_argument("--out_json", default=None)
    return parser.parse_args()


def main() -> None:
    payload = search_casebank(parse_args())
    print(f"similar_cases={len(payload['similar_cases'])}")
    if payload["warnings"]:
        print("warnings:")
        for warning in payload["warnings"]:
            print(f"- {warning}")


if __name__ == "__main__":
    main()
