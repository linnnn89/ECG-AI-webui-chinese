from __future__ import annotations

import argparse
import csv
import json
import os
from collections import defaultdict
from typing import Dict, Iterable, List, Sequence

import numpy as np

from code.casebank.inference import project_root
from code.casebank.io import search_result_to_dict, write_json
from code.casebank.search import CaseBankStore, SearchEngine
from code.casebank.similarity import jaccard_labels


def _dcg(relevances: Sequence[float]) -> float:
    total = 0.0
    for idx, rel in enumerate(relevances, start=1):
        total += float(rel) / np.log2(idx + 1)
    return float(total)


def _ndcg_at_k(relevances: Sequence[float], ideal_relevances: Sequence[float], k: int) -> float:
    dcg = _dcg(list(relevances)[:k])
    idcg = _dcg(sorted(ideal_relevances, reverse=True)[:k])
    if idcg <= 0:
        return 0.0
    return float(dcg / idcg)


def _hit_at(labels: Iterable[str], result_labels: List[Iterable[str]], k: int) -> int:
    query = set(labels)
    if not query:
        return 0
    return int(any(query & set(lbls) for lbls in result_labels[:k]))


def _query_indices(store: CaseBankStore, split: str) -> List[int]:
    if split == "all":
        return list(range(len(store.cases)))
    return [idx for idx, case in enumerate(store.cases) if case.split == split]


def _ideal_relevances(store: CaseBankStore, row_id: int, exclude_same_patient: bool) -> List[float]:
    case = store.cases[row_id]
    labels = set(case.labels)
    out = []
    for idx, other in enumerate(store.cases):
        if idx == row_id:
            continue
        if exclude_same_patient and case.patient_id_hash and case.patient_id_hash == other.patient_id_hash:
            continue
        out.append(jaccard_labels(labels, other.labels))
    return out


def _summarize(rows: List[Dict[str, object]], classes: List[str]) -> Dict[str, object]:
    n = len(rows)
    if n == 0:
        return {"num_queries": 0}
    total_results = sum(int(r["num_results"]) for r in rows)
    weak_results = sum(int(r["weak_results"]) for r in rows)
    summary = {
        "num_queries": n,
        "top1_hit": float(np.mean([float(r["top1_hit"]) for r in rows])),
        "top3_hit": float(np.mean([float(r["top3_hit"]) for r in rows])),
        "top5_hit": float(np.mean([float(r["top5_hit"]) for r in rows])),
        "top10_hit": float(np.mean([float(r["top10_hit"]) for r in rows])),
        "mean_jaccard_top1": float(np.mean([float(r["mean_jaccard_top1"]) for r in rows])),
        "mean_jaccard_top3": float(np.mean([float(r["mean_jaccard_top3"]) for r in rows])),
        "mean_jaccard_top10": float(np.mean([float(r["mean_jaccard_top10"]) for r in rows])),
        "ndcg_at_10": float(np.mean([float(r["ndcg_at_10"]) for r in rows])),
        "mean_score_top10": float(np.mean([float(r["mean_score_top10"]) for r in rows])),
        "low_confidence_rate": float(weak_results / total_results) if total_results else 0.0,
        "no_result_rate": float(np.mean([1.0 if int(r["num_results"]) == 0 else 0.0 for r in rows])),
    }
    per_class_hit = {}
    for cls in classes:
        cls_rows = [r for r in rows if cls in str(r["query_labels"]).split("|")]
        per_class_hit[cls] = (
            float(np.mean([float(r["top10_hit"]) for r in cls_rows])) if cls_rows else None
        )
    summary["per_class_top10_hit"] = per_class_hit
    return summary


def evaluate_casebank(args: argparse.Namespace) -> Dict[str, object]:
    store = CaseBankStore.load(os.path.abspath(args.casebank))
    engine = SearchEngine(store)
    modes = ["margin_only", "retrieval_only", "full"] if args.ablation == "all" else [args.ablation]
    query_indices = _query_indices(store, args.split)
    if args.limit > 0:
        query_indices = query_indices[: args.limit]
    os.makedirs(os.path.dirname(os.path.abspath(args.out_csv)), exist_ok=True)
    all_rows: List[Dict[str, object]] = []
    summary: Dict[str, object] = {"casebank": os.path.abspath(args.casebank), "split": args.split, "modes": {}}

    for mode in modes:
        mode_rows: List[Dict[str, object]] = []
        for row_id in query_indices:
            case = store.cases[row_id]
            result = engine.search_from_row(
                row_id,
                top_k=args.top_k,
                prefetch_k=args.prefetch_k,
                min_candidates=args.min_candidates,
                score_threshold=args.score_threshold,
                exclude_same_patient=bool(args.exclude_same_patient),
                ablation=mode,
            )
            payload = search_result_to_dict(result)
            result_label_sets = [item["labels"] for item in payload["similar_cases"]]
            result_scores = [float(item["score"]) for item in payload["similar_cases"]]
            query_labels = set(case.labels)
            jaccards = [jaccard_labels(query_labels, labels) for labels in result_label_sets]
            row = {
                "mode": mode,
                "row_id": row_id,
                "case_id": case.case_id,
                "split": case.split,
                "query_labels": "|".join(case.labels),
                "num_results": len(payload["similar_cases"]),
                "weak_results": sum(1 for item in payload["similar_cases"] if item["score_level"] == "weak"),
                "top1_hit": _hit_at(query_labels, result_label_sets, 1),
                "top3_hit": _hit_at(query_labels, result_label_sets, 3),
                "top5_hit": _hit_at(query_labels, result_label_sets, 5),
                "top10_hit": _hit_at(query_labels, result_label_sets, 10),
                "mean_jaccard_top1": float(np.mean(jaccards[:1])) if jaccards[:1] else 0.0,
                "mean_jaccard_top3": float(np.mean(jaccards[:3])) if jaccards[:3] else 0.0,
                "mean_jaccard_top10": float(np.mean(jaccards[:10])) if jaccards[:10] else 0.0,
                "ndcg_at_10": _ndcg_at_k(
                    jaccards,
                    _ideal_relevances(store, row_id, bool(args.exclude_same_patient)),
                    10,
                ),
                "mean_score_top10": float(np.mean(result_scores[:10])) if result_scores[:10] else 0.0,
                "warnings": " | ".join(payload["warnings"]),
            }
            mode_rows.append(row)
            all_rows.append(row)
        summary["modes"][mode] = _summarize(mode_rows, store.classes)

    if all_rows:
        with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
            writer.writeheader()
            writer.writerows(all_rows)
    else:
        with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
            f.write("")
    write_json(args.out_json, summary)
    return summary


def parse_args() -> argparse.Namespace:
    root = project_root()
    parser = argparse.ArgumentParser(description="Evaluate CaseBank retrieval quality.")
    parser.add_argument("--casebank", default=os.path.join(root, "data", "casebank_vector_index"))
    parser.add_argument("--split", choices=["train", "val", "test", "all"], default="test")
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--prefetch_k", type=int, default=200)
    parser.add_argument("--min_candidates", type=int, default=50)
    parser.add_argument("--score_threshold", type=float, default=0.55)
    parser.add_argument("--exclude_same_patient", type=int, choices=[0, 1], default=1)
    parser.add_argument("--ablation", choices=["all", "margin_only", "retrieval_only", "full"], default="all")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--out_csv", default=os.path.join(root, "runs", "casebank_eval.csv"))
    parser.add_argument("--out_json", default=os.path.join(root, "runs", "casebank_eval_summary.json"))
    return parser.parse_args()


def main() -> None:
    summary = evaluate_casebank(parse_args())
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
