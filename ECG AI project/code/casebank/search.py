from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from .features import build_single_retrieval_vector, zscore
from .io import load_case_rows, read_json, validate_casebank_dir
from .schema import CaseRecord, SearchResult, SimilarCase
from .similarity import (
    batch_cosine_similarity_01,
    score_level,
)


@dataclass
class CaseBankStore:
    casebank_dir: str
    classes: List[str]
    cases: List[CaseRecord]
    probabilities: np.ndarray
    margins: np.ndarray
    wave_features: np.ndarray
    retrieval_vectors: np.ndarray
    vector_stats: Dict[str, Any]
    build_config: Dict[str, Any]
    embeddings: Optional[np.ndarray] = None

    @classmethod
    def load(cls, casebank_dir: str) -> "CaseBankStore":
        validate_casebank_dir(casebank_dir)
        base = casebank_dir
        build_config = read_json(os.path.join(base, "build_config.json"))
        vector_stats = read_json(os.path.join(base, "vector_stats.json"))
        cases = load_case_rows(os.path.join(base, "case_index.sqlite"))
        probabilities = np.load(os.path.join(base, "case_probs.npy")).astype(np.float32, copy=False)
        margins = np.load(os.path.join(base, "case_margins.npy")).astype(np.float32, copy=False)
        wave_features = np.load(os.path.join(base, "case_wave_features.npy")).astype(np.float32, copy=False)
        retrieval_vectors = np.load(os.path.join(base, "case_retrieval_vectors.npy")).astype(np.float32, copy=False)
        embeddings_path = os.path.join(base, "case_embeddings.npy")
        embeddings = np.load(embeddings_path).astype(np.float32, copy=False) if os.path.exists(embeddings_path) else None
        classes = list(build_config.get("classes", []))
        if not classes and cases:
            classes = list(cases[0].probabilities.keys())
        n = len(cases)
        for name, arr in [
            ("case_probs.npy", probabilities),
            ("case_margins.npy", margins),
            ("case_wave_features.npy", wave_features),
            ("case_retrieval_vectors.npy", retrieval_vectors),
        ]:
            if arr.shape[0] != n:
                raise ValueError(f"{name} row count {arr.shape[0]} does not match sqlite cases {n}")
        for idx, case in enumerate(cases):
            if case.row_id != idx:
                raise ValueError(f"SQLite row_id {case.row_id} does not match npy row index {idx}")
        return cls(
            casebank_dir=casebank_dir,
            classes=classes,
            cases=cases,
            probabilities=probabilities,
            margins=margins,
            wave_features=wave_features,
            retrieval_vectors=retrieval_vectors,
            vector_stats=vector_stats,
            build_config=build_config,
            embeddings=embeddings,
        )

    def make_query_vector(self, probabilities: np.ndarray, margins: np.ndarray, wave_features: np.ndarray) -> np.ndarray:
        return build_single_retrieval_vector(probabilities, margins, wave_features, self.vector_stats)

    def standardized_wave_features(self, wave_features: np.ndarray) -> np.ndarray:
        return zscore(np.asarray(wave_features, dtype=np.float32), self.vector_stats["wave_features"])


def _bad_signal_quality(case: CaseRecord) -> bool:
    q = case.signal_quality or {}
    return bool(q.get("bad") is True or str(q.get("quality", "")).lower() == "bad")


def _probability_dict(classes: Sequence[str], values: np.ndarray) -> Dict[str, float]:
    return {cls: float(v) for cls, v in zip(classes, np.asarray(values, dtype=np.float32))}


class SearchEngine:
    def __init__(self, store: CaseBankStore):
        self.store = store

    def search(
        self,
        *,
        probabilities: np.ndarray,
        margins: np.ndarray,
        wave_features: np.ndarray,
        retrieval_vector: Optional[np.ndarray] = None,
        predicted_labels: Optional[List[str]] = None,
        query_case_id: Optional[str] = None,
        query_record_path: Optional[str] = None,
        query_patient_id_hash: Optional[str] = None,
        top_k: int = 10,
        prefetch_k: int = 200,
        min_candidates: int = 50,
        score_threshold: float = 0.55,
        ablation: str = "retrieval_only",
    ) -> SearchResult:
        if ablation not in {"retrieval_only", "pure_vector"}:
            raise ValueError(f"Unsupported ablation mode: {ablation}")

        probabilities = np.asarray(probabilities, dtype=np.float32)
        margins = np.asarray(margins, dtype=np.float32)
        wave_features = np.asarray(wave_features, dtype=np.float32)
        retrieval_vector = (
            np.asarray(retrieval_vector, dtype=np.float32)
            if retrieval_vector is not None
            else self.store.make_query_vector(probabilities, margins, wave_features)
        )
        predicted_labels = list(predicted_labels or [])
        warnings: List[str] = []

        candidate_indices = []
        normalized_query_record = os.path.normcase(os.path.normpath(query_record_path)) if query_record_path else None
        for idx, case in enumerate(self.store.cases):
            if query_case_id and case.case_id == query_case_id:
                continue
            if normalized_query_record and case.record_path:
                if os.path.normcase(os.path.normpath(case.record_path)) == normalized_query_record:
                    continue
            if query_patient_id_hash and case.patient_id_hash == query_patient_id_hash:
                continue
            if _bad_signal_quality(case):
                continue
            candidate_indices.append(idx)

        if not candidate_indices:
            warnings.append("No candidate cases were available after self/same-patient/quality exclusion.")
            return SearchResult(query=self._query_payload(probabilities, margins, predicted_labels), similar_cases=[], warnings=warnings)

        candidate_indices_np = np.asarray(candidate_indices, dtype=np.int64)
        retrieval_sims = batch_cosine_similarity_01(
            retrieval_vector, self.store.retrieval_vectors[candidate_indices_np]
        )
        order = np.argsort(-retrieval_sims)[: max(1, min(prefetch_k, len(candidate_indices)))]
        scored = [
            (
                float(retrieval_sims[int(local_idx)]),
                int(candidate_indices_np[int(local_idx)]),
                {"retrieval_cosine": round(float(retrieval_sims[int(local_idx)]), 6)},
            )
            for local_idx in order
        ]
        visible = [item for item in scored if item[0] >= score_threshold][:top_k]
        if len(visible) < top_k:
            warnings.append(
                f"Only {len(visible)} cases passed score_threshold={score_threshold}; results were not padded to {top_k}."
            )
        if self.store.embeddings is None:
            warnings.append(
                "CaseBank is running without model embedding; results are based on probabilities, threshold margins and waveform features."
            )

        similar = []
        for rank, (score, case_idx, components) in enumerate(visible, start=1):
            case = self.store.cases[case_idx]
            signal_quality = case.signal_quality or {}
            similar.append(
                SimilarCase(
                    rank=rank,
                    case_id=case.case_id,
                    score=round(float(score), 6),
                    score_level=score_level(float(score)),
                    components=components,
                    record_path=case.record_path,
                    image_path=case.image_path,
                    source=case.source,
                    true_diagnosis=signal_quality.get("true_diagnosis") or signal_quality.get("diagnosis_text"),
                    label_scope=signal_quality.get("label_scope"),
                    image_url=f"/casebank_image/{case.case_id}",
                    labels=case.labels,
                    predicted_labels=case.predicted_labels,
                    probabilities=case.probabilities,
                    margins=case.margins,
                )
            )
        return SearchResult(
            query=self._query_payload(probabilities, margins, predicted_labels),
            similar_cases=similar,
            warnings=warnings,
        )

    def search_from_row(
        self,
        row_id: int,
        *,
        top_k: int = 10,
        prefetch_k: int = 200,
        min_candidates: int = 50,
        score_threshold: float = 0.55,
        exclude_same_patient: bool = True,
        ablation: str = "retrieval_only",
    ) -> SearchResult:
        case = self.store.cases[row_id]
        prob = self.store.probabilities[row_id]
        margins = self.store.margins[row_id]
        wave = self.store.wave_features[row_id]
        return self.search(
            probabilities=prob,
            margins=margins,
            wave_features=wave,
            retrieval_vector=self.store.retrieval_vectors[row_id],
            predicted_labels=case.predicted_labels,
            query_case_id=case.case_id,
            query_record_path=case.record_path,
            query_patient_id_hash=case.patient_id_hash if exclude_same_patient else None,
            top_k=top_k,
            prefetch_k=prefetch_k,
            min_candidates=min_candidates,
            score_threshold=score_threshold,
            ablation=ablation,
        )

    def _query_payload(
        self,
        probabilities: np.ndarray,
        margins: np.ndarray,
        predicted_labels: List[str],
    ) -> Dict[str, Any]:
        return {
            "predicted_labels": predicted_labels,
            "probabilities": _probability_dict(self.store.classes, probabilities),
            "threshold_margins": _probability_dict(self.store.classes, margins),
            "has_embedding": self.store.embeddings is not None,
        }
