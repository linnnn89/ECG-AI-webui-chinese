from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set

import numpy as np

from .features import build_single_retrieval_vector, zscore
from .io import load_case_rows, read_json, validate_casebank_dir
from .schema import CaseRecord, SearchResult, SimilarCase
from .similarity import (
    batch_cosine_similarity_01,
    cosine_similarity_01,
    jaccard_labels,
    score_level,
    wave_similarity_01,
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


def _case_label_set(case: CaseRecord) -> Set[str]:
    return set(case.labels) | set(case.predicted_labels)


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
        top_probability_labels: Optional[List[str]] = None,
        query_case_id: Optional[str] = None,
        query_record_path: Optional[str] = None,
        query_patient_id_hash: Optional[str] = None,
        top_k: int = 10,
        prefetch_k: int = 200,
        min_candidates: int = 50,
        score_threshold: float = 0.55,
        ablation: str = "full",
    ) -> SearchResult:
        if ablation not in {"full", "retrieval_only", "margin_only"}:
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
        top_probability_labels = list(top_probability_labels or [])
        warnings: List[str] = []
        if not predicted_labels:
            if top_probability_labels:
                warnings.append(
                    "No class passed threshold; top-2 probability labels were used for candidate filtering."
                )
            filter_labels = set(top_probability_labels)
        else:
            filter_labels = set(predicted_labels)

        base_indices = []
        label_filtered = []
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
            base_indices.append(idx)
            if not filter_labels or (_case_label_set(case) & filter_labels):
                label_filtered.append(idx)

        candidate_indices = label_filtered
        if len(candidate_indices) < min_candidates and len(base_indices) > len(candidate_indices):
            warnings.append(
                f"Candidate count {len(candidate_indices)} was below min_candidates={min_candidates}; label filter was relaxed."
            )
            candidate_indices = base_indices

        if not candidate_indices:
            warnings.append("No candidate cases were available after filtering.")
            return SearchResult(query=self._query_payload(probabilities, margins, predicted_labels), similar_cases=[], warnings=warnings)

        candidate_indices_np = np.asarray(candidate_indices, dtype=np.int64)
        retrieval_sims = batch_cosine_similarity_01(
            retrieval_vector, self.store.retrieval_vectors[candidate_indices_np]
        )
        prefetch_order = np.argsort(-retrieval_sims)[: max(1, min(prefetch_k, len(candidate_indices)))]
        prefetched = candidate_indices_np[prefetch_order]
        prefetched_retrieval_sims = retrieval_sims[prefetch_order]

        query_wave_z = self.store.standardized_wave_features(wave_features)
        query_label_for_overlap = set(predicted_labels) if predicted_labels else set(top_probability_labels)
        scored = []
        for local_idx, case_idx in enumerate(prefetched):
            case = self.store.cases[int(case_idx)]
            retrieval_cos = float(prefetched_retrieval_sims[local_idx])
            margin_cos = cosine_similarity_01(margins, self.store.margins[int(case_idx)])
            label_overlap = jaccard_labels(query_label_for_overlap, _case_label_set(case))
            case_wave_z = self.store.standardized_wave_features(self.store.wave_features[int(case_idx)])
            wave_sim = wave_similarity_01(query_wave_z, case_wave_z)
            if ablation == "retrieval_only":
                final_score = retrieval_cos
            elif ablation == "margin_only":
                final_score = margin_cos
            else:
                final_score = (
                    0.50 * retrieval_cos
                    + 0.25 * margin_cos
                    + 0.20 * label_overlap
                    + 0.05 * wave_sim
                )
            scored.append(
                (
                    float(final_score),
                    int(case_idx),
                    {
                        "retrieval_cosine": round(float(retrieval_cos), 6),
                        "margin_cosine": round(float(margin_cos), 6),
                        "label_overlap": round(float(label_overlap), 6),
                        "wave_similarity": round(float(wave_sim), 6),
                    },
                )
            )

        scored.sort(key=lambda item: item[0], reverse=True)
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
        ablation: str = "full",
    ) -> SearchResult:
        case = self.store.cases[row_id]
        prob = self.store.probabilities[row_id]
        margins = self.store.margins[row_id]
        wave = self.store.wave_features[row_id]
        top_labels = self._top_probability_labels(prob)
        return self.search(
            probabilities=prob,
            margins=margins,
            wave_features=wave,
            retrieval_vector=self.store.retrieval_vectors[row_id],
            predicted_labels=case.predicted_labels,
            top_probability_labels=top_labels,
            query_case_id=case.case_id,
            query_record_path=case.record_path,
            query_patient_id_hash=case.patient_id_hash if exclude_same_patient else None,
            top_k=top_k,
            prefetch_k=prefetch_k,
            min_candidates=min_candidates,
            score_threshold=score_threshold,
            ablation=ablation,
        )

    def _top_probability_labels(self, probabilities: np.ndarray, n: int = 2) -> List[str]:
        order = np.argsort(-np.asarray(probabilities, dtype=np.float32))[:n]
        return [self.store.classes[int(i)] for i in order]

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
