from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class CaseRecord:
    row_id: int
    case_id: str
    source: str
    source_record_id: Optional[str]
    record_path: Optional[str]
    header_path: Optional[str]
    image_path: Optional[str]
    patient_id_hash: Optional[str]
    split: Optional[str]
    labels: List[str]
    predicted_labels: List[str]
    probabilities: Dict[str, float]
    margins: Dict[str, float]
    signal_quality: Dict[str, Any]
    has_embedding: bool


@dataclass
class SimilarCase:
    rank: int
    case_id: str
    score: float
    score_level: str
    components: Dict[str, float]
    record_path: Optional[str]
    image_path: Optional[str]
    source: str
    true_diagnosis: Optional[str]
    label_scope: Optional[str]
    image_url: Optional[str]
    labels: List[str]
    predicted_labels: List[str]
    probabilities: Dict[str, float]
    margins: Dict[str, float]


@dataclass
class SearchResult:
    query: Dict[str, Any]
    similar_cases: List[SimilarCase]
    warnings: List[str]
