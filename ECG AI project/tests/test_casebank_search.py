import tempfile
import unittest
from pathlib import Path

import numpy as np

from code.casebank.io import write_case_rows, write_json
from code.casebank.search import CaseBankStore, SearchEngine


def _write_synthetic_casebank(tmp_path):
    rows = [
        {
            "row_id": 0,
            "case_id": "ptbxl_00001_hr",
            "source": "PTBXL",
            "source_record_id": "1",
            "record_path": "records500/00000/00001_hr",
            "patient_id_hash": "p0",
            "split": "train",
            "labels": ["A"],
            "predicted_labels": ["A"],
            "probabilities": {"A": 0.9, "B": 0.1},
            "margins": {"A": 0.4, "B": -0.4},
            "signal_quality": {},
            "has_embedding": False,
            "build_version": "test",
            "created_at": "2026-05-26T00:00:00",
        },
        {
            "row_id": 1,
            "case_id": "ptbxl_00002_hr",
            "source": "PTBXL",
            "source_record_id": "2",
            "record_path": "records500/00000/00002_hr",
            "patient_id_hash": "p0",
            "split": "train",
            "labels": ["A"],
            "predicted_labels": ["A"],
            "probabilities": {"A": 0.88, "B": 0.12},
            "margins": {"A": 0.38, "B": -0.38},
            "signal_quality": {},
            "has_embedding": False,
            "build_version": "test",
            "created_at": "2026-05-26T00:00:00",
        },
        {
            "row_id": 2,
            "case_id": "ptbxl_00003_hr",
            "source": "PTBXL",
            "source_record_id": "3",
            "record_path": "records500/00000/00003_hr",
            "patient_id_hash": "p2",
            "split": "train",
            "labels": ["A"],
            "predicted_labels": ["A"],
            "probabilities": {"A": 0.87, "B": 0.13},
            "margins": {"A": 0.37, "B": -0.37},
            "signal_quality": {},
            "has_embedding": False,
            "build_version": "test",
            "created_at": "2026-05-26T00:00:00",
        },
        {
            "row_id": 3,
            "case_id": "ptbxl_00004_hr",
            "source": "PTBXL",
            "source_record_id": "4",
            "record_path": "records500/00000/00004_hr",
            "patient_id_hash": "p3",
            "split": "train",
            "labels": ["B"],
            "predicted_labels": ["B"],
            "probabilities": {"A": 0.1, "B": 0.9},
            "margins": {"A": -0.4, "B": 0.4},
            "signal_quality": {},
            "has_embedding": False,
            "build_version": "test",
            "created_at": "2026-05-26T00:00:00",
        },
    ]
    probs = np.array([[0.9, 0.1], [0.88, 0.12], [0.87, 0.13], [0.1, 0.9]], dtype=np.float32)
    margins = np.array([[0.4, -0.4], [0.38, -0.38], [0.37, -0.37], [-0.4, 0.4]], dtype=np.float32)
    wave = np.array([[0.0, 0.0], [0.01, 0.0], [0.02, 0.0], [3.0, 3.0]], dtype=np.float32)
    retrieval = np.array([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [-1.0, 0.0]], dtype=np.float32)
    np.save(tmp_path / "case_probs.npy", probs)
    np.save(tmp_path / "case_margins.npy", margins)
    np.save(tmp_path / "case_wave_features.npy", wave)
    np.save(tmp_path / "case_retrieval_vectors.npy", retrieval)
    write_case_rows(tmp_path / "case_index.sqlite", rows)
    write_json(
        tmp_path / "vector_stats.json",
        {
            "probabilities": {"mean": [0.0, 0.0], "std": [1.0, 1.0]},
            "margins": {"mean": [0.0, 0.0], "std": [1.0, 1.0]},
            "wave_features": {"mean": [0.0, 0.0], "std": [1.0, 1.0]},
        },
    )
    write_json(tmp_path / "build_config.json", {"classes": ["A", "B"], "num_cases": 4})
    return tmp_path


class CaseBankSearchTests(unittest.TestCase):
    def test_search_excludes_self_and_same_patient_without_padding(self):
        with tempfile.TemporaryDirectory() as tmp:
            casebank = _write_synthetic_casebank(Path(tmp))
            engine = SearchEngine(CaseBankStore.load(str(casebank)))
            result = engine.search_from_row(
                0,
                top_k=10,
                min_candidates=1,
                score_threshold=0.55,
                exclude_same_patient=True,
            )
            self.assertEqual([case.case_id for case in result.similar_cases], ["ptbxl_00003_hr"])
            self.assertEqual(len(result.similar_cases), 1)
            self.assertTrue(all(case.case_id != "ptbxl_00001_hr" for case in result.similar_cases))
            self.assertTrue(all(case.case_id != "ptbxl_00002_hr" for case in result.similar_cases))

    def test_search_respects_score_threshold(self):
        with tempfile.TemporaryDirectory() as tmp:
            casebank = _write_synthetic_casebank(Path(tmp))
            engine = SearchEngine(CaseBankStore.load(str(casebank)))
            result = engine.search_from_row(
                0,
                top_k=10,
                min_candidates=1,
                score_threshold=1.01,
                exclude_same_patient=True,
            )
            self.assertEqual(result.similar_cases, [])
            self.assertTrue(any("not padded" in warning for warning in result.warnings))


if __name__ == "__main__":
    unittest.main()
