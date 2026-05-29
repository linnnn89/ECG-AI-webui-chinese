import numpy as np
import unittest

from code.casebank.similarity import cosine_similarity_01, jaccard_labels, wave_similarity_01


class CaseBankSimilarityTests(unittest.TestCase):
    def test_cosine_similarity_01_range_and_direction(self):
        self.assertEqual(cosine_similarity_01(np.array([1, 0]), np.array([1, 0])), 1.0)
        self.assertEqual(cosine_similarity_01(np.array([1, 0]), np.array([-1, 0])), 0.0)
        value = cosine_similarity_01(np.array([1, 0]), np.array([0, 1]))
        self.assertTrue(0.49 <= value <= 0.51)

    def test_jaccard_labels_edges(self):
        self.assertEqual(jaccard_labels(set(), set()), 0.0)
        self.assertEqual(jaccard_labels({"MI"}, {"MI", "STTC"}), 0.5)
        self.assertEqual(jaccard_labels({"MI"}, {"MI"}), 1.0)

    def test_wave_similarity_range(self):
        value = wave_similarity_01(np.array([0.0, 1.0]), np.array([0.0, 2.0]))
        self.assertTrue(0.0 < value <= 1.0)


if __name__ == "__main__":
    unittest.main()
