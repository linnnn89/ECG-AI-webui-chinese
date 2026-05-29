import numpy as np
import unittest

from code.casebank.features import extract_basic_wave_features


class CaseBankFeatureTests(unittest.TestCase):
    def test_basic_wave_features_shape_and_nan_handling(self):
        x = np.zeros((12, 100), dtype=np.float32)
        x[0, 0] = np.nan
        x[1, 1] = np.inf
        features = extract_basic_wave_features(x)
        self.assertEqual(features.shape, (65,))
        self.assertEqual(features.dtype, np.float32)
        self.assertTrue(np.isfinite(features).all())

    def test_basic_wave_features_rejects_wrong_lead_count(self):
        with self.assertRaisesRegex(ValueError, "12 leads"):
            extract_basic_wave_features(np.zeros((11, 100), dtype=np.float32))


if __name__ == "__main__":
    unittest.main()
