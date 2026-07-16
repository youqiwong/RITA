import unittest

import numpy as np

from sidset_robustness_protocol import CORRUPTION_LEVELS, apply_corruption, stratified_sample_indices


class ProtocolTests(unittest.TestCase):
    def test_exact_levels(self):
        self.assertEqual(CORRUPTION_LEVELS["noise"], [0, 3, 7, 11, 15, 19, 23])
        self.assertEqual(CORRUPTION_LEVELS["blur"], [1, 3, 7, 11, 15, 19, 23])
        self.assertEqual(CORRUPTION_LEVELS["jpeg"], [100, 90, 80, 70, 60, 50, 40])
        self.assertEqual(CORRUPTION_LEVELS["resize"], [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4])

    def test_stratified_sample_is_fixed(self):
        ratios = np.linspace(0.001, 0.999, 1000)
        first = stratified_sample_indices(ratios, 100, 42)
        second = stratified_sample_indices(ratios, 100, 42)
        self.assertEqual(first, second)
        self.assertEqual(np.bincount(np.asarray(first) // 100, minlength=10).tolist(), [10] * 10)

    def test_corruption_is_deterministic_and_shape_preserving(self):
        image = np.arange(32 * 48 * 3, dtype=np.uint8).reshape(32, 48, 3)
        np.testing.assert_array_equal(apply_corruption(image, "noise", 0, 42, 3), image)
        np.testing.assert_array_equal(apply_corruption(image, "blur", 1, 42, 3), image)
        np.testing.assert_array_equal(apply_corruption(image, "resize", 1.0, 42, 3), image)
        first = apply_corruption(image, "noise", 7, 42, 3)
        second = apply_corruption(image, "noise", 7, 42, 3)
        np.testing.assert_array_equal(first, second)
        self.assertEqual(apply_corruption(image, "resize", 0.4, 42, 3).shape, image.shape)


if __name__ == "__main__":
    unittest.main()
