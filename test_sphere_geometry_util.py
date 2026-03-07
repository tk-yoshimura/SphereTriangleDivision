import unittest

import numpy as np

from sphere_geometry_util import normalize


class SphereGeometryUtilTest(unittest.TestCase):
    def test_normalize_vector(self):
        actual = normalize(np.array([3.0, 0.0, 4.0]))
        expected = np.array([0.6, 0.0, 0.8])
        np.testing.assert_allclose(actual, expected)

    def test_normalize_row_vectors(self):
        actual = normalize(np.array([[3.0, 0.0, 4.0], [0.0, 5.0, 12.0]]))
        expected = np.array([[0.6, 0.0, 0.8], [0.0, 5.0 / 13.0, 12.0 / 13.0]])
        np.testing.assert_allclose(actual, expected)

    def test_normalize_rejects_zero_row_vector(self):
        with self.assertRaises(ValueError):
            normalize(np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0]]))


if __name__ == "__main__":
    unittest.main()
