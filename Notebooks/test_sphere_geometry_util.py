import unittest

import numpy as np

from sphere_geometry_util import normalize


class SphereGeometryUtilTest(unittest.TestCase):
    """Tests for low-level spherical geometry helpers."""

    def test_normalize_vector(self):
        """A single vector should be normalized in the usual way."""
        actual = normalize(np.array([3.0, 0.0, 4.0]))
        expected = np.array([0.6, 0.0, 0.8])
        np.testing.assert_allclose(actual, expected)

    def test_normalize_row_vectors(self):
        """A batch of row vectors should be normalized row by row."""
        actual = normalize(np.array([[3.0, 0.0, 4.0], [0.0, 5.0, 12.0]]))
        expected = np.array([[0.6, 0.0, 0.8], [0.0, 5.0 / 13.0, 12.0 / 13.0]])
        np.testing.assert_allclose(actual, expected)

    def test_normalize_rejects_zero_row_vector(self):
        """A zero vector in a batch should still be rejected."""
        with self.assertRaises(ValueError):
            normalize(np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0]]))


if __name__ == "__main__":
    unittest.main()
