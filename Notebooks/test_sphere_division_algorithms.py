import unittest

import numpy as np

from sphere_division_algorithms import spherical_triangle_area


class SphereDivisionAlgorithmsTest(unittest.TestCase):
    """Tests for spherical area computations."""

    def test_spherical_triangle_area_single_triangle(self):
        """One octant triangle should have the expected area."""
        a = np.array([1.0, 0.0, 0.0])
        b = np.array([0.0, 1.0, 0.0])
        c = np.array([0.0, 0.0, 1.0])

        actual = spherical_triangle_area(a, b, c)

        np.testing.assert_allclose(actual, np.pi / 2.0)

    def test_spherical_triangle_area_row_vectors(self):
        """Batched row-vector input should return one area per triangle."""
        a = np.array(
            [
                [1.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
            ]
        )
        b = np.array(
            [
                [0.0, 1.0, 0.0],
                [0.0, 1.0, 0.0],
            ]
        )
        c = np.array(
            [
                [0.0, 0.0, 1.0],
                [0.0, 1.0, 1.0],
            ]
        )

        actual = spherical_triangle_area(a, b, c)
        expected = np.array([np.pi / 2.0, np.pi / 4.0])

        np.testing.assert_allclose(actual, expected)


if __name__ == "__main__":
    unittest.main()
