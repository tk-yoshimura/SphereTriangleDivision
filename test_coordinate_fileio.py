import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from coordinate_fileio import load_division_result, save_division_result


class CoordinateFileIoTest(unittest.TestCase):
    def test_save_is_sorted_and_compact(self):
        n = 3
        positions = {
            (0, 0, 3): np.array([0.0, 0.0, 1.0]),
            (0, 1, 2): np.array([0.0, 0.3, 0.95]),
            (1, 0, 2): np.array([0.3, 0.0, 0.95]),
            (0, 2, 1): np.array([0.0, 0.7, 0.7]),
            (2, 0, 1): np.array([0.7, 0.0, 0.7]),
            (1, 1, 1): np.array([0.58, 0.58, 0.58]),
            (0, 3, 0): np.array([0.0, 1.0, 0.0]),
            (1, 2, 0): np.array([0.4, 0.9, 0.0]),
            (2, 1, 0): np.array([0.9, 0.4, 0.0]),
            (3, 0, 0): np.array([1.0, 0.0, 0.0]),
        }

        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "division_result_3.json"
            save_division_result(out, n, positions)
            payload = json.loads(out.read_text(encoding="utf-8"))

        self.assertEqual(payload["N"], 3)
        points = payload["points"]
        self.assertTrue(all(p["i"] <= p["j"] for p in points))

        actual_pairs = [(p["i"], p["j"]) for p in points]
        expected_pairs = sorted(set((i, j) for (i, j, _k) in positions if i <= j), key=lambda t: (t[0], t[1]))
        self.assertEqual(actual_pairs, expected_pairs)

    def test_load_restores_swapped_points(self):
        n = 3
        payload = {
            "N": n,
            "points": [
                {"i": 0, "j": 1, "xyz": [0.0, 0.3, 0.95]},
                {"i": 1, "j": 1, "xyz": [0.58, 0.58, 0.58]},
                {"i": 0, "j": 3, "xyz": [0.0, 1.0, 0.0]},
            ],
        }

        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "division_result_3.json"
            src.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            loaded_n, positions = load_division_result(src)

        self.assertEqual(loaded_n, n)
        self.assertTrue((0, 1, 2) in positions)
        self.assertTrue((1, 0, 2) in positions)
        np.testing.assert_allclose(positions[(0, 1, 2)], np.array([0.0, 0.3, 0.95]))
        np.testing.assert_allclose(positions[(1, 0, 2)], np.array([0.3, 0.0, 0.95]))


if __name__ == "__main__":
    unittest.main()
