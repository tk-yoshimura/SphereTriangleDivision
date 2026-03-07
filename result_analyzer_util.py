import numpy as np

from coordinate_fileio import load_division_result
from sphere_index_util import iter_valid_ijk


def division_positions_to_array(n, positions):
    rows = []
    for i, j, k in iter_valid_ijk(n):
        x, y, z = np.asarray(positions[i, j], dtype=float)
        rows.append([i / n, j / n, k / n, x, y, z])
    return np.asarray(rows, dtype=float)


def load_division_result_as_array(path):
    n, positions = load_division_result(path)
    return division_positions_to_array(n, positions)
