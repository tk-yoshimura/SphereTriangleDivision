import numpy as np

from coordinate_fileio import load_division_result


def division_positions_to_array(n, positions):
    rows = []
    for i, j, k in sorted(positions.keys(), key=lambda t: (t[0], t[1], t[2])):
        x, y, z = np.asarray(positions[(i, j, k)], dtype=float)
        rows.append([i / n, j / n, k / n, x, y, z])
    return np.asarray(rows, dtype=float)


def load_division_result_as_array(path):
    n, positions = load_division_result(path)
    return division_positions_to_array(n, positions)
