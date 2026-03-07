import numpy as np

from coordinate_fileio import load_division_result
from sphere_index_util import point_ij_array


def division_positions_to_array(n, positions):
    point_ij = point_ij_array(n)
    k = n - point_ij[:, 0] - point_ij[:, 1]
    xyz = np.asarray(positions[point_ij[:, 0], point_ij[:, 1]], dtype=float)
    ijk = np.column_stack((point_ij[:, 0] / n, point_ij[:, 1] / n, k / n))
    return np.hstack((ijk, xyz))


def load_division_result_as_array(path):
    n, positions = load_division_result(path)
    return division_positions_to_array(n, positions)
