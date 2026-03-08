import numpy as np


def validate_n(n):
    """Validate the subdivision count used across the project."""
    if n < 2:
        raise ValueError("N must be >= 2.")


def iter_valid_ij(n):
    """Yield valid lattice indices (i, j) inside the simplex grid."""
    for i in range(n + 1):
        for j in range(n + 1 - i):
            yield i, j


def k_from_ij(n, i, j):
    """Recover the third simplex index k from (n, i, j)."""
    return n - i - j


def iter_valid_ijk(n):
    """Yield all valid simplex lattice triplets (i, j, k)."""
    for i, j in iter_valid_ij(n):
        yield i, j, k_from_ij(n, i, j)


def point_ij_array(n):
    """Return valid point indices as an integer array of shape (P, 2)."""
    return np.array(list(iter_valid_ij(n)), dtype=int)


def triangle_vertex_array(triangle_keys):
    """Convert triangle key tuples into an integer array for fancy indexing."""
    return np.asarray(triangle_keys, dtype=int)


def full_point_count(n):
    """Return the number of valid lattice points in one simplex face."""
    return (n + 1) * (n + 2) // 2


def compact_point_count(n):
    """Return the number of symmetry-reduced representative lattice points."""
    # Number of integer triples (i, j, k) with i+j+k=n and i<=j<=k.
    return (n * (n + 6) + 12) // 12
