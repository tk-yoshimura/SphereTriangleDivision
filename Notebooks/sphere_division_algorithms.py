import itertools
import numpy as np

from sphere_geometry_util import normalize
from sphere_index_util import (
    iter_valid_ij,
    iter_valid_ijk,
    k_from_ij,
    point_ij_array,
    triangle_vertex_array,
    validate_n,
)


def lattice_to_octant_point(i, j, k, n):
    """Map a simplex lattice index to an unnormalized octant point."""
    ring_ij, ring_jk, ring_ki = i + j, j + k, k + i

    theta_ij = (np.pi * ring_ij) / (2.0 * n)
    theta_jk = (np.pi * ring_jk) / (2.0 * n)
    theta_ki = (np.pi * ring_ki) / (2.0 * n)

    phi_ij = 0.0 if ring_ij == 0 else (np.pi * j) / (2.0 * ring_ij)
    phi_jk = 0.0 if ring_jk == 0 else (np.pi * k) / (2.0 * ring_jk)
    phi_ki = 0.0 if ring_ki == 0 else (np.pi * i) / (2.0 * ring_ki)

    return np.array(
        [
            np.sin(theta_ij) * np.cos(phi_ij) + np.sin(theta_ki) * np.sin(phi_ki) + np.cos(theta_jk),
            np.sin(theta_jk) * np.cos(phi_jk) + np.sin(theta_ij) * np.sin(phi_ij) + np.cos(theta_ki),
            np.sin(theta_ki) * np.cos(phi_ki) + np.sin(theta_jk) * np.sin(phi_jk) + np.cos(theta_ij),
        ],
        dtype=float,
    ) / 3.0


def build_octant_points(n):
    """Build the point array for one octant simplex face."""
    validate_n(n)
    points = np.full((n + 1, n + 1, 3), np.nan, dtype=float)
    for i, j, k in iter_valid_ijk(n):
        points[i, j] = lattice_to_octant_point(i, j, k, n)
    return points


def build_octant_triangle_keys(n):
    """Build lattice triangle connectivity for one octant face."""
    validate_n(n)
    triangles = []
    for i, j in iter_valid_ij(n - 1):
        k = k_from_ij(n, i, j)
        a = (i, j)
        b = (i + 1, j)
        c = (i, j + 1)
        triangles.append((a, b, c))
        if k >= 2:
            d = (i + 1, j + 1)
            triangles.append((b, d, c))
    return triangles


def build_octant_mesh(n):
    """Build points, triangle keys, and triangle coordinates for one octant."""
    points = build_octant_points(n)
    triangle_keys = build_octant_triangle_keys(n)
    tri_ij = triangle_vertex_array(triangle_keys)
    tri_xyz = points[tri_ij[:, :, 0], tri_ij[:, :, 1]]
    return points, triangle_keys, tri_xyz


def build_point_index(points):
    """Create stable integer ids for valid point indices."""
    n = points.shape[0] - 1
    point_keys = [tuple(idx) for idx in point_ij_array(n).tolist()]
    point_index = {key: idx for idx, key in enumerate(point_keys)}
    return point_keys, point_index


def triangle_side_lengths(tri):
    """Return spherical side lengths for one triangle."""
    def ang(u, v):
        return np.arccos(np.clip(np.dot(u, v), -1.0, 1.0))

    a, b, c = tri
    return np.array([ang(a, b), ang(b, c), ang(c, a)])


def planar_triangle_areas(points, triangle_keys):
    """Return planar triangle areas for all triangles in the mesh."""
    tri_xyz = _triangle_xyz_from_positions(points, triangle_keys)
    return 0.5 * np.linalg.norm(np.cross(tri_xyz[:, 1] - tri_xyz[:, 0], tri_xyz[:, 2] - tri_xyz[:, 0]), axis=1)


def outward_normals_check(points, triangle_keys):
    """Check whether all planar triangle normals point outward."""
    tri_xyz = _triangle_xyz_from_positions(points, triangle_keys)
    normals = np.cross(tri_xyz[:, 1] - tri_xyz[:, 0], tri_xyz[:, 2] - tri_xyz[:, 0])
    centroids = np.mean(tri_xyz, axis=1)
    inward_ids = np.flatnonzero(np.sum(normals * centroids, axis=1) <= 0.0).tolist()
    return len(inward_ids) == 0, inward_ids


def lattice_permutation_error(n):
    """Measure equivariance error of the lattice mapping under axis permutation."""
    perms = list(itertools.permutations([0, 1, 2]))
    max_perm_err = 0.0
    worst_case = None

    for i, j, k in iter_valid_ijk(n):
        v_base = lattice_to_octant_point(i, j, k, n)
        base_idx = np.array([i, j, k], dtype=int)

        for p in perms:
            ip = base_idx[list(p)]
            v_perm_input = lattice_to_octant_point(int(ip[0]), int(ip[1]), int(ip[2]), n)
            v_perm_output = v_base[list(p)]
            err = float(np.linalg.norm(v_perm_input - v_perm_output))
            if err > max_perm_err:
                max_perm_err = err
                worst_case = ((i, j, k), p)
    return max_perm_err, worst_case


def positions_permutation_error(positions):
    """Measure equivariance error of optimized positions under axis permutation."""
    n = positions.shape[0] - 1
    perms = list(itertools.permutations([0, 1, 2]))
    max_perm_err = 0.0
    worst_case = None

    for i, j, k in iter_valid_ijk(n):
        base = positions[i, j]
        idx = np.array([i, j, k], dtype=int)
        for p in perms:
            kp_arr = idx[list(p)]
            lhs = positions[int(kp_arr[0]), int(kp_arr[1])]
            rhs = base[list(p)]
            err = float(np.linalg.norm(lhs - rhs))
            if err > max_perm_err:
                max_perm_err = err
                worst_case = ((i, j), p)
    return max_perm_err, worst_case


def spherical_triangle_area(a, b, c):
    """Compute spherical triangle area for one triangle or a batch of triangles."""
    a = normalize(a)
    b = normalize(b)
    c = normalize(c)
    cross_bc = np.cross(b, c)

    if a.ndim == 1:
        det = abs(np.dot(a, cross_bc))
        denom = 1.0 + np.dot(a, b) + np.dot(b, c) + np.dot(c, a)
        return 2.0 * np.arctan2(det, max(denom, 1e-15))

    if a.ndim == 2:
        det = np.abs(np.sum(a * cross_bc, axis=1))
        denom = 1.0 + np.sum(a * b, axis=1) + np.sum(b * c, axis=1) + np.sum(c * a, axis=1)
        return 2.0 * np.arctan2(det, np.maximum(denom, 1e-15))

    raise ValueError("spherical_triangle_area expects 1D vectors or 2D arrays of row vectors.")


def spherical_triangle_areas(positions, triangle_keys):
    """Compute spherical triangle areas for all mesh triangles."""
    tri_xyz = _triangle_xyz_from_positions(positions, triangle_keys)
    return spherical_triangle_area(tri_xyz[:, 0], tri_xyz[:, 1], tri_xyz[:, 2])


def classify_vertex_constraint(key, n):
    """Classify a lattice point as corner, edge, or interior."""
    i, j = key
    k = k_from_ij(n, i, j)
    zeros = [i == 0, j == 0, k == 0]
    zc = sum(zeros)

    if zc >= 2:
        return ("corner", None)
    if i == 0:
        return ("edge", 0)
    if j == 0:
        return ("edge", 1)
    if k == 0:
        return ("edge", 2)
    return ("interior", None)


def project_vertex(v, key, n):
    """Project one vertex onto the valid octant constraint manifold."""
    mode, axis = classify_vertex_constraint(key, n)

    if mode == "corner":
        i, j = key
        if i == n:
            return np.array([1.0, 0.0, 0.0])
        if j == n:
            return np.array([0.0, 1.0, 0.0])
        return np.array([0.0, 0.0, 1.0])

    v = np.asarray(v, dtype=float).copy()
    v = np.maximum(v, 0.0)

    if mode == "edge":
        v[axis] = 0.0
        free = [0, 1, 2]
        free.remove(axis)
        if v[free[0]] == 0.0 and v[free[1]] == 0.0:
            v[free] = 1.0 / np.sqrt(2.0)

    norm = np.linalg.norm(v)
    if norm <= 0.0:
        i, j = key
        v = normalize(lattice_to_octant_point(i, j, k_from_ij(n, i, j), n))
        mode2, axis2 = classify_vertex_constraint(key, n)
        if mode2 == "edge":
            v[axis2] = 0.0
            v = normalize(np.maximum(v, 0.0))
        return v

    return v / norm


def project_center_for_vertex(center, key, n):
    """Project a triangle center into the feasible region of a given vertex."""
    mode, axis = classify_vertex_constraint(key, n)
    c = normalize(center)
    if mode == "corner":
        return project_vertex(c, key, n)
    if mode == "edge":
        c = np.maximum(c, 0.0)
        c[axis] = 0.0
        return normalize(c)
    return c


def build_projected_positions(n):
    """Build the initial projected positions used by the optimizer."""
    points, _, _ = build_octant_mesh(n)
    point_ij = point_ij_array(n)
    positions = np.full_like(points, np.nan)
    positions[point_ij[:, 0], point_ij[:, 1]] = _project_vertices_batch(
        normalize(points[point_ij[:, 0], point_ij[:, 1]]),
        point_ij,
        n,
    )
    return positions


def _project_vertices_batch(vertices, point_ij, n):
    """Project a batch of vertices onto sphere and boundary constraints."""
    projected = np.asarray(vertices, dtype=float).copy()
    projected = np.maximum(projected, 0.0)

    pi = point_ij[:, 0]
    pj = point_ij[:, 1]
    pk = n - pi - pj

    is_corner = ((pi == n) & (pj == 0)) | ((pi == 0) & (pj == n)) | ((pi == 0) & (pj == 0))
    is_edge_i0 = (pi == 0) & (pj > 0) & (pk > 0)
    is_edge_j0 = (pj == 0) & (pi > 0) & (pk > 0)
    is_edge_k0 = (pk == 0) & (pi > 0) & (pj > 0)

    projected[is_edge_i0, 0] = 0.0
    projected[is_edge_j0, 1] = 0.0
    projected[is_edge_k0, 2] = 0.0

    zero_edge_i0 = is_edge_i0 & (projected[:, 1] == 0.0) & (projected[:, 2] == 0.0)
    zero_edge_j0 = is_edge_j0 & (projected[:, 0] == 0.0) & (projected[:, 2] == 0.0)
    zero_edge_k0 = is_edge_k0 & (projected[:, 0] == 0.0) & (projected[:, 1] == 0.0)

    projected[zero_edge_i0, 1:] = 1.0 / np.sqrt(2.0)
    projected[zero_edge_j0, 0] = 1.0 / np.sqrt(2.0)
    projected[zero_edge_j0, 2] = 1.0 / np.sqrt(2.0)
    projected[zero_edge_k0, 0:2] = 1.0 / np.sqrt(2.0)

    norms = np.linalg.norm(projected, axis=1)
    zero_norm = norms <= 0.0
    if np.any(zero_norm):
        for idx in np.flatnonzero(zero_norm):
            i = int(pi[idx])
            j = int(pj[idx])
            projected[idx] = normalize(lattice_to_octant_point(i, j, k_from_ij(n, i, j), n))
            if is_edge_i0[idx]:
                projected[idx, 0] = 0.0
                projected[idx] = normalize(np.maximum(projected[idx], 0.0))
            elif is_edge_j0[idx]:
                projected[idx, 1] = 0.0
                projected[idx] = normalize(np.maximum(projected[idx], 0.0))
            elif is_edge_k0[idx]:
                projected[idx, 2] = 0.0
                projected[idx] = normalize(np.maximum(projected[idx], 0.0))
        norms = np.linalg.norm(projected, axis=1)

    non_corner = ~is_corner
    projected[non_corner] /= norms[non_corner, None]

    projected[is_corner & (pi == n)] = np.array([1.0, 0.0, 0.0])
    projected[is_corner & (pj == n)] = np.array([0.0, 1.0, 0.0])
    projected[is_corner & (pk == n)] = np.array([0.0, 0.0, 1.0])

    return projected


def _project_centers_for_vertices_batch(centers, point_ij, n):
    """Project a batch of triangle centers for their incident vertices."""
    projected = normalize(centers)

    pi = point_ij[:, 0]
    pj = point_ij[:, 1]
    pk = n - pi - pj

    is_corner = ((pi == n) & (pj == 0)) | ((pi == 0) & (pj == n)) | ((pi == 0) & (pj == 0))
    is_edge_i0 = (pi == 0) & (pj > 0) & (pk > 0)
    is_edge_j0 = (pj == 0) & (pi > 0) & (pk > 0)
    is_edge_k0 = (pk == 0) & (pi > 0) & (pj > 0)

    edge_mask = is_edge_i0 | is_edge_j0 | is_edge_k0
    projected[edge_mask] = np.maximum(projected[edge_mask], 0.0)
    projected[is_edge_i0, 0] = 0.0
    projected[is_edge_j0, 1] = 0.0
    projected[is_edge_k0, 2] = 0.0
    if np.any(edge_mask):
        projected[edge_mask] = normalize(projected[edge_mask])

    projected[is_corner & (pi == n)] = np.array([1.0, 0.0, 0.0])
    projected[is_corner & (pj == n)] = np.array([0.0, 1.0, 0.0])
    projected[is_corner & (pk == n)] = np.array([0.0, 0.0, 1.0])

    return projected


def _triangle_xyz_from_positions(positions, triangle_keys):
    """Gather triangle coordinates from a point array and triangle keys."""
    tri_ij = triangle_vertex_array(triangle_keys)
    return positions[tri_ij[:, :, 0], tri_ij[:, :, 1]]


def run_tension_equalizer(n, iterations=500, lr=0.2, lr_decay=True, verbose_every=25):
    """Iteratively reduce spherical area variance under octant constraints."""
    _, triangle_keys, _ = build_octant_mesh(n)
    point_ij = point_ij_array(n)
    tri_ij = triangle_vertex_array(triangle_keys)

    positions = build_projected_positions(n)
    history = []

    std_area_prev, max_rel_prev = np.inf, np.inf

    for it in range(1, iterations + 1):
        tri_va = positions[tri_ij[:, 0, 0], tri_ij[:, 0, 1]]
        tri_vb = positions[tri_ij[:, 1, 0], tri_ij[:, 1, 1]]
        tri_vc = positions[tri_ij[:, 2, 0], tri_ij[:, 2, 1]]
        tri_areas = spherical_triangle_area(tri_va, tri_vb, tri_vc)
        tri_centers = normalize(tri_va + tri_vb + tri_vc)

        mean_area = float(tri_areas.mean())
        std_area = float(tri_areas.std(ddof=0))
        max_rel = float(np.max(np.abs(tri_areas - mean_area) / max(mean_area, 1e-15)))
        history.append((it, mean_area, std_area, max_rel))

        eps = 1e-14
        if lr_decay and std_area_prev / std_area - 1 <= eps and max_rel_prev / max_rel - 1 <= eps:
            lr *= 0.99
        std_area_prev = std_area
        max_rel_prev = max_rel

        if verbose_every and (it % verbose_every == 0 or it == iterations):
            print(f"iter={it:5d} std={std_area:.12e} max_rel={max_rel:.8e} lr={lr:.4e}")

        if it == iterations or lr < 1e-15:
            break

        move_sum = np.zeros_like(positions)
        move_count = np.zeros((n + 1, n + 1), dtype=int)
        rel = (tri_areas - mean_area) / max(mean_area, 1e-15)
        rel_vertices = np.repeat(rel, 3)
        center_vertices = np.repeat(tri_centers, 3, axis=0)
        vertex_ij = tri_ij.reshape(-1, 2)
        vertex_positions = positions[vertex_ij[:, 0], vertex_ij[:, 1]]
        projected_centers = _project_centers_for_vertices_batch(center_vertices, vertex_ij, n)
        deltas = rel_vertices[:, None] * (projected_centers - vertex_positions)
        np.add.at(move_sum, (vertex_ij[:, 0], vertex_ij[:, 1], slice(None)), deltas)
        np.add.at(move_count, (vertex_ij[:, 0], vertex_ij[:, 1]), 1)

        new_positions = np.full_like(positions, np.nan)
        moves = np.zeros((len(point_ij), 3), dtype=float)
        valid_move = move_count[point_ij[:, 0], point_ij[:, 1]] > 0
        if np.any(valid_move):
            moves[valid_move] = (
                move_sum[point_ij[valid_move, 0], point_ij[valid_move, 1]]
                / move_count[point_ij[valid_move, 0], point_ij[valid_move, 1], None]
            )
        new_positions[point_ij[:, 0], point_ij[:, 1]] = _project_vertices_batch(
            positions[point_ij[:, 0], point_ij[:, 1]] + lr * moves,
            point_ij,
            n,
        )
        positions = new_positions

    return positions, triangle_keys, np.array(history, dtype=float)
