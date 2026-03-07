import itertools
import numpy as np

from sphere_geometry_util import normalize


def _validate_n(n):
    if n < 1:
        raise ValueError("N must be >= 1.")


def _iter_valid_ij(n):
    for i in range(n + 1):
        for j in range(n + 1 - i):
            yield i, j


def _k_from_ij(n, i, j):
    return n - i - j


def lattice_to_octant_point(i, j, k, n):
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
    _validate_n(n)
    points = np.full((n + 1, n + 1, 3), np.nan, dtype=float)
    for i, j in _iter_valid_ij(n):
        points[i, j] = lattice_to_octant_point(i, j, _k_from_ij(n, i, j), n)
    return points


def build_octant_triangle_keys(n):
    _validate_n(n)
    triangles = []
    for i in range(n):
        for j in range(n - i):
            k = _k_from_ij(n, i, j)
            a = (i, j)
            b = (i + 1, j)
            c = (i, j + 1)
            triangles.append((a, b, c))
            if k >= 2:
                d = (i + 1, j + 1)
                triangles.append((b, d, c))
    return triangles


def build_octant_mesh(n):
    points = build_octant_points(n)
    triangle_keys = build_octant_triangle_keys(n)
    tri_xyz = [np.array([points[i, j] for i, j in tri]) for tri in triangle_keys]
    return points, triangle_keys, tri_xyz


def build_point_index(points):
    n = points.shape[0] - 1
    point_keys = sorted(_iter_valid_ij(n), key=lambda t: (t[0] + t[1], t[0], t[1]))
    point_index = {key: idx for idx, key in enumerate(point_keys)}
    return point_keys, point_index


def triangle_side_lengths(tri):
    def ang(u, v):
        return np.arccos(np.clip(np.dot(u, v), -1.0, 1.0))

    a, b, c = tri
    return np.array([ang(a, b), ang(b, c), ang(c, a)])


def planar_triangle_areas(points, triangle_keys):
    areas = []
    for tri in triangle_keys:
        a, b, c = [points[i, j] for i, j in tri]
        areas.append(0.5 * np.linalg.norm(np.cross(b - a, c - a)))
    return np.array(areas, dtype=float)


def outward_normals_check(points, triangle_keys):
    inward_ids = []
    for t_id, tri in enumerate(triangle_keys):
        i, j, k = [points[x, y] for x, y in tri]
        normal = np.cross(j - i, k - i)
        centroid = (i + j + k) / 3.0
        if np.dot(normal, centroid) <= 0.0:
            inward_ids.append(t_id)
    return len(inward_ids) == 0, inward_ids


def lattice_permutation_error(n):
    perms = list(itertools.permutations([0, 1, 2]))
    max_perm_err = 0.0
    worst_case = None

    for i, j in _iter_valid_ij(n):
        k = _k_from_ij(n, i, j)
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
    n = positions.shape[0] - 1
    perms = list(itertools.permutations([0, 1, 2]))
    point_keys = sorted(_iter_valid_ij(n), key=lambda t: (t[0] + t[1], t[0], t[1]))
    max_perm_err = 0.0
    worst_case = None

    for i, j in point_keys:
        k = _k_from_ij(n, i, j)
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
    a = normalize(a)
    b = normalize(b)
    c = normalize(c)
    det = abs(np.dot(a, np.cross(b, c)))
    denom = 1.0 + np.dot(a, b) + np.dot(b, c) + np.dot(c, a)
    return 2.0 * np.arctan2(det, max(denom, 1e-15))


def spherical_triangle_areas(positions, triangle_keys):
    areas = []
    for tri in triangle_keys:
        a, b, c = [positions[i, j] for i, j in tri]
        areas.append(spherical_triangle_area(a, b, c))
    return np.array(areas, dtype=float)


def classify_vertex_constraint(key, n):
    i, j = key
    k = _k_from_ij(n, i, j)
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
        v = normalize(lattice_to_octant_point(i, j, _k_from_ij(n, i, j), n))
        mode2, axis2 = classify_vertex_constraint(key, n)
        if mode2 == "edge":
            v[axis2] = 0.0
            v = normalize(np.maximum(v, 0.0))
        return v

    return v / norm


def project_center_for_vertex(center, key, n):
    mode, axis = classify_vertex_constraint(key, n)
    c = normalize(center)
    if mode == "corner":
        return project_vertex(c, key, n)
    if mode == "edge":
        c = np.maximum(c, 0.0)
        c[axis] = 0.0
        return normalize(c)
    return c


def run_tension_equalizer(n, iterations=500, lr=0.2, lr_decay=True, verbose_every=25):
    points0, triangle_keys, _ = build_octant_mesh(n)
    point_keys = sorted(_iter_valid_ij(n), key=lambda t: (t[0] + t[1], t[0], t[1]))

    positions = np.full_like(points0, np.nan)
    for i, j in point_keys:
        positions[i, j] = project_vertex(normalize(points0[i, j]), (i, j), n)
    history = []

    std_area_prev, max_rel_prev = np.inf, np.inf

    for it in range(1, iterations + 1):
        tri_areas = np.empty(len(triangle_keys), dtype=float)
        tri_centers = []

        for t_id, tri in enumerate(triangle_keys):
            a, b, c = [positions[i, j] for i, j in tri]
            tri_areas[t_id] = spherical_triangle_area(a, b, c)
            tri_centers.append(normalize(a + b + c))

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

        for t_id, tri in enumerate(triangle_keys):
            rel = (tri_areas[t_id] - mean_area) / max(mean_area, 1e-15)
            center = tri_centers[t_id]
            for vk in tri:
                vi, vj = vk
                v = positions[vi, vj]
                c = project_center_for_vertex(center, vk, n)
                delta = lr * rel * (c - v)
                move_sum[vi, vj] += delta
                move_count[vi, vj] += 1

        new_positions = np.full_like(positions, np.nan)
        for i, j in point_keys:
            if move_count[i, j] > 0:
                move = move_sum[i, j] / float(move_count[i, j])
            else:
                move = np.zeros(3, dtype=float)
            new_positions[i, j] = project_vertex(positions[i, j] + move, (i, j), n)
        positions = new_positions

    return positions, triangle_keys, np.array(history, dtype=float)
