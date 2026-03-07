import json
import itertools
from pathlib import Path

import numpy as np
from sphere_index_util import full_point_count, compact_point_count, iter_valid_ij, k_from_ij


def _canonicalize_triplet(i, j, k, xyz):
    idx = np.array([int(i), int(j), int(k)], dtype=int)
    xyz = np.asarray(xyz, dtype=float)

    candidates = []
    for p in itertools.permutations([0, 1, 2]):
        ip = np.array([idx[p[0]], idx[p[1]], idx[p[2]]], dtype=int)
        vp = np.array([xyz[p[0]], xyz[p[1]], xyz[p[2]]], dtype=float)

        if ip[0] > ip[1]:
            continue
        if ip[1] == ip[2] and ip[0] != ip[1]:
            continue
        candidates.append((ip, vp))

    if not candidates:
        return (int(idx[0]), int(idx[1]), int(idx[2])), np.asarray(xyz, dtype=float)

    best_ip, best_vp = min(
        candidates,
        key=lambda t: (0 if t[0][0] == t[0][1] else 1, int(t[0][0]), int(t[0][1]), int(t[0][2])),
    )
    return (int(best_ip[0]), int(best_ip[1]), int(best_ip[2])), np.asarray(best_vp, dtype=float)


def _validate_positions_shape(n, positions):
    if not isinstance(positions, np.ndarray):
        raise TypeError("positions must be a numpy.ndarray.")
    expected_shape = (n + 1, n + 1, 3)
    if positions.shape != expected_shape:
        raise ValueError(f"positions shape must be {expected_shape}, got {positions.shape}.")


def save_division_result(path, n, positions, index_averaging=True):
    _validate_positions_shape(n, positions)

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    canonical = {}
    for i, j in iter_valid_ij(n):
        if np.isnan(positions[i, j]).any():
            continue
        k = k_from_ij(n, i, j)
        key_c, xyz_c = _canonicalize_triplet(i, j, k, positions[i, j])
        if key_c not in canonical:
            canonical[key_c] = xyz_c

    points = []
    for i, j, k in sorted(canonical.keys()):
        if i > j:
            continue
        if j == k and i != j:
            continue
        x, y, z = canonical[(i, j, k)].tolist()

        if index_averaging:
            if i == j and j == k:
                x = y = z = np.sqrt(3) / 3
            elif i == j:
                x = y = (x + y) / 2 if k > 0 else np.sqrt(2) / 2
            elif j == k:
                y = z = (y + z) / 2 if i > 0 else np.sqrt(2) / 2
            elif i == k:
                x = z = (x + z) / 2 if j > 0 else np.sqrt(2) / 2

        points.append({"i": int(i), "j": int(j), "xyz": [float(x), float(y), float(z)]})

    payload = {"N": int(n), "points": points}
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def load_division_result(path):
    path = Path(path)
    payload = json.loads(path.read_text(encoding="utf-8"))

    n = int(payload["N"])
    canonical = {}
    for rec in payload["points"]:
        i = int(rec["i"])
        j = int(rec["j"])
        k = k_from_ij(n, i, j)
        xyz = np.asarray(rec["xyz"], dtype=float)
        key_c, xyz_c = _canonicalize_triplet(i, j, k, xyz)
        canonical[key_c] = xyz_c

    positions = np.full((n + 1, n + 1, 3), np.nan, dtype=float)
    perms = list(itertools.permutations([0, 1, 2]))
    for (i, j, k), xyz in canonical.items():
        idx = np.array([i, j, k], dtype=int)
        for p in perms:
            ip = np.array([idx[p[0]], idx[p[1]], idx[p[2]]], dtype=int)
            xyz_p = np.array([xyz[p[0]], xyz[p[1]], xyz[p[2]]], dtype=float)
            pi = int(ip[0])
            pj = int(ip[1])
            if 0 <= pi <= n and 0 <= pj <= n and (pi + pj) <= n:
                positions[pi, pj] = xyz_p

    return n, positions


def validate_division_result(path, tol=1e-12):
    path = Path(path)
    payload = json.loads(path.read_text(encoding="utf-8"))

    n = int(payload["N"])
    point_records = payload.get("points", [])
    stored_count = len(point_records)
    expected_compact = compact_point_count(n)
    expected_full = full_point_count(n)

    index_errors = []
    for rec in point_records:
        i = int(rec["i"])
        j = int(rec["j"])
        if i < 0 or j < 0 or i > n or j > n or i + j > n:
            index_errors.append({"i": i, "j": j})

    _, positions = load_division_result(path)
    full_count = 0
    for i, j in iter_valid_ij(n):
        if not np.isnan(positions[i, j]).any():
            full_count += 1

    sphere_violations = []
    arc_violations = []
    for i, j in iter_valid_ij(n):
        k = k_from_ij(n, i, j)
        v = positions[i, j]
        x, y, z = np.asarray(v, dtype=float)
        norm = float(np.linalg.norm([x, y, z]))
        if abs(norm - 1.0) > tol:
            sphere_violations.append({"key": [i, j, k], "norm": norm})

        if i == 0 and abs(x) > tol:
            arc_violations.append({"key": [i, j, k], "axis": "x", "value": float(x)})
        if j == 0 and abs(y) > tol:
            arc_violations.append({"key": [i, j, k], "axis": "y", "value": float(y)})
        if k == 0 and abs(z) > tol:
            arc_violations.append({"key": [i, j, k], "axis": "z", "value": float(z)})

    counts_ok = stored_count == expected_compact and full_count == expected_full and len(index_errors) == 0
    sphere_ok = len(sphere_violations) == 0
    arc_ok = len(arc_violations) == 0
    valid = counts_ok and sphere_ok and arc_ok

    return {
        "valid": valid,
        "N": n,
        "counts": {
            "stored_points": stored_count,
            "expected_stored_points": expected_compact,
            "expanded_points": full_count,
            "expected_expanded_points": expected_full,
            "ok": counts_ok,
        },
        "index_check": {
            "ok": len(index_errors) == 0,
            "error_count": len(index_errors),
            "errors_preview": index_errors[:10],
        },
        "sphere_constraint": {
            "ok": sphere_ok,
            "violation_count": len(sphere_violations),
            "violations_preview": sphere_violations[:10],
        },
        "arc_constraint": {
            "ok": arc_ok,
            "violation_count": len(arc_violations),
            "violations_preview": arc_violations[:10],
        },
    }
