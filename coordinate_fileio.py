import json
import itertools
from pathlib import Path

import numpy as np


def _sorted_keys(positions):
    return sorted(positions.keys(), key=lambda t: (t[0], t[1], t[2]))


def _expected_point_counts(n):
    full_count = (n + 1) * (n + 2) // 2
    compact_count = 0
    for i in range(n + 1):
        for j in range(n + 1 - i):
            k = n - i - j
            if i <= j <= k:
                compact_count += 1
    return compact_count, full_count


def _canonicalize_triplet(i, j, k, xyz):
    idx = np.array([int(i), int(j), int(k)], dtype=int)
    xyz = np.asarray(xyz, dtype=float)

    candidates = []
    for p in itertools.permutations([0, 1, 2]):
        ip = np.array([idx[p[0]], idx[p[1]], idx[p[2]]], dtype=int)
        vp = np.array([xyz[p[0]], xyz[p[1]], xyz[p[2]]], dtype=float)

        # Keep i<=j, and drop redundant j==k (except all-equal).
        if ip[0] > ip[1]:
            continue
        if ip[1] == ip[2] and ip[0] != ip[1]:
            continue
        candidates.append((ip, vp))

    if not candidates:
        # Fallback for safety; should not happen for valid lattice indices.
        return (int(idx[0]), int(idx[1]), int(idx[2])), np.asarray(xyz, dtype=float)

    # Prefer i==j representative, then lexicographic order.
    best_ip, best_vp = min(candidates, key=lambda t: (0 if t[0][0] == t[0][1] else 1, int(t[0][0]), int(t[0][1]), int(t[0][2])))
    return (int(best_ip[0]), int(best_ip[1]), int(best_ip[2])), np.asarray(best_vp, dtype=float)


def save_division_result(path, n, positions):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    canonical = {}
    for i, j, _k in _sorted_keys(positions):
        k = n - i - j
        key_c, xyz_c = _canonicalize_triplet(i, j, k, positions[(i, j, k)])
        if key_c not in canonical:
            canonical[key_c] = xyz_c

    points = []
    for i, j, k in sorted(canonical.keys()):
        if i > j:
            continue
        if j == k and i != j:
            continue
        xyz = canonical[(i, j, k)].tolist()
        points.append({"i": int(i), "j": int(j), "xyz": [float(xyz[0]), float(xyz[1]), float(xyz[2])]})

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
        k = n - i - j
        xyz = np.asarray(rec["xyz"], dtype=float)
        key_c, xyz_c = _canonicalize_triplet(i, j, k, xyz)
        canonical[key_c] = xyz_c

    positions = {}
    perms = list(itertools.permutations([0, 1, 2]))
    for (i, j, k), xyz in canonical.items():
        idx = np.array([i, j, k], dtype=int)
        for p in perms:
            key_p = (int(idx[p[0]]), int(idx[p[1]]), int(idx[p[2]]))
            xyz_p = np.array([xyz[p[0]], xyz[p[1]], xyz[p[2]]], dtype=float)
            positions[key_p] = xyz_p

    return n, positions


def validate_division_result(path, tol=1e-12):
    path = Path(path)
    payload = json.loads(path.read_text(encoding="utf-8"))

    n = int(payload["N"])
    point_records = payload.get("points", [])
    stored_count = len(point_records)
    expected_compact, expected_full = _expected_point_counts(n)

    index_errors = []
    for rec in point_records:
        i = int(rec["i"])
        j = int(rec["j"])
        if i < 0 or j < 0 or i > n or j > n or i + j > n:
            index_errors.append({"i": i, "j": j})

    _, positions = load_division_result(path)
    full_count = len(positions)

    sphere_violations = []
    arc_violations = []
    for (i, j, k), v in positions.items():
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
