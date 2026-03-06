import json
from pathlib import Path

import numpy as np


def _sorted_keys(positions):
    return sorted(positions.keys(), key=lambda t: (t[0], t[1], t[2]))


def save_division_result(path, n, positions):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    points = []
    for i, j, _k in _sorted_keys(positions):
        if i > j:
            continue
        xyz = np.asarray(positions[(i, j, n - i - j)], dtype=float).tolist()
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
        xyz = np.asarray(rec["xyz"], dtype=float)

        if i <= j:
            canonical[(i, j)] = xyz
        else:
            # Backward compatibility: normalize old redundant entries.
            canonical[(j, i)] = np.array([xyz[1], xyz[0], xyz[2]], dtype=float)

    positions = {}
    for (i, j), xyz in canonical.items():
        k = n - i - j
        positions[(i, j, k)] = xyz
        if i < j:
            positions[(j, i, k)] = np.array([xyz[1], xyz[0], xyz[2]], dtype=float)

    return n, positions
