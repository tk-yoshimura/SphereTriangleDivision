import numpy as np


def normalize(v):
    v = np.asarray(v, dtype=float)
    if v.ndim == 1:
        n = np.linalg.norm(v)
        if n == 0.0:
            raise ValueError('Zero vector cannot be normalized.')
        return v / n

    if v.ndim == 2:
        n = np.linalg.norm(v, axis=1, keepdims=True)
        if np.any(n == 0.0):
            raise ValueError('Zero vector cannot be normalized.')
        return v / n

    raise ValueError('normalize expects a 1D vector or a 2D array of row vectors.')


def slerp(p0, p1, t):
    p0 = normalize(p0)
    p1 = normalize(p1)
    dot = np.clip(np.dot(p0, p1), -1.0, 1.0)
    omega = np.arccos(dot)
    if np.isclose(omega, 0.0):
        return p0
    so = np.sin(omega)
    return np.sin((1.0 - t) * omega) / so * p0 + np.sin(t * omega) / so * p1


def geodesic_arc(p0, p1, samples=24):
    ts = np.linspace(0.0, 1.0, samples)
    pts = np.array([slerp(p0, p1, t) for t in ts])
    return pts
