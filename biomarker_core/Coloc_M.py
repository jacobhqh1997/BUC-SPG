import numpy as np
def compute_coloc_m(t, m) -> float:

    t, m = np.asarray(t, dtype=float), np.asarray(m, dtype=float)
    if t.ndim != 1 or m.shape != t.shape:
        raise ValueError("t and m must be matching one-dimensional arrays")
    if not np.isfinite(t).all() or not np.isfinite(m).all():
        raise ValueError("Non-finite local proportions")
    if np.any(t < 0) or np.any(m < 0) or np.any(t > 1) or np.any(m > 1):
        raise ValueError("Local proportions must lie in [0, 1]")
    denominator = float(np.sum(t*t) + np.sum(m*m))
    return float(2*np.sum(t*m)/denominator) if denominator > 0 else 0.0
