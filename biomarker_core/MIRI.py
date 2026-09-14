import numpy as np

def compute_miri(states: list[dict], coloc_m: float, delta_epsilon=1e-12) -> float:
    if not np.isfinite(coloc_m) or not 0 <= coloc_m <= 1:
        raise ValueError("coloc_m must lie in [0, 1]")
    if not np.isfinite(delta_epsilon) or delta_epsilon < 0:
        raise ValueError("delta_epsilon must be finite and nonnegative")
    if len(states) < 2:
        return float('nan')
    xy = np.asarray([[s['norm_col_center'], s['norm_row_center']] for s in states], dtype=float)
    tau = np.asarray([s['mean_reference_pseudotime'] for s in states], dtype=float)
    if not np.isfinite(xy).all() or not np.isfinite(tau).all():
        raise ValueError("Non-finite state centres or pseudotimes")
    distance = np.sqrt(np.sum((xy[:, None] - xy[None, :])**2, axis=2))
    if distance.max() <= 0:
        return float('nan')
    distance /= distance.max()
    delta = np.abs(tau[:, None] - tau[None, :])
    valid = np.triu(np.ones_like(delta, dtype=bool), 1) & (delta > delta_epsilon)
    if not valid.any():
        return float('nan')
    return float(100 * coloc_m * np.log1p(np.mean(distance[valid]/delta[valid])))
