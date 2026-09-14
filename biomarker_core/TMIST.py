from __future__ import annotations
import math
import numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates

TUMOUR_CLASS = 0
MUSCULARIS_CLASS = 4
DELTA_EPSILON = 1e-12

def validate_tissue_probabilities(
    tissue: np.ndarray, expected_shape: tuple[int, int]
) -> np.ndarray:
    if tissue.ndim != 3 or tissue.shape[:2] != expected_shape or tissue.shape[-1] != 8:
        raise ValueError(f"Expected tissue {expected_shape + (8,)}, found {tissue.shape}")
    values = np.asarray(tissue, dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("Tissue probability map contains NaN/Inf")
    minimum = float(values.min())
    if minimum < -1e-5:
        raise ValueError(f"Tissue probability below tolerance: {minimum}")
    values = np.clip(values, 0.0, None)
    totals = values.sum(axis=-1, keepdims=True)
    if np.any(totals <= 0):
        raise ValueError("Tissue probability vector has non-positive total")
    return values / totals


def sample_state_interface_exposure(
    tissue_probabilities: np.ndarray,
    states: list[dict],
    sigma: float,
    tumour_class: int = TUMOUR_CLASS,
    muscularis_class: int = MUSCULARIS_CLASS,
) -> tuple[np.ndarray, dict[str, float]]:

    if sigma <= 0:
        raise ValueError("sigma must be positive")
    tumour_context = gaussian_filter(
        tissue_probabilities[..., tumour_class], sigma=float(sigma), mode="nearest"
    )
    muscularis_context = gaussian_filter(
        tissue_probabilities[..., muscularis_class], sigma=float(sigma), mode="nearest"
    )
    height, width = tissue_probabilities.shape[:2]
    cols = np.asarray([row["norm_col_center"] * width for row in states])
    rows = np.asarray([row["norm_row_center"] * height for row in states])
    coordinates = np.vstack((rows, cols))
    tumour_at_state = map_coordinates(tumour_context, coordinates, order=1, mode="nearest")
    muscularis_at_state = map_coordinates(
        muscularis_context, coordinates, order=1, mode="nearest"
    )
    exposure = tumour_at_state * muscularis_at_state
    if not np.isfinite(exposure).all() or np.any(exposure < 0):
        raise ValueError("Invalid state interface exposure")
    return exposure, {
        "state_interface_exposure_mean": float(exposure.mean()),
        "state_interface_exposure_median": float(np.median(exposure)),
        "state_interface_exposure_max": float(exposure.max()),
        "state_tumour_context_mean": float(tumour_at_state.mean()),
        "state_muscularis_context_mean": float(muscularis_at_state.mean()),
        "global_muscularis_probability_mean": float(
            tissue_probabilities[..., muscularis_class].mean()
        ),
    }


def invasive_front_pair_metrics(
    states: list[dict],
    state_exposure: np.ndarray,
    delta_epsilon: float = DELTA_EPSILON,
) -> dict[str, float | int | str]:
    if len(states) != len(state_exposure):
        raise ValueError("State/exposure length mismatch")
    if len(states) < 2:
        return {
            "status": "insufficient_clusters",
            "state_count": len(states),
            "valid_pair_count": 0,
            "ifti_raw": math.nan,
            "ifti_log1p": math.nan,
            "conditional_speed_raw": math.nan,
            "conditional_speed_log1p": math.nan,
            "unweighted_pairmean_speed": math.nan,
            "unweighted_pairmean_log1p": math.nan,
            "pair_interface_weight_mean": math.nan,
        }
    coordinates = np.asarray(
        [[row["norm_col_center"], row["norm_row_center"]] for row in states],
        dtype=np.float64,
    )
    times = np.asarray(
        [row["mean_reference_pseudotime"] for row in states], dtype=np.float64
    )
    if not np.isfinite(coordinates).all() or not np.isfinite(times).all():
        raise ValueError("Frozen state table contains non-finite values")
    state_exposure = np.asarray(state_exposure, dtype=np.float64)
    if not np.isfinite(state_exposure).all() or np.any(state_exposure < 0):
        raise ValueError("Invalid state exposure")

    differences = coordinates[:, None, :] - coordinates[None, :, :]
    distances = np.sqrt(np.sum(differences * differences, axis=2))
    maximum_distance = float(distances.max())
    if maximum_distance <= 0:
        return {"status": "zero_spatial_extent", "state_count": len(states)}
    distances /= maximum_distance
    delta = np.abs(times[:, None] - times[None, :])
    upper = np.triu(np.ones_like(delta, dtype=bool), k=1)
    valid = upper & (delta > delta_epsilon)
    if not np.any(valid):
        return {"status": "no_valid_pairs", "state_count": len(states)}

    pair_speed = distances[valid] / delta[valid]
    pair_weight = ((state_exposure[:, None] + state_exposure[None, :]) / 2.0)[valid]
    contribution = pair_weight * pair_speed
    ifti_raw = float(contribution.mean())
    weight_sum = float(pair_weight.sum())
    conditional = float(contribution.sum() / weight_sum) if weight_sum > 0 else math.nan
    pairmean = float(pair_speed.mean())
    return {
        "status": "ok",
        "state_count": len(states),
        "valid_pair_count": int(valid.sum()),
        "zero_delta_pair_count": int(np.count_nonzero(upper & (delta <= delta_epsilon))),
        "ifti_raw": ifti_raw,
        "ifti_log1p": math.log1p(ifti_raw),
        "conditional_speed_raw": conditional,
        "conditional_speed_log1p": (
            math.log1p(conditional) if math.isfinite(conditional) else math.nan
        ),
        "unweighted_pairmean_speed": pairmean,
        "unweighted_pairmean_log1p": math.log1p(pairmean),
        "pair_interface_weight_mean": float(pair_weight.mean()),
        "pair_interface_weight_max": float(pair_weight.max()),
        "minimum_pair_pseudotime_delta": float(delta[valid].min()),
    }


def compute_tmist(tissue_probabilities: np.ndarray, states: list[dict]) -> float:
    tissue = validate_tissue_probabilities(tissue_probabilities, tissue_probabilities.shape[:2])
    if len(states) < 2:
        return math.nan
    exposure, _ = sample_state_interface_exposure(tissue, states, sigma=2.0)
    metrics = invasive_front_pair_metrics(states, exposure)
    return float(metrics.get("ifti_log1p", math.nan))

