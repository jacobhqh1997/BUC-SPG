"""TNSL-Net: contextual UNI features -> niche score distribution heatmap.

Method illustration using model_architectures/TNSL_Net.py (class TNSLNetV2).
The four niche scores are continuous softmax probabilities, not differentiation
scores or the Lum/IM/B-EMT/B-IS cell-state labels.
"""
import numpy as np
import torch
from .spatial_utils import (
    OFFSETS_XY, prepare_inputs, neighborhood_batches, save_probability_heatmaps,
)

NICHE_NAMES = ('N1', 'N2', 'N3', 'N4')


@torch.inference_mode()
def infer_niche_probabilities(model, features, coords_xy, tissue_probabilities,
                              batch_size=256, device='cpu'):
    """Return [H, W, 4] niche probabilities registered to the CTP-Net grid.

    Only tissue argmax == 0 (Tumor) positions are prediction centres.
    Neighbours use ALL available UNI embeddings, including non-tumour tissue.
    Non-tumour / unevaluated positions remain NaN, not a fifth niche class.
    """
    tissue = np.asarray(tissue_probabilities)
    if tissue.ndim != 3 or tissue.shape[-1] != 8:
        raise ValueError('Expected CTP-Net tissue probabilities [H, W, 8].')
    features, coords, lookup = prepare_inputs(features, coords_xy, tissue.shape[:2])
    finite = np.isfinite(tissue).all(axis=-1)
    tumour = finite & (np.argmax(np.where(np.isfinite(tissue), tissue, -np.inf),
                                  axis=-1) == 0)
    # Do not silently omit a tumour centre that lacks its matched UNI feature.
    for y, x in np.argwhere(tumour):
        if (x, y) not in lookup:
            raise ValueError('A tumour centre is missing its matching UNI embedding.')
    centre_rows = np.flatnonzero(tumour[coords[:, 1], coords[:, 0]])
    model = model.to(device).eval()
    positions = torch.as_tensor(OFFSETS_XY, dtype=torch.float32, device=device)
    heatmap = np.full((*tissue.shape[:2], 4), np.nan, dtype=np.float32)
    for rows, tokens, valid in neighborhood_batches(
            features, coords, lookup, centre_rows, batch_size):
        logits = model(uni_features=tokens.to(device), relative_positions=positions,
                       valid_mask=valid.to(device))
        probabilities = logits.softmax(dim=-1).cpu().numpy()
        if probabilities.shape != (len(rows), 4):
            raise ValueError('TNSL-Net must produce four niche channels.')
        x, y = coords[rows].T
        heatmap[y, x] = probabilities
    return heatmap


def export_niche_heatmaps(probabilities, output_dir):
    save_probability_heatmaps(probabilities, NICHE_NAMES, output_dir,
                              stem='niche_score_distribution_heatmap')
