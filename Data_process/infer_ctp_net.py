"""CTP-Net: contextual UNI features -> tissue probability heatmaps.

Method illustration using model_architectures/CTP_Net.py (class CTPNet).
A caller supplies a model with matching trained weights; no weights/data ship here.
"""
import numpy as np
import torch
from .spatial_utils import (
    OFFSETS_XY, prepare_inputs, neighborhood_batches, save_probability_heatmaps,
)

# Must match the class order used during training.
TISSUE_NAMES = (
    'Tumor', 'Empty area', 'Connective tissue', 'Non-ROI',
    'Muscularis', 'Lymphovascular area', 'Immune area', 'Adipose area',
)


@torch.inference_mode()
def infer_tissue_probabilities(model, features, coords_xy, grid_shape,
                               batch_size=256, device='cpu'):
    """Return [H, W, 8] probabilities on the original patch lattice.

    features: [N, 1024] frozen UNI embeddings.
    coords_xy: [N, 2] integer (column, row) grid indices.
    grid_shape: (H, W), retained even if some positions have no embedding.
    All available patches serve as centres. Unobserved positions remain NaN.
    """
    features, coords, lookup = prepare_inputs(features, coords_xy, grid_shape)
    model = model.to(device).eval()  # Disable model and neighbour dropout.
    positions = torch.as_tensor(OFFSETS_XY, dtype=torch.float32, device=device)
    heatmap = np.full((*grid_shape, 8), np.nan, dtype=np.float32)
    for rows, tokens, valid in neighborhood_batches(
            features, coords, lookup, np.arange(len(features)), batch_size):
        logits = model(features=tokens.to(device), valid_mask=valid.to(device),
                       relative_positions=positions)
        probabilities = logits.softmax(dim=-1).cpu().numpy()
        if probabilities.shape != (len(rows), 8):
            raise ValueError('CTP-Net must produce eight tissue channels.')
        x, y = coords[rows].T
        heatmap[y, x] = probabilities
    return heatmap


def export_tissue_heatmaps(probabilities, output_dir):
    save_probability_heatmaps(probabilities, TISSUE_NAMES, output_dir,
                              stem='tissue_probability_heatmaps')
