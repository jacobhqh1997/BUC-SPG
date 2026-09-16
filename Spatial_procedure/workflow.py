
import numpy as np
import pandas as pd
from .preprocessing import preprocess, integrate_and_annotate
from .ecosystems import whole_tissue_domains, anchor_context, fit_ecosystems, tile_soft_labels
from .differentiation import matched_program_scores, pooled_reference_coordinates


def discovery_workflow(raw, reviewed_annotation, tile_membership, config):
    """See README for exact caller-owned inputs.

    Required configuration is explicit: max_epochs, min_total_cells,
    min_tumour_cells, per_sample, n_pcs, raw_to_niche_order, reference_anchor_ids.
    A fresh GMM requires its own reviewed label mapping before producing N1–N4.
    """
    a = preprocess(raw)
    # Indexed annotations preserve alignment when QC removes cells.
    reviewed = reviewed_annotation.reindex(a.obs_names)
    a = integrate_and_annotate(a, reviewed.to_numpy(), config['max_epochs'])
    a, domain_search = whole_tissue_domains(a)
    ids, features, weights = anchor_context(
        a, config['min_total_cells'], config['min_tumour_cells'])
    model, anchors, ecosystem_search = fit_ecosystems(
        features, a.obs.iloc[ids]['sample'].to_numpy(), weights,
        config['per_sample'], config['n_pcs'])
    # CellCharter in the study returns a torch tensor from predict_proba.
    q_raw = model.predict_proba(anchors.obsm['X_ecosystem']).detach().cpu().numpy()
    order = config['raw_to_niche_order']
    if sorted(order) != [0,1,2,3]:
        raise ValueError('Provide a reviewed raw-cluster -> N1–N4 permutation.')
    anchor_table = pd.DataFrame(q_raw, columns=['q0','q1','q2','q3'])
    anchor_table.insert(0, 'anchor_id', a.obs_names[ids])
    targets = tile_soft_labels(anchor_table, tile_membership, order)
    # Fixed reviewed reference subset, not every available cell or every anchor.
    reference = np.isin(a.obs_names[ids], config['reference_anchor_ids'])
    if not reference.any():
        raise ValueError('Supply reference anchor IDs from the reviewed reference subset.')
    scores = matched_program_scores(a.X[ids[reference]], a.var_names)
    d, tau = pooled_reference_coordinates(scores, q_raw[reference][:,order],
                                          np.ones(reference.sum(), dtype=bool))
    return {'annotated_cells': a, 'tile_soft_targets': targets,
            'reference_differentiation': d, 'niche_coordinates': tau,
            'domain_stability': domain_search, 'ecosystem_stability': ecosystem_search}
