"""Whole-tissue domains, tumour-anchor ecosystems and tile-level soft labels.

CellCharter calls reflect the study workflow; package versions must be compatible.
This transparent in-memory illustration omits production chunking/caching.
"""
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


def whole_tissue_domains(a):
    import squidpy as sq
    import cellcharter as cc
    a.obs['sample'] = a.obs['sample'].astype('category')
    # Spatial coordinates must be in a common unit; no edges between specimens.
    a.obsm['spatial'] = np.asarray(a.obsm['spatial_um'])
    sq.gr.spatial_neighbors(a, library_key='sample', coord_type='generic', delaunay=True)
    cc.gr.aggregate_neighbors(a, n_layers=3, use_rep='X_scVI', out_key='X_context')
    search = cc.tl.ClusterAutoK(n_clusters=(8, 14), max_runs=2,
                              model_params={'random_state': 12345})
    search.fit(a, use_rep='X_context')
    # K=12 is the study's reviewed solution, not automatically asserted optimal.
    model = cc.tl.Cluster(n_clusters=12, random_state=12345)
    model.fit(a, use_rep='X_context')
    a.obs['whole_tissue_domain'] = model.predict(a, use_rep='X_context')
    return a, search


def anchor_context(a, min_total_cells, min_tumour_cells, window_um=112):
    """112-µm square centred on reviewed epithelial tumour anchors.

    Three latent blocks: anchor, all neighbours, tumour neighbours; followed by
    lineage fractions, log1p cell count and tumour-anchor fraction.
    """
    xy = np.asarray(a.obsm['spatial_um'])
    latent = np.asarray(a.obsm['X_scVI'])
    sample = a.obs['sample'].astype(str).to_numpy()
    labels = a.obs['annotation'].astype(str).to_numpy()
    tumour = (a.obs['is_epithelial'].to_numpy(bool)
              & a.obs['in_tumour_roi'].to_numpy(bool))
    categories = np.unique(labels)
    indices, features = [], []
    for s in np.unique(sample):
        local = np.flatnonzero(sample == s)
        tree = cKDTree(xy[local])
        for centre in local[tumour[local]]:
            near = local[tree.query_ball_point(xy[centre], window_um / 2, p=np.inf)]
            tn = near[tumour[near]]
            if len(near) < min_total_cells or len(tn) < min_tumour_cells:
                continue
            fractions = [(labels[near] == c).mean() for c in categories]
            features.append(np.r_[latent[centre], latent[near].mean(0),
                                  latent[tn].mean(0), fractions,
                                  np.log1p(len(near)), len(tn) / len(near)])
            indices.append(centre)
    if not indices:
        raise ValueError('No eligible tumour anchors.')
    d, c = latent.shape[1], len(categories)
    block_weights = np.r_[np.full(3*d, 1/np.sqrt(d)),
                          np.full(c, 1.5/np.sqrt(c)), np.full(2, .5/np.sqrt(2))]
    return np.asarray(indices), np.asarray(features), block_weights


def fit_ecosystems(features, samples, block_weights, per_sample, n_pcs):
    import anndata as ad
    import cellcharter as cc
    rng = np.random.default_rng(12345)
    samples = np.asarray(samples)
    # Same cap per specimen, as in the original balanced fitting routine.
    chosen = np.concatenate([rng.choice(np.flatnonzero(samples == s),
                            min(per_sample, (samples == s).sum()), replace=False)
                            for s in np.unique(samples)])
    scaler = StandardScaler().fit(features[chosen])
    z = scaler.transform(features) * block_weights
    pca = PCA(n_components=min(n_pcs, z.shape[1], len(chosen)-1), random_state=12345)
    pca.fit(z[chosen])
    representation = pca.transform(z).astype('float32')
    all_anchors = ad.AnnData(np.empty((len(z), 0)))
    all_anchors.obsm['X_ecosystem'] = representation
    train = all_anchors[chosen].copy()
    search = cc.tl.ClusterAutoK(n_clusters=(3, 10), max_runs=2,
                              model_class=cc.tl.GaussianMixture,
                              model_params={'covariance_type': 'full', 'random_state': 12345})
    search.fit(train, use_rep='X_ecosystem')
    # Retain K=4 after stability and biological review; do not select using PFS.
    model = cc.tl.Cluster(n_clusters=4, covariance_type='full',
                         covariance_regularization=1e-4, random_state=12345)
    model.fit(train, use_rep='X_ecosystem')
    return model, all_anchors, search


def tile_soft_labels(anchor_table, tile_membership, raw_to_niche_order):
    """Average anchor posteriors within registered tiles with >=5 anchors.

    anchor_table: anchor_id, q0..q3 (raw GMM order).
    tile_membership: sample, tile_id, anchor_id; registration is supplied, not
    fabricated here. Overlapping tiles may contain the same anchor.
    raw_to_niche_order lists raw channel indices in the fixed N1..N4 order.
    """
    if sorted(raw_to_niche_order) != [0, 1, 2, 3]:
        raise ValueError('Supply the reviewed permutation of four GMM channels.')
    m = tile_membership.merge(anchor_table, on='anchor_id', validate='many_to_one')
    cols = [f'q{k}' for k in raw_to_niche_order]
    group = m.groupby(['sample', 'tile_id'])
    means = group[cols].mean()
    means = means.loc[group.size() >= 5]
    means.columns = ['N1', 'N2', 'N3', 'N4']
    return means.reset_index()
