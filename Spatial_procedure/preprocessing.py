
import numpy as np


def preprocess(adata):
    import scanpy as sc
    a = adata.copy()  # Input X must contain non-negative raw segmented-cell counts.
    sc.pp.filter_cells(a, min_counts=5)
    sc.pp.filter_cells(a, min_genes=3)
    sc.pp.filter_genes(a, min_cells=5)
    a = a[:, ~a.var_names.isin(['IGKC', 'IGHG1'])].copy()
    a.layers['counts'] = a.X.copy()
    sc.pp.normalize_total(a, target_sum=1e4)
    sc.pp.log1p(a)
    return a


def integrate_and_annotate(a, reviewed_labels, max_epochs):
    """Labels are marker-reviewed inputs, not inferred from outcomes.

    reviewed_labels: length n_cells; missing values indicate unreviewed cells.
    X remains log-normalized; scVI reads the counts layer only.
    """
    import pandas as pd
    import scvi
    from sklearn.linear_model import LogisticRegression
    scvi.settings.seed = 12345
    scvi.model.SCVI.setup_anndata(a, layer='counts', batch_key='sample')
    model = scvi.model.SCVI(a, n_latent=10)
    model.train(max_epochs=max_epochs)
    a.obsm['X_scVI'] = model.get_latent_representation()
    labels = np.asarray(reviewed_labels, dtype=object)
    known = pd.notna(labels)
    classifier = LogisticRegression(max_iter=2000)
    classifier.fit(a.obsm['X_scVI'][known], labels[known].astype(str))
    p = classifier.predict_proba(a.obsm['X_scVI'])
    predicted = classifier.classes_[p.argmax(1)].astype(object)
    predicted[p.max(1) < 0.30] = 'Uncertain'
    predicted[known] = labels[known]  # Retain reviewed annotations.
    a.obs['annotation'] = predicted
    a.obs['annotation_confidence'] = p.max(1)
    return a
