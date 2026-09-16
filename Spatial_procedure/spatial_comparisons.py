
from itertools import product
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.stats import wilcoxon, ttest_rel
from statsmodels.stats.multitest import multipletests


def nearest_muscle_distance(cells):
    """Coordinates already converted to µm using specimen-specific metadata.

    Input: sample, x_um, y_um, is_smooth_muscle. No cross-specimen neighbours.
    """
    distance = np.full(len(cells), np.nan)
    for sample in cells['sample'].unique():
        rows = np.flatnonzero(cells['sample'].to_numpy() == sample)
        sub = cells.iloc[rows]
        muscle = sub.loc[sub.is_smooth_muscle, ['x_um', 'y_um']].to_numpy()
        if len(muscle):
            distance[rows] = cKDTree(muscle).query(sub[['x_um','y_um']].to_numpy())[0]
    return distance


def exact_sign_flip(differences):
    d = np.asarray(differences, float)
    d = d[np.isfinite(d)]
    if not len(d):
        return np.nan
    # The study uses eight specimens; exhaustive 2**n enumeration is feasible.
    signs = np.asarray(list(product([-1., 1.], repeat=len(d))))
    return float((np.abs(signs @ d / len(d)) >= abs(d.mean())-1e-12).mean())


def paired_region_test(region_table, method='sign_flip'):
    """Input one row per sample/region/metric, with value and core/front regions.

    Regional summaries are inputs so pathology-defined fronts are never replaced
    silently by geometric edges. BH correction is across supplied metrics.
    """
    rows = []
    for metric, g in region_table.groupby('metric'):
        paired = g.pivot(index='sample', columns='region', values='value')
        paired = paired.reindex(columns=['core','front']).dropna()
        delta = (paired.front-paired.core).to_numpy()
        if method == 'sign_flip':
            p = exact_sign_flip(delta)
        elif method == 'paired_t':
            p = ttest_rel(paired.front, paired.core).pvalue if len(paired)>1 else np.nan
        else:
            raise ValueError('Use sign_flip or paired_t.')
        rows.append({'metric': metric, 'n': len(delta),
                     'mean_front_minus_core': delta.mean() if len(delta) else np.nan,
                     'p': p})
    result = pd.DataFrame(rows)
    result['fdr'] = np.nan
    valid = result.p.notna()
    if valid.any():
        result.loc[valid, 'fdr'] = multipletests(result.loc[valid,'p'], method='fdr_bh')[1]
    return result


def adc_threshold_sensitivity(epithelial_cells):
    """Eligible epithelial rows; gene columns contain log-normalized expression.

    Distance is nearest smooth-muscle distance in µm. Fixed distal 80–160 µm;
    main proximal threshold 32 µm, alternative 24/40/48 µm. Tests use specimens,
    not individual cells, as replicates. BH across three genes WITHIN threshold.
    """
    rows = []
    for threshold in [24, 32, 40, 48]:
        band = []
        for gene in ['TACSTD2','NECTIN4','ERBB2']:
            differences = []
            for _, g in epithelial_cells.groupby('sample'):
                near = g.loc[g.distance_um.between(0, threshold), gene].mean()
                far = g.loc[g.distance_um.between(80, 160), gene].mean()
                if np.isfinite(near-far):
                    differences.append(near-far)
            d = np.asarray(differences)
            p = (np.nan if not len(d) else 1. if np.all(d == 0)
                 else wilcoxon(d, alternative='two-sided').pvalue)
            band.append({'proximal_um': threshold, 'gene': gene, 'n': len(d),
                         'n_lower': int((d<0).sum()),
                         'median_proximal_minus_distal': np.median(d) if len(d) else np.nan,
                         'p': p})
        table = pd.DataFrame(band)
        valid = table.p.notna()
        table['fdr'] = np.nan
        if valid.any():
            table.loc[valid,'fdr'] = multipletests(table.loc[valid,'p'],method='fdr_bh')[1]
        rows.append(table)
    return pd.concat(rows, ignore_index=True)


def public_tumour_depth(coords_um, malignant):
    """ONE specimen: distance of malignant spots to the nearest non-malignant spot.

    NMIBC malignant masks are supplied from CNV-supported calls. MIBC annotations
    follow their source dataset. This is a computational boundary, not pathology.
    """
    xy, malignant = np.asarray(coords_um), np.asarray(malignant, bool)
    if malignant.all() or not malignant.any():
        raise ValueError('Need both malignant and non-malignant spots.')
    d = np.full(len(xy), np.nan)
    d[malignant] = cKDTree(xy[~malignant]).query(xy[malignant])[0]
    return d
