"""Three epithelial programs -> continuous score -> pooled ecosystem ruler.

Gene lists below are the study's curated signatures, not patient measurements.
They are distinct from four-state phenotype annotations and N1–N4 ecosystems.
"""
import numpy as np
from scipy.special import softmax

SIGNATURES = {
    'Luminal': ['DHRS2','ERBB2','FOXA1','GATA3','KRT20','PPARG','PSCA',
                'RAB25','TBX3','TMEM97','UPK1A','UPK2','UPK3B','VGLL1'],
    'Intermediate': ['ANXA1','COL17A1','DSC2','EGFR','ITGB4','KRT14','KRT17','LAMB3'],
    'Basal_EMT': ['CALML3','CD44','DSG3','KRT16','KRT5','KRT6A','S100A2'],
}


def matched_program_scores(log_expression, genes):
    """Input is the chosen reference subset, normalized to 10,000 and log1p.

    Expression-matched controls: 25 linspace edges as in the source routine;
    <=50 controls per target gene, excluding the union of all signatures.
    """
    x = log_expression
    genes = np.asarray(genes)
    lookup = {g: i for i, g in enumerate(genes)}
    excluded = set(sum(SIGNATURES.values(), []))
    pool = np.array([i for i, g in enumerate(genes) if g not in excluded])
    mean = np.asarray(x.mean(0)).ravel()
    if not len(pool):
        raise ValueError('A non-signature background gene pool is required.')
    edges = np.linspace(mean[pool].min(), mean[pool].max()+1e-12, 25)
    bins = np.digitize(mean, edges)
    rng = np.random.default_rng(20260907)
    scores = []
    for signature in SIGNATURES.values():
        target = np.array([lookup[g] for g in signature if g in lookup])
        if len(target) < 3:
            raise ValueError('Insufficient observed signature genes.')
        controls = []
        for b in bins[target]:
            candidates = pool[bins[pool] == b]
            if len(candidates):
                controls.extend(rng.choice(candidates, min(50, len(candidates)), replace=False))
        controls = np.unique(controls).astype(int)
        if not len(controls):
            raise ValueError('No expression-matched controls.')
        scores.append(np.asarray(x[:, target].mean(1)).ravel()
                      - np.asarray(x[:, controls].mean(1)).ravel())
    return np.column_stack(scores)


def pooled_reference_coordinates(program_scores, niche_probabilities, eligible):
    """No patient-equal averaging: pool probability-weighted eligible cells.

    eligible identifies the reviewed epithelial/tumour reference population.
    Channels must already follow the frozen N1–N4 mapping; never relabel per slide.
    """
    d = softmax(program_scores, axis=1) @ np.array([0., .5, 1.])
    q = np.asarray(niche_probabilities, dtype=float)
    use = np.asarray(eligible, bool) & np.isfinite(d) & np.isfinite(q).all(1)
    if q.shape != (len(d), 4) or not use.any():
        raise ValueError('Require eligible reference cells and four niche channels.')
    if (q[use] < 0).any() or not np.allclose(q[use].sum(1), 1):
        raise ValueError('Expected posterior probability vectors.')
    mass = q[use].sum(0)
    if (mass <= 0).any():
        raise ValueError('Each niche requires positive reference mass.')
    tau = q[use].T @ d[use] / mass
    return d, tau


def project_patch_scores(patch_probabilities, tau):
    # Relative differentiation position; not elapsed time or a lineage velocity.
    return np.asarray(patch_probabilities) @ np.asarray(tau)
