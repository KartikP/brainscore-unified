"""Topographic-organization metric — folding TDANN / TopoLM-style spatial
structure into Brain-Score as a *type of metric*.

Conventional Brain-Score metrics ask "do model unit responses predict neural
responses?" A topographic metric asks a different question: "is the model's
spatial layout of units organized like cortex?" Topographic models (TDANN,
TopoLM) place units on a 2-D tissue sheet under a spatial-smoothness objective,
so nearby units develop correlated tuning — exactly as nearby cortical sites do.

The descriptor that captures this is the **correlation-vs-distance profile**
r(d): the mean pairwise response correlation between units as a function of
their distance on the sheet. Cortex (and a topographic model) shows r high at
small d, decaying with d; a non-topographic model with arbitrary unit positions
shows a flat profile. From it we derive:

  * :func:`correlation_distance_profile` — r(d) itself.
  * :func:`spatial_smoothness` — a scalar: ``-spearman(distance, correlation)``,
    positive when nearby units are more correlated (topographic), ~0 otherwise.
  * :class:`TopographicMetric` — brain correspondence: correlate the model's
    r(d) profile with the brain's r(d) profile (voxels on the cortical surface).

fMRI is a natural target: voxels carry surface coordinates, so the brain r(d)
profile is directly computable. A model only has a meaningful profile if its
units carry tissue positions (``tissue_x`` / ``tissue_y`` neuroid coords); for a
TDANN these come from the model, for a conv feature map from the spatial grid.
"""
from typing import Optional, Tuple

import numpy as np

from brainscore_core.metrics import Metric, Score


def _pairwise_upper(n: int) -> Tuple[np.ndarray, np.ndarray]:
    iu = np.triu_indices(n, k=1)
    return iu[0], iu[1]


def correlation_distance_profile(responses: np.ndarray, positions: np.ndarray, *,
                                 n_bins: int = 15, normalize_distance: bool = True,
                                 max_pairs: int = 2_000_000, seed: int = 0):
    """Mean pairwise response correlation as a function of unit distance.

    :param responses: ``(n_stimuli, n_units)`` response matrix.
    :param positions: ``(n_units, n_dims)`` tissue/surface coordinates.
    :param n_bins: number of distance bins.
    :param normalize_distance: scale distances to ``[0, 1]`` (by max pairwise
        distance) so model and brain profiles share an x-axis.
    :param max_pairs: cap the number of unit pairs (subsample for large layers).
    :returns: ``(bin_centers, mean_corr, counts)``; empty bins are ``nan``.
    """
    responses = np.asarray(responses, dtype=float)
    positions = np.asarray(positions, dtype=float)
    n_units = responses.shape[1]
    # per-unit z-score across stimuli so correlation == normalized dot product
    r = responses - responses.mean(axis=0, keepdims=True)
    std = r.std(axis=0, keepdims=True)
    std[std < 1e-9] = 1e-9
    rz = r / std

    i, j = _pairwise_upper(n_units)
    if i.size > max_pairs:
        rng = np.random.RandomState(seed)
        sel = rng.choice(i.size, size=max_pairs, replace=False)
        i, j = i[sel], j[sel]

    corr = (rz[:, i] * rz[:, j]).mean(axis=0)               # (n_pairs,)
    dist = np.linalg.norm(positions[i] - positions[j], axis=1)
    if normalize_distance and dist.max() > 0:
        dist = dist / dist.max()

    hi = 1.0 if normalize_distance else dist.max()
    edges = np.linspace(0, hi, n_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    mean_corr = np.full(n_bins, np.nan)
    counts = np.zeros(n_bins, dtype=int)
    binidx = np.clip(np.digitize(dist, edges) - 1, 0, n_bins - 1)
    for b in range(n_bins):
        m = binidx == b
        counts[b] = int(m.sum())
        if counts[b]:
            mean_corr[b] = float(corr[m].mean())
    return centers, mean_corr, counts


def spatial_smoothness(responses: np.ndarray, positions: np.ndarray, *,
                       max_pairs: int = 2_000_000, seed: int = 0) -> float:
    """Scalar topographic index in ``[-1, 1]``.

    Defined as ``-spearman(distance, correlation)`` over unit pairs: positive
    when nearby units are more correlated than distant ones (topographic),
    ~0 for arbitrary positions, negative for anti-topographic layouts.
    """
    from scipy.stats import spearmanr
    responses = np.asarray(responses, dtype=float)
    positions = np.asarray(positions, dtype=float)
    n_units = responses.shape[1]
    r = responses - responses.mean(axis=0, keepdims=True)
    std = r.std(axis=0, keepdims=True); std[std < 1e-9] = 1e-9
    rz = r / std
    i, j = _pairwise_upper(n_units)
    if i.size > max_pairs:
        rng = np.random.RandomState(seed)
        sel = rng.choice(i.size, size=max_pairs, replace=False)
        i, j = i[sel], j[sel]
    corr = (rz[:, i] * rz[:, j]).mean(axis=0)
    dist = np.linalg.norm(positions[i] - positions[j], axis=1)
    rho, _ = spearmanr(dist, corr)
    return float(-rho) if np.isfinite(rho) else 0.0


def topographic_alignment(model_responses, model_positions,
                          brain_responses, brain_positions, *,
                          n_bins: int = 15) -> Tuple[float, dict]:
    """Pearson correlation between the model's r(d) profile and the brain's.

    Both profiles are computed on the normalized ``[0, 1]`` distance axis so the
    comparison is scale-free. Returns ``(score, detail)`` where detail carries
    both profiles for plotting.
    """
    mc_x, mc_y, _ = correlation_distance_profile(
        model_responses, model_positions, n_bins=n_bins)
    bc_x, bc_y, _ = correlation_distance_profile(
        brain_responses, brain_positions, n_bins=n_bins)
    mask = np.isfinite(mc_y) & np.isfinite(bc_y)
    if mask.sum() < 3:
        return float('nan'), {'model_profile': (mc_x, mc_y),
                              'brain_profile': (bc_x, bc_y)}
    score = float(np.corrcoef(mc_y[mask], bc_y[mask])[0, 1])
    return score, {'model_profile': (mc_x, mc_y), 'brain_profile': (bc_x, bc_y),
                   'model_smoothness': spatial_smoothness(model_responses, model_positions),
                   'brain_smoothness': spatial_smoothness(brain_responses, brain_positions)}


def _extract(assembly):
    """Pull (responses, positions) from a (presentation, neuroid) assembly that
    carries ``tissue_x``/``tissue_y`` (or ``x``/``y``/``z``) neuroid coords."""
    responses = np.asarray(assembly.transpose('presentation', 'neuroid').values)
    coords = {name for name, _, _ in __import__(
        'brainscore_core.supported_data_standards.brainio.assemblies',
        fromlist=['walk_coords']).walk_coords(assembly)}
    if 'tissue_x' in coords and 'tissue_y' in coords:
        pos = np.stack([assembly['tissue_x'].values, assembly['tissue_y'].values], -1)
    elif {'x', 'y', 'z'} <= coords:
        pos = np.stack([assembly['x'].values, assembly['y'].values,
                        assembly['z'].values], -1)
    elif {'x', 'y'} <= coords:
        pos = np.stack([assembly['x'].values, assembly['y'].values], -1)
    else:
        raise ValueError("assembly needs tissue_x/tissue_y or x/y[/z] neuroid "
                         f"coords for a topographic profile; has {sorted(coords)}.")
    return responses, np.asarray(pos, dtype=float)


def selectivity_maps(responses: np.ndarray, labels) -> Tuple[list, np.ndarray]:
    """Per-unit one-vs-rest selectivity (Cohen's d) for each category.

    :param responses: ``(n_stimuli, n_units)``.
    :param labels: length-``n_stimuli`` category label per stimulus.
    :returns: ``(categories, selectivity)`` where selectivity is
        ``(n_categories, n_units)`` — d = (mean_in - mean_out) / pooled_sd.
    """
    responses = np.asarray(responses, dtype=float)
    labels = np.asarray(labels)
    cats = sorted(set(labels.tolist()))
    out = []
    for c in cats:
        m = labels == c
        a, b = responses[m], responses[~m]
        if a.shape[0] < 2 or b.shape[0] < 2:
            out.append(np.zeros(responses.shape[1]))
            continue
        pooled = np.sqrt((a.var(0, ddof=1) + b.var(0, ddof=1)) / 2)
        pooled[pooled < 1e-9] = 1e-9
        out.append((a.mean(0) - b.mean(0)) / pooled)
    return cats, np.stack(out)


def selectivity_centroids(selectivity: np.ndarray, positions: np.ndarray,
                          top_k_frac: float = 0.1) -> np.ndarray:
    """``(n_categories, n_dims)`` — centroid of each category's top-k most
    selective units' tissue positions (the localized selective cluster)."""
    n_units = selectivity.shape[1]
    k = max(1, int(round(n_units * top_k_frac)))
    cents = [positions[np.argsort(selectivity[c])[-k:]].mean(0)
             for c in range(selectivity.shape[0])]
    return np.stack(cents)


def selectivity_topographic_alignment(m_resp, m_pos, m_lab, b_resp, b_pos, b_lab, *,
                                      top_k_frac: float = 0.1) -> Tuple[float, dict]:
    """Spearman correlation of the off-diagonal category-centroid distance
    matrices on the model sheet vs the cortex — "is FFA-near-EBA-far-from-V1"
    layout preserved." Uses the categories common to both sides (needs >=3 for
    a non-trivial RSA). Permutation-sensitive over units."""
    from scipy.spatial.distance import pdist
    from scipy.stats import spearmanr
    m_cats, m_sel = selectivity_maps(m_resp, m_lab)
    b_cats, b_sel = selectivity_maps(b_resp, b_lab)
    common = [c for c in m_cats if c in set(b_cats)]
    detail = {'categories': common}
    if len(common) < 3:
        return float('nan'), detail
    mi = [m_cats.index(c) for c in common]
    bi = [b_cats.index(c) for c in common]
    m_cent = selectivity_centroids(m_sel[mi], np.asarray(m_pos, float), top_k_frac)
    b_cent = selectivity_centroids(b_sel[bi], np.asarray(b_pos, float), top_k_frac)
    rho, _ = spearmanr(pdist(m_cent), pdist(b_cent))
    detail.update({'model_centroids': m_cent, 'brain_centroids': b_cent})
    return (float(rho) if np.isfinite(rho) else 0.0), detail


class SelectivityTopographicMetric(Metric):
    """Second topographic axis: does the model's spatial arrangement of
    *category selectivity* match cortex's category-selective layout (FFA/PPA/
    EBA/VWFA adjacency)? Distinct from :class:`TopographicMetric`'s
    across-stimulus response-correlation axis — a model can be topographic on
    one and flat on the other (e.g. Topo-Omni). Both assemblies need per-unit
    tissue positions and a per-presentation category coord (``label_coord``)."""

    def __init__(self, label_coord: str = 'category', top_k_frac: float = 0.1):
        self.label_coord = label_coord
        self.top_k_frac = top_k_frac

    def __call__(self, model_assembly, brain_assembly) -> Score:
        mr, mp = _extract(model_assembly)
        br, bp = _extract(brain_assembly)
        ml = np.asarray(model_assembly[self.label_coord].values)
        bl = np.asarray(brain_assembly[self.label_coord].values)
        value, detail = selectivity_topographic_alignment(
            mr, mp, ml, br, bp, bl, top_k_frac=self.top_k_frac)
        score = Score(value)
        score.attrs['categories'] = detail.get('categories')
        score.attrs['model_centroids'] = detail.get('model_centroids')
        score.attrs['brain_centroids'] = detail.get('brain_centroids')
        return score


class TopographicMetric(Metric):
    """Score a model's spatial organization against cortical topography.

    ``__call__(model_assembly, brain_assembly)`` -> :class:`Score` equal to the
    Pearson correlation of the two correlation-vs-distance profiles. Both
    assemblies must carry per-unit positions (see :func:`_extract`). The score's
    ``attrs`` hold both profiles and both smoothness indices for plotting.
    """

    def __init__(self, n_bins: int = 15):
        self.n_bins = n_bins

    def __call__(self, model_assembly, brain_assembly) -> Score:
        mr, mp = _extract(model_assembly)
        br, bp = _extract(brain_assembly)
        value, detail = topographic_alignment(mr, mp, br, bp, n_bins=self.n_bins)
        score = Score(value)
        score.attrs['model_smoothness'] = detail.get('model_smoothness')
        score.attrs['brain_smoothness'] = detail.get('brain_smoothness')
        score.attrs['model_profile'] = detail['model_profile']
        score.attrs['brain_profile'] = detail['brain_profile']
        return score
