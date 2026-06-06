"""Layer mapping with unit selection — a productionized Brain-Score tool.

Given per-layer model features and a brain target, this finds the brain-optimal
layer and builds a :class:`CompositeSelector` — top-K units drawn from the top-N
layers — which is the scientifically principled mapping (a functional unit
population spanning depth, not a single hand-picked layer). It scores four
mapping approaches head-to-head so they can be compared:

  1. standard layer mapping      — a single best full layer
  2. unit selection in a layer   — top-K units of the best layer
  3. multiple full layers        — concatenation of the top-N full layers
  4. CompositeSelector           — top-K units from each of the top-N layers

Selection is non-circular by construction: a **functional-localization split**
separates the data into a LOCALIZER set (used to rank layers and identify
selective units) and a held-out TEST set (used to fit the readout and score).
No stimulus used to choose a unit is used to score it — the encoding analogue of
an fMRI localizer run, and the same logic as ``FunctionalSelection``.

Scoring is decoupled from extraction (``explore_layer_mapping`` takes features +
target), so it is testable offline; :func:`sweep_model` extracts then maps.
"""
import dataclasses
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from brainscore_core.model_interface import CompositeSelector


def _per_voxel_pearson(Y_true: np.ndarray, Y_pred: np.ndarray) -> np.ndarray:
    Yc = Y_true - Y_true.mean(0, keepdims=True)
    Pc = Y_pred - Y_pred.mean(0, keepdims=True)
    num = (Yc * Pc).sum(0)
    den = np.sqrt((Yc ** 2).sum(0) * (Pc ** 2).sum(0))
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.where(den > 0, num / den, np.nan)


def per_voxel_train_test(X_tr, Y_tr, X_te, Y_te,
                         alpha: Union[float, Sequence[float]] = 1.0) -> np.ndarray:
    """Fit ridge on the localizer rows, score per-voxel Pearson r on the test rows.

    No feature scaler: Ridge auto-centers via its intercept; an explicit
    StandardScaler over-rescales heterogeneous-variance features and depresses
    the fit (established in the Lahner scoring work).

    :param alpha: a single ridge penalty (``Ridge``), OR a sequence of
        candidate penalties, in which case ``RidgeCV(alpha_per_target=True)``
        selects the best penalty *per voxel* by efficient leave-one-out — the
        scientifically correct choice when feature sets differ in size, since a
        fixed penalty otherwise flatters smaller feature sets (the layer-mapping
        feature-count confound).
    """
    X_tr = np.asarray(X_tr, np.float64); Y_tr = np.asarray(Y_tr, np.float64)
    X_te = np.asarray(X_te, np.float64); Y_te = np.asarray(Y_te, np.float64)
    if np.ndim(alpha) == 0:
        from sklearn.linear_model import Ridge
        reg = Ridge(alpha=float(alpha)).fit(X_tr, Y_tr)
    else:
        from sklearn.linear_model import RidgeCV
        reg = RidgeCV(alphas=np.asarray(alpha, np.float64),
                      alpha_per_target=True).fit(X_tr, Y_tr)
    return _per_voxel_pearson(Y_te, reg.predict(X_te))


def effective_dimensionality(X: np.ndarray) -> float:
    """Participation ratio of a feature matrix — the *effective* number of
    dimensions its variance occupies.

    ``PR = (Σλ)² / Σλ²`` over the covariance eigenvalues ``λ``. PR = 1 when all
    variance is in one direction; PR = p when variance is spread evenly over all
    ``p`` features. A low PR relative to the feature count means the signal is
    low-dimensional (and therefore recoverable from a small random subset of
    neurons) — the structural explanation for why random unit-subsets score high
    on distributed naturalistic signal.
    """
    X = np.asarray(X, np.float64)
    Xc = X - X.mean(0, keepdims=True)
    s = np.linalg.svd(Xc, compute_uv=False)
    lam = s ** 2
    denom = float((lam ** 2).sum())
    return float((lam.sum() ** 2) / denom) if denom > 0 else 0.0


def normalize_by_ceiling(r: np.ndarray, ceiling: np.ndarray,
                         min_ceiling: float = 0.1) -> np.ndarray:
    """Divide per-voxel r by each voxel's noise ceiling (split-half reliability).

    Puts the score on a 'fraction of explainable signal' scale. Voxels whose
    ceiling is below ``min_ceiling`` are dropped (set to NaN) because their
    normalized score is dominated by measurement noise — standard practice in
    encoding-model evaluation. Aggregate with ``nanmedian`` afterwards.
    """
    r = np.asarray(r, np.float64); c = np.asarray(ceiling, np.float64)
    out = np.full(r.shape, np.nan)
    ok = c > min_ceiling
    out[ok] = r[ok] / c[ok]
    return out


def per_voxel_cv_ridge(X, Y, alpha: float = 1.0, n_splits: int = 5,
                       seed: int = 0) -> np.ndarray:
    """K-fold ridge per voxel (no scaler) — kept as a utility for whole-set CV."""
    from sklearn.model_selection import KFold
    from sklearn.linear_model import Ridge
    X = np.asarray(X, np.float64); Y = np.asarray(Y, np.float64)
    preds = np.full_like(Y, np.nan)
    for tr, te in KFold(n_splits=n_splits, shuffle=True, random_state=seed).split(np.arange(X.shape[0])):
        preds[te] = Ridge(alpha=alpha).fit(X[tr], Y[tr]).predict(X[te])
    return _per_voxel_pearson(Y, preds)


@dataclasses.dataclass
class LayerMappingResult:
    """Per-layer test scores + per-unit localizer predictivity; selectors on demand.

    ``per_layer_r`` and the approach scores are computed on the held-out TEST
    split; ``unit_predictivity`` (used to pick units) is computed on the
    LOCALIZER split — so a unit is never selected and scored on the same data.
    """
    layer_order: List[str]
    per_layer_r: List[float]
    unit_predictivity: np.ndarray   # (n_layers, n_units): per-unit |corr| on the LOCALIZER
    localizer_idx: np.ndarray
    test_idx: np.ndarray
    alpha: float = 1.0

    @property
    def best_index(self) -> int:
        return int(np.nanargmax(self.per_layer_r))

    @property
    def best_layer(self) -> str:
        return self.layer_order[self.best_index]

    @property
    def best_r(self) -> float:
        return float(self.per_layer_r[self.best_index])

    def top_layers(self, n: int = 3) -> List[str]:
        order = np.argsort(np.nan_to_num(self.per_layer_r, nan=-np.inf))[::-1]
        return [self.layer_order[int(i)] for i in order[:n]]

    def top_units(self, layer: str, k: int) -> List[int]:
        li = self.layer_order.index(layer)
        return sorted(int(u) for u in np.argsort(self.unit_predictivity[li])[::-1][:k])

    def top_units_pooled(self, layers: Sequence[str], k: int) -> List[Tuple[str, int]]:
        """Top-``k`` units ranked by localizer predictivity across ``layers``
        pooled together — returns ``(layer, unit)`` pairs. The across-layers
        analogue of :meth:`top_units`, used for the budget-matched comparison
        of within-one-layer vs across-layers selection at equal feature count.
        """
        cand = []
        for layer in layers:
            li = self.layer_order.index(layer)
            for u in range(self.unit_predictivity.shape[1]):
                cand.append((float(self.unit_predictivity[li, u]), layer, int(u)))
        cand.sort(key=lambda t: -t[0])
        return [(layer, u) for _, layer, u in cand[:k]]

    def composite_selector(self, n_layers: int = 3, k: int = 100) -> CompositeSelector:
        """top-``k`` localizer-selected units from each of the top-``n_layers``."""
        return CompositeSelector(layers=tuple(
            (layer, tuple(self.top_units(layer, k)))
            for layer in self.top_layers(n_layers)))


def _split(n: int, localizer_frac: float, seed: int):
    rng = np.random.RandomState(seed)
    perm = rng.permutation(n)
    n_loc = max(1, int(round(localizer_frac * n)))
    return np.sort(perm[:n_loc]), np.sort(perm[n_loc:])


def explore_layer_mapping(features_by_layer: Dict[str, np.ndarray], target: np.ndarray,
                          localizer_frac: float = 0.5, alpha: float = 1.0,
                          seed: int = 0) -> LayerMappingResult:
    """Rank layers (test score) and rank units (localizer predictivity).

    :param features_by_layer: ``{layer_path: (n_stimuli, n_units)}`` (same row
        order as ``target``).
    :param target: ``(n_stimuli, n_voxels)`` brain responses.
    :param localizer_frac: fraction of stimuli used as the localizer (the rest
        is held out as test).
    """
    Y = np.asarray(target, np.float64)
    L, T = _split(Y.shape[0], localizer_frac, seed)
    Y_loc = Y[L]
    Yz = (Y_loc - Y_loc.mean(0)) / (Y_loc.std(0) + 1e-8)
    layer_order = list(features_by_layer.keys())
    n_units = next(iter(features_by_layer.values())).shape[1]
    per_layer_r: List[float] = []
    unit_pred = np.zeros((len(layer_order), n_units), dtype=np.float64)
    for li, layer in enumerate(layer_order):
        X = np.asarray(features_by_layer[layer], np.float64)
        # layer score: readout fit on localizer, evaluated on held-out test
        per_layer_r.append(float(np.nanmedian(
            per_voxel_train_test(X[L], Y[L], X[T], Y[T], alpha))))
        # unit predictivity on the LOCALIZER only (selection sees no test data)
        Xl = X[L]
        Xz = (Xl - Xl.mean(0)) / (Xl.std(0) + 1e-8)
        unit_pred[li] = np.abs((Xz.T @ Yz) / Xl.shape[0]).max(1)
    return LayerMappingResult(layer_order, per_layer_r, unit_pred, L, T, alpha)


def score_approaches(features_by_layer: Dict[str, np.ndarray], target: np.ndarray,
                     result: LayerMappingResult, top_n_layers: int = 3,
                     top_k: int = 100, alpha: float = 1.0,
                     n_null_seeds: int = 5) -> List[dict]:
    """Score the four mapping approaches on the held-out TEST split, each paired
    with a matched **random-selection null**.

    Unit/layer choices come from the localizer (via ``result``); each approach's
    readout is fit on the localizer and scored on the test split. For every
    approach a null is computed with the SAME feature budget but the selection
    randomized (random units / random layers), averaged over ``n_null_seeds``
    seeds and scored identically — so a selection only "counts" if it beats the
    random draw of the same size. Each result carries ``random_null`` (mean) and
    ``random_null_sd``.
    """
    Y = np.asarray(target, np.float64)
    L, T = result.localizer_idx, result.test_idx
    layer_order = result.layer_order
    n_units = features_by_layer[result.best_layer].shape[1]

    def score(X_full):
        X = np.asarray(X_full, np.float64)
        return float(np.nanmedian(per_voxel_train_test(X[L], Y[L], X[T], Y[T], alpha)))

    def null_over_seeds(make_X):
        vals = [score(make_X(np.random.RandomState(s))) for s in range(n_null_seeds)]
        return float(np.mean(vals)), float(np.std(vals))

    best = result.best_layer
    top = result.top_layers(top_n_layers)

    # selected (signal) scores
    standard = result.best_r
    unit_within = score(features_by_layer[best][:, result.top_units(best, top_k)])
    multi_full = score(np.concatenate([features_by_layer[l] for l in top], axis=1))
    composite = score(np.concatenate(
        [features_by_layer[l][:, result.top_units(l, top_k)] for l in top], axis=1))

    # matched random-selection nulls (same feature budget, selection randomized)
    rand_units = lambda rng, layer, k: features_by_layer[layer][:, rng.choice(n_units, k, replace=False)]
    std_null = null_over_seeds(
        lambda rng: features_by_layer[layer_order[rng.randint(len(layer_order))]])   # random full layer
    unit_null = null_over_seeds(
        lambda rng: rand_units(rng, best, top_k))                                    # random K units, best layer
    multi_null = null_over_seeds(lambda rng: np.concatenate(
        [features_by_layer[layer_order[i]] for i in
         rng.choice(len(layer_order), len(top), replace=False)], axis=1))            # random N full layers
    comp_null = null_over_seeds(lambda rng: np.concatenate(
        [rand_units(rng, l, top_k) for l in top], axis=1))                           # random K units / top-N layers

    def row(name, detail, nf, r, null):
        return {'name': name, 'detail': detail, 'n_features': nf, 'r': round(r, 4),
                'random_null': round(null[0], 4), 'random_null_sd': round(null[1], 4)}

    return [
        row('standard layer mapping', f'single best full layer ({best})', n_units, standard, std_null),
        row('unit selection within a layer', f'top-{top_k} units of {best}', top_k, unit_within, unit_null),
        row('multiple full layers', f'concat of {top}', n_units * len(top), multi_full, multi_null),
        row('CompositeSelector', f'top-{top_k} units from each of {top}',
            top_k * len(top), composite, comp_null),
    ]


def extract_features_by_layer(wrapper, stimulus_set, layers: List[str],
                              time_reduce: str = 'mean') -> Dict[str, np.ndarray]:
    """Extract ``(n_stimuli, n_units)`` per layer from an activations wrapper.

    One ``wrapper(stimulus_set, layers=layers)`` pass (layers share the forward);
    any ``time_bin`` axis is reduced by ``time_reduce`` and the neuroid axis is
    split per layer. Returns ``{layer: (n_stimuli, n_units)}`` plus
    ``'_stimulus_id'`` (the presentation order).
    """
    asm = wrapper(stimulus_set, layers=layers)
    if 'time_bin' in asm.dims and time_reduce == 'mean':
        asm = asm.mean('time_bin')
    asm = asm.transpose('presentation', 'neuroid')
    layer_coord = np.asarray(asm['layer'].values).ravel()
    vals = np.asarray(asm.values)
    out: Dict[str, np.ndarray] = {
        layer: vals[:, np.where(layer_coord == layer)[0]] for layer in layers}
    out['_stimulus_id'] = np.array([str(s) for s in asm['stimulus_id'].values])
    return out


def _gather_units(features_by_layer: Dict[str, np.ndarray],
                  pairs: Sequence[Tuple[str, int]]) -> np.ndarray:
    """Stack the named ``(layer, unit)`` columns into one ``(n_stim, len)`` matrix."""
    return np.column_stack([features_by_layer[layer][:, u] for layer, u in pairs])


def score_budget_curve(features_by_layer: Dict[str, np.ndarray], target: np.ndarray,
                       result: LayerMappingResult, budgets: Sequence[int],
                       top_n_layers: int = 3,
                       alpha_grid: Sequence[float] = (1., 10., 100., 1000., 10000.),
                       n_null_seeds: int = 5,
                       noise_ceiling: Optional[np.ndarray] = None) -> dict:
    """The scientifically valid strategy comparison: a *budget-matched* curve.

    For each feature budget ``K`` it scores two selection strategies at exactly
    ``K`` features each — **within one layer** (top-K units of the best layer)
    and **pooled across the top-N layers** (top-K units ranked over all of them)
    — so the comparison isolates the *selection strategy* from the feature count.
    Every fit uses ``RidgeCV(alpha_per_target=True)`` (per-voxel penalty tuning),
    removing the fixed-α confound, and divides by the noise ceiling when given.
    Each strategy is paired with a matched random-selection null at the same K.

    Two full-budget **reference points** (the whole best layer; all top-N layers
    concatenated) are scored the same way, so the curve shows whether a compact
    selection matches the full layer (efficiency) and which option peaks highest.

    :returns: dict with ``within_layer`` and ``pooled_layers`` (each a list of
        ``{k, r, random_null, random_null_sd}``), ``whole_layer_r``,
        ``several_layers_r``, plus metadata. Scores are ceiling-normalized iff
        ``noise_ceiling`` is provided (``normalized`` flag records which).
    """
    Y = np.asarray(target, np.float64)
    L, T = result.localizer_idx, result.test_idx
    best = result.best_layer
    top = result.top_layers(top_n_layers)
    n_units = features_by_layer[best].shape[1]
    all_pairs = [(layer, u) for layer in top for u in range(n_units)]

    def score(cols) -> float:
        cols = np.asarray(cols, np.float64)
        r = per_voxel_train_test(cols[L], Y[L], cols[T], Y[T], alpha=alpha_grid)
        if noise_ceiling is not None:
            r = normalize_by_ceiling(r, noise_ceiling)
        return float(np.nanmedian(r))

    def null_over_seeds(make_cols):
        vals = [score(make_cols(np.random.RandomState(s))) for s in range(n_null_seeds)]
        return round(float(np.mean(vals)), 4), round(float(np.std(vals)), 4)

    within, pooled = [], []
    for K in budgets:
        K = int(K)
        if K <= n_units:
            wr = score(features_by_layer[best][:, result.top_units(best, K)])
            wn = null_over_seeds(
                lambda rng, K=K: features_by_layer[best][:, rng.choice(n_units, K, replace=False)])
            within.append({'k': K, 'r': round(wr, 4),
                           'random_null': wn[0], 'random_null_sd': wn[1]})
        if K <= len(all_pairs):
            pr = score(_gather_units(features_by_layer, result.top_units_pooled(top, K)))
            pn = null_over_seeds(lambda rng, K=K: _gather_units(
                features_by_layer,
                [all_pairs[i] for i in rng.choice(len(all_pairs), K, replace=False)]))
            pooled.append({'k': K, 'r': round(pr, 4),
                           'random_null': pn[0], 'random_null_sd': pn[1]})

    return {
        'budgets': [int(b) for b in budgets],
        'within_layer': within,        # top-K units of the best layer (+ random null)
        'pooled_layers': pooled,       # top-K units across the top-N layers (+ random null)
        'whole_layer_r': round(score(features_by_layer[best]), 4),
        'several_layers_r': round(
            score(np.concatenate([features_by_layer[l] for l in top], axis=1)), 4),
        'best_layer': best, 'top_layers': list(top),
        'alpha_grid': list(alpha_grid), 'normalized': noise_ceiling is not None,
    }


def sweep_model(wrapper, stimulus_set, target, target_stimulus_ids, layers,
                localizer_frac: float = 0.5, alpha: float = 1.0,
                top_n_layers: int = 3, top_k: int = 100):
    """End-to-end: extract per-layer features, align to the target, then run
    :func:`explore_layer_mapping` + :func:`score_approaches`.

    Returns ``(result, approaches, composite_selector)``.
    """
    feats = extract_features_by_layer(wrapper, stimulus_set, layers)
    f_ids = list(feats.pop('_stimulus_id'))
    f_index = {s: i for i, s in enumerate(f_ids)}
    tgt_ids = [str(s) for s in target_stimulus_ids]
    common = [s for s in tgt_ids if s in f_index]
    yi = [tgt_ids.index(s) for s in common]
    fi = [f_index[s] for s in common]
    Y = np.asarray(target)[yi]
    feats = {l: feats[l][fi] for l in layers}
    result = explore_layer_mapping(feats, Y, localizer_frac=localizer_frac, alpha=alpha)
    approaches = score_approaches(feats, Y, result, top_n_layers, top_k, alpha)
    return result, approaches, result.composite_selector(top_n_layers, top_k)
