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
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from brainscore_core.model_interface import CompositeSelector


def _per_voxel_pearson(Y_true: np.ndarray, Y_pred: np.ndarray) -> np.ndarray:
    Yc = Y_true - Y_true.mean(0, keepdims=True)
    Pc = Y_pred - Y_pred.mean(0, keepdims=True)
    num = (Yc * Pc).sum(0)
    den = np.sqrt((Yc ** 2).sum(0) * (Pc ** 2).sum(0))
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.where(den > 0, num / den, np.nan)


def per_voxel_train_test(X_tr, Y_tr, X_te, Y_te, alpha: float = 1.0) -> np.ndarray:
    """Fit ridge on the localizer rows, score per-voxel Pearson r on the test rows.

    No feature scaler: Ridge auto-centers via its intercept; an explicit
    StandardScaler over-rescales heterogeneous-variance features and depresses
    the fit (established in the Lahner scoring work).
    """
    from sklearn.linear_model import Ridge
    reg = Ridge(alpha=alpha).fit(np.asarray(X_tr, np.float64), np.asarray(Y_tr, np.float64))
    pred = reg.predict(np.asarray(X_te, np.float64))
    return _per_voxel_pearson(np.asarray(Y_te, np.float64), pred)


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
                     top_k: int = 100, alpha: float = 1.0) -> List[dict]:
    """Score the four mapping approaches on the held-out TEST split.

    Unit/layer choices come from the localizer (via ``result``); each approach's
    readout is fit on the localizer and scored on the test split.
    """
    Y = np.asarray(target, np.float64)
    L, T = result.localizer_idx, result.test_idx

    def score(X_full):
        X = np.asarray(X_full, np.float64)
        return float(np.nanmedian(per_voxel_train_test(X[L], Y[L], X[T], Y[T], alpha)))

    best = result.best_layer
    top = result.top_layers(top_n_layers)
    n_units = features_by_layer[best].shape[1]
    standard = result.best_r
    unit_within = score(features_by_layer[best][:, result.top_units(best, top_k)])
    multi_full = score(np.concatenate([features_by_layer[l] for l in top], axis=1))
    composite = score(np.concatenate(
        [features_by_layer[l][:, result.top_units(l, top_k)] for l in top], axis=1))
    return [
        {'name': 'standard layer mapping', 'detail': f'single best full layer ({best})',
         'n_features': n_units, 'r': round(standard, 4)},
        {'name': 'unit selection within a layer', 'detail': f'top-{top_k} units of {best}',
         'n_features': top_k, 'r': round(unit_within, 4)},
        {'name': 'multiple full layers', 'detail': f'concat of {top}',
         'n_features': n_units * len(top), 'r': round(multi_full, 4)},
        {'name': 'CompositeSelector', 'detail': f'top-{top_k} units from each of {top}',
         'n_features': top_k * len(top), 'r': round(composite, 4)},
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
