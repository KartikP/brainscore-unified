"""Layer mapping with unit selection — a productionized Brain-Score tool.

Given per-layer model features and a brain target, this finds the brain-optimal
layer and builds a :class:`CompositeSelector` — top-K units drawn from the top-N
layers — which is the scientifically principled mapping (a functional unit
population spanning depth, not a single hand-picked layer). It also scores the
four mapping approaches head-to-head so they can be compared:

  1. standard layer mapping      — a single best full layer
  2. unit selection in a layer   — top-K units of the best layer
  3. multiple full layers        — concatenation of the top-N full layers
  4. CompositeSelector           — top-K units from each of the top-N layers

Scoring is decoupled from extraction (``explore_layer_mapping`` takes features +
target), so it is testable offline; :func:`sweep_model` is the convenience that
extracts a model's per-layer activations and then maps.
"""
import dataclasses
from typing import Dict, List, Optional, Tuple

import numpy as np

from brainscore_core.model_interface import CompositeSelector


def per_voxel_cv_ridge(X, Y, alpha: float = 1.0, n_splits: int = 5,
                       seed: int = 0) -> np.ndarray:
    """K-fold ridge per voxel (no feature scaler); return per-voxel Pearson r.

    Ridge auto-centers via its intercept, so an explicit StandardScaler is
    intentionally omitted (it over-rescales heterogeneous-variance features and
    depresses the fit — established in the Lahner scoring work).
    """
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import KFold
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    preds = np.full_like(Y, np.nan)
    for tr, te in KFold(n_splits=n_splits, shuffle=True,
                        random_state=seed).split(np.arange(X.shape[0])):
        preds[te] = Ridge(alpha=alpha).fit(X[tr], Y[tr]).predict(X[te])
    Yc = Y - Y.mean(0, keepdims=True)
    Pc = preds - preds.mean(0, keepdims=True)
    num = (Yc * Pc).sum(0)
    den = np.sqrt((Yc ** 2).sum(0) * (Pc ** 2).sum(0))
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.where(den > 0, num / den, np.nan)


@dataclasses.dataclass
class LayerMappingResult:
    """Per-layer scores + per-unit predictivity, with selectors built on demand."""
    layer_order: List[str]
    per_layer_r: List[float]
    unit_predictivity: np.ndarray   # (n_layers, n_units): per-unit |corr| with its best voxel
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
        """Build a CompositeSelector: top-``k`` units from each of the top-``n_layers``."""
        layers = tuple(
            (layer, tuple(self.top_units(layer, k)))
            for layer in self.top_layers(n_layers))
        return CompositeSelector(layers=layers)


def explore_layer_mapping(features_by_layer: Dict[str, np.ndarray],
                          target: np.ndarray, alpha: float = 1.0,
                          n_splits: int = 5, seed: int = 0) -> LayerMappingResult:
    """Score every layer against ``target`` and rank units within each layer.

    :param features_by_layer: ``{layer_path: (n_stimuli, n_units)}`` (same row
        order as ``target``).
    :param target: ``(n_stimuli, n_voxels)`` brain responses.
    :returns: a :class:`LayerMappingResult`.
    """
    Y = np.asarray(target, dtype=np.float64)
    Yz = (Y - Y.mean(0)) / (Y.std(0) + 1e-8)
    layer_order = list(features_by_layer.keys())
    n_units = next(iter(features_by_layer.values())).shape[1]
    per_layer_r: List[float] = []
    unit_pred = np.zeros((len(layer_order), n_units), dtype=np.float64)
    for li, layer in enumerate(layer_order):
        X = np.asarray(features_by_layer[layer], dtype=np.float64)
        per_layer_r.append(float(np.nanmedian(
            per_voxel_cv_ridge(X, Y, alpha=alpha, n_splits=n_splits, seed=seed))))
        Xz = (X - X.mean(0)) / (X.std(0) + 1e-8)
        corr = (Xz.T @ Yz) / X.shape[0]              # (n_units, n_voxels)
        unit_pred[li] = np.abs(corr).max(1)
    return LayerMappingResult(layer_order, per_layer_r, unit_pred, alpha)


def score_approaches(features_by_layer: Dict[str, np.ndarray], target: np.ndarray,
                     result: LayerMappingResult, top_n_layers: int = 3,
                     top_k: int = 100, alpha: float = 1.0) -> List[dict]:
    """Score the four mapping approaches head-to-head (median per-voxel r)."""
    Y = np.asarray(target, dtype=np.float64)

    def score(X):
        return float(np.nanmedian(per_voxel_cv_ridge(X, Y, alpha=alpha)))

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

    Calls ``wrapper(stimulus_set, layers=layers)`` (one pass; layers share the
    forward), reduces any ``time_bin`` axis by ``time_reduce``, and splits the
    neuroid axis back out per layer. Returns ``{layer: (n_stimuli, n_units)}``
    plus, under the key ``'_stimulus_id'``, the presentation order.
    """
    asm = wrapper(stimulus_set, layers=layers)
    if 'time_bin' in asm.dims and time_reduce == 'mean':
        asm = asm.mean('time_bin')
    asm = asm.transpose('presentation', 'neuroid')
    layer_coord = np.asarray(asm['layer'].values).ravel()
    vals = np.asarray(asm.values)
    out: Dict[str, np.ndarray] = {}
    for layer in layers:
        cols = np.where(layer_coord == layer)[0]
        out[layer] = vals[:, cols]
    out['_stimulus_id'] = np.array([str(s) for s in asm['stimulus_id'].values])
    return out


def sweep_model(wrapper, stimulus_set, target, target_stimulus_ids, layers,
                alpha: float = 1.0, top_n_layers: int = 3, top_k: int = 100):
    """End-to-end: extract a model's per-layer features, align to the target,
    and run :func:`explore_layer_mapping` + :func:`score_approaches`.

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
    result = explore_layer_mapping(feats, Y, alpha=alpha)
    approaches = score_approaches(feats, Y, result, top_n_layers, top_k, alpha)
    composite = result.composite_selector(top_n_layers, top_k)
    return result, approaches, composite
