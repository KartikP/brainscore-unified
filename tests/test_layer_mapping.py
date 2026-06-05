"""Unit tests for the layer-mapping + unit-selection tool.

Synthetic data: one layer's units linearly generate the brain target, with a
known subset of units carrying the signal. The tool should rank that layer
first, rank those units highest, and build a CompositeSelector over them.
"""
import numpy as np
import pytest

from brainscore_core.model_interface import CompositeSelector
from brainscore.tools.layer_mapping import (
    explore_layer_mapping, score_approaches, LayerMappingResult)


def _synthetic(n_stim=200, n_units=40, n_voxels=15, seed=0):
    rng = np.random.RandomState(seed)
    # three layers; layer 'L1' drives the target via its first 8 units
    layers = {f'L{i}': rng.randn(n_stim, n_units).astype(np.float32) for i in range(3)}
    signal_units = list(range(8))
    W = rng.randn(len(signal_units), n_voxels)
    Y = layers['L1'][:, signal_units] @ W + 0.05 * rng.randn(n_stim, n_voxels)
    return layers, Y, signal_units


class TestExplore:
    def test_finds_driving_layer(self):
        layers, Y, _ = _synthetic()
        res = explore_layer_mapping(layers, Y, alpha=1.0)
        assert res.best_layer == 'L1'
        assert res.best_r > 0.8                       # near-perfect recovery
        # the other layers (pure noise) score far lower
        others = [r for l, r in zip(res.layer_order, res.per_layer_r) if l != 'L1']
        assert res.best_r > max(others) + 0.3

    def test_unit_predictivity_ranks_signal_units(self):
        layers, Y, signal_units = _synthetic()
        res = explore_layer_mapping(layers, Y, alpha=1.0)
        top8 = set(res.top_units('L1', 8))
        # the 8 signal units should dominate L1's top-8 by predictivity
        assert len(top8 & set(signal_units)) >= 6

    def test_composite_selector_shape(self):
        layers, Y, _ = _synthetic()
        res = explore_layer_mapping(layers, Y, alpha=1.0)
        sel = res.composite_selector(n_layers=2, k=5)
        assert isinstance(sel, CompositeSelector)
        assert len(sel.layers) == 2
        assert sel.layer_paths[0] == 'L1'             # best layer first
        for layer_path, indices in sel.layers:
            assert len(indices) == 5
            assert all(0 <= i < 40 for i in indices)


class TestApproaches:
    def test_four_approaches_scored(self):
        layers, Y, _ = _synthetic()
        res = explore_layer_mapping(layers, Y, alpha=1.0)
        approaches = score_approaches(layers, Y, res, top_n_layers=2, top_k=8)
        names = [a['name'] for a in approaches]
        assert names == ['standard layer mapping', 'unit selection within a layer',
                         'multiple full layers', 'CompositeSelector']
        for a in approaches:
            assert -1.0 <= a['r'] <= 1.0 and a['n_features'] > 0

    def test_unit_selection_recovers_most_of_full_layer(self):
        # selecting the 8 signal units should score close to the full 40-unit layer
        layers, Y, _ = _synthetic()
        res = explore_layer_mapping(layers, Y, alpha=1.0)
        a = {x['name']: x['r'] for x in score_approaches(layers, Y, res, 2, 8)}
        assert a['unit selection within a layer'] >= a['standard layer mapping'] - 0.1
        # composite uses fewer features than multiple-full-layers
        feats = {x['name']: x['n_features'] for x in score_approaches(layers, Y, res, 2, 8)}
        assert feats['CompositeSelector'] < feats['multiple full layers']


def test_result_dataclass_accessors():
    res = LayerMappingResult(['a', 'b', 'c'], [0.1, 0.5, 0.3],
                             np.zeros((3, 4)), localizer_idx=np.arange(5),
                             test_idx=np.arange(5, 10), alpha=1.0)
    assert res.best_layer == 'b' and res.best_r == 0.5
    assert res.top_layers(2) == ['b', 'c']


def test_localizer_test_split_is_disjoint():
    layers, Y, _ = _synthetic(n_stim=120)
    res = explore_layer_mapping(layers, Y, localizer_frac=0.5, seed=0)
    L, T = set(res.localizer_idx.tolist()), set(res.test_idx.tolist())
    assert not (L & T)                 # disjoint: no stimulus selects AND scores
    assert L | T == set(range(120))    # together they cover all stimuli
    assert abs(len(L) - len(T)) <= 1   # ~50/50
