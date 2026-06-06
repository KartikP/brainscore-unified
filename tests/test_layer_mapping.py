"""Unit tests for the layer-mapping + unit-selection tool.

Synthetic data: one layer's units linearly generate the brain target, with a
known subset of units carrying the signal. The tool should rank that layer
first, rank those units highest, and build a CompositeSelector over them.
"""
import numpy as np
import pytest

from brainscore_core.model_interface import CompositeSelector
from brainscore.tools.layer_mapping import (
    explore_layer_mapping, score_approaches, score_budget_curve,
    normalize_by_ceiling, per_voxel_train_test, LayerMappingResult)


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

    def test_each_approach_has_a_random_null(self):
        layers, Y, _ = _synthetic()
        res = explore_layer_mapping(layers, Y, alpha=1.0)
        approaches = score_approaches(layers, Y, res, top_n_layers=2, top_k=8, n_null_seeds=4)
        for a in approaches:
            assert 'random_null' in a and 'random_null_sd' in a
            assert -1.0 <= a['random_null'] <= 1.0 and a['random_null_sd'] >= 0.0

    def test_unit_selection_beats_its_random_null(self):
        # the signal units (top-8 by predictivity) must beat 8 RANDOM units of
        # the same layer — the whole point of selection.
        layers, Y, _ = _synthetic()
        res = explore_layer_mapping(layers, Y, alpha=1.0)
        a = {x['name']: x for x in score_approaches(layers, Y, res, top_n_layers=2, top_k=8)}
        unit = a['unit selection within a layer']
        comp = a['CompositeSelector']
        assert unit['r'] > unit['random_null'] + 0.1     # selected ≫ random units
        assert comp['r'] > comp['random_null']

    def test_unit_selection_recovers_most_of_full_layer(self):
        # selecting the 8 signal units should score close to the full 40-unit layer
        layers, Y, _ = _synthetic()
        res = explore_layer_mapping(layers, Y, alpha=1.0)
        a = {x['name']: x['r'] for x in score_approaches(layers, Y, res, 2, 8)}
        assert a['unit selection within a layer'] >= a['standard layer mapping'] - 0.1
        # composite uses fewer features than multiple-full-layers
        feats = {x['name']: x['n_features'] for x in score_approaches(layers, Y, res, 2, 8)}
        assert feats['CompositeSelector'] < feats['multiple full layers']


class TestRigorousScoring:
    """The scientifically-valid comparison: RidgeCV per-voxel α tuning, ceiling
    normalization, budget-matched curve with matched random nulls."""

    def test_ridgecv_recovers_signal(self):
        """Passing an α grid uses RidgeCV (per-voxel tuning) and still recovers
        the planted signal."""
        layers, Y, _ = _synthetic()
        L = np.arange(100); T = np.arange(100, 200)
        r = per_voxel_train_test(layers['L1'][L], Y[L], layers['L1'][T], Y[T],
                                 alpha=(1., 10., 100., 1000.))
        assert float(np.nanmedian(r)) > 0.8

    def test_ridgecv_matches_ridge_when_grid_is_singleton_ish(self):
        """Sanity: scalar α and a grid bracketing it give comparable fits."""
        layers, Y, _ = _synthetic()
        L = np.arange(100); T = np.arange(100, 200)
        r_fixed = float(np.nanmedian(
            per_voxel_train_test(layers['L1'][L], Y[L], layers['L1'][T], Y[T], alpha=1.0)))
        r_cv = float(np.nanmedian(
            per_voxel_train_test(layers['L1'][L], Y[L], layers['L1'][T], Y[T],
                                 alpha=(0.1, 1., 10.))))
        assert r_cv >= r_fixed - 0.05      # CV never much worse; usually better

    def test_normalize_by_ceiling_divides_and_drops_unreliable(self):
        r = np.array([0.4, 0.2, 0.1])
        ceiling = np.array([0.8, 0.5, 0.05])      # last voxel below min_ceiling
        out = normalize_by_ceiling(r, ceiling, min_ceiling=0.1)
        assert np.isclose(out[0], 0.5) and np.isclose(out[1], 0.4)
        assert np.isnan(out[2])                    # unreliable voxel dropped

    def test_top_units_pooled_ranks_signal_units(self):
        layers, Y, signal_units = _synthetic()
        res = explore_layer_mapping(layers, Y, alpha=1.0)
        pairs = res.top_units_pooled(res.top_layers(2), k=8)
        assert len(pairs) == 8
        # the signal lives in L1's first 8 units — they should dominate the pool
        l1_signal = [(l, u) for (l, u) in pairs if l == 'L1' and u in signal_units]
        assert len(l1_signal) >= 6

    def test_budget_curve_structure_and_monotonic_reference(self):
        layers, Y, _ = _synthetic()
        res = explore_layer_mapping(layers, Y, alpha=1.0)
        curve = score_budget_curve(layers, Y, res, budgets=[2, 4, 8, 16, 40],
                                   top_n_layers=2, alpha_grid=(1., 10., 100.),
                                   n_null_seeds=3)
        # both strategies populated, each entry has r + matched null
        assert len(curve['within_layer']) == 5
        for e in curve['within_layer']:
            assert {'k', 'r', 'random_null', 'random_null_sd'} <= set(e)
        # reference points present
        assert 'whole_layer_r' in curve and 'several_layers_r' in curve
        # the within-layer curve should rise with K toward the whole-layer ref
        rs = [e['r'] for e in curve['within_layer']]
        assert rs[-1] >= rs[0]                     # more units → not worse (RidgeCV)

    def test_budget_curve_selection_beats_random_at_small_k(self):
        """At a small budget, picking the signal units must beat random picks —
        the whole point. (Distributed-signal data would show no gap; this
        synthetic has localized signal, so the gap must appear.)"""
        layers, Y, _ = _synthetic()
        res = explore_layer_mapping(layers, Y, alpha=1.0)
        curve = score_budget_curve(layers, Y, res, budgets=[8], top_n_layers=2,
                                   alpha_grid=(1., 10., 100.), n_null_seeds=4)
        w = curve['within_layer'][0]
        assert w['r'] > w['random_null'] + 0.1

    def test_budget_curve_ceiling_normalization(self):
        layers, Y, _ = _synthetic(n_voxels=15)
        res = explore_layer_mapping(layers, Y, alpha=1.0)
        ceiling = np.full(15, 0.9)                  # uniform high ceiling
        raw = score_budget_curve(layers, Y, res, budgets=[8], top_n_layers=2,
                                 alpha_grid=(1., 10.), n_null_seeds=2)
        norm = score_budget_curve(layers, Y, res, budgets=[8], top_n_layers=2,
                                  alpha_grid=(1., 10.), n_null_seeds=2,
                                  noise_ceiling=ceiling)
        assert raw['normalized'] is False and norm['normalized'] is True
        # dividing by 0.9 inflates the score by ~1/0.9
        assert norm['within_layer'][0]['r'] > raw['within_layer'][0]['r']

    def test_budget_curve_respects_max_budget(self):
        """Budgets above the layer size are skipped for within-layer; pooled
        allows up to n_units * n_top_layers."""
        layers, Y, _ = _synthetic(n_units=40)
        res = explore_layer_mapping(layers, Y, alpha=1.0)
        curve = score_budget_curve(layers, Y, res, budgets=[40, 60], top_n_layers=2,
                                   alpha_grid=(1., 10.), n_null_seeds=2)
        within_ks = [e['k'] for e in curve['within_layer']]
        pooled_ks = [e['k'] for e in curve['pooled_layers']]
        assert 60 not in within_ks                  # 60 > 40 units in one layer
        assert 60 in pooled_ks                       # 60 <= 80 pooled across 2 layers


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
