"""Tests for the topographic-organization metric. The metric must (a) give a
high spatial-smoothness index to a topographically organized synthetic model
and ~0 to a random one, (b) produce a decreasing correlation-vs-distance
profile for the topographic case, and (c) — as a brain-correspondence Metric —
score topographic-vs-topographic high and random-vs-topographic low.

The synthetic "topographic" model places units on a 2-D sheet and gives each
stimulus a smooth low-frequency field sampled at the unit positions, so nearby
units are correlated — the defining TDANN property.
"""
import numpy as np
import pytest

from brainscore_core.supported_data_standards.brainio.assemblies import NeuroidAssembly
from brainscore.metrics.topographic import (
    correlation_distance_profile, spatial_smoothness, topographic_alignment,
    TopographicMetric,
)


def _grid_positions(grid):
    # row-major (C-order) over [row, col] so a (grid, grid) field's .ravel()
    # aligns element-for-element with these positions.
    rr, cc = np.meshgrid(np.arange(grid), np.arange(grid), indexing='ij')
    return (np.stack([rr.ravel(), cc.ravel()], axis=-1).astype(float)
            / max(1, grid - 1))


def make_topographic(n_stim=60, grid=16, seed=0, sigma=2.0):
    # Each stimulus is a smooth low-pass field (white noise blurred by a Gaussian
    # of scale `sigma` cells), so nearby units are correlated and correlation
    # decays monotonically with distance — the defining TDANN property.
    from scipy.ndimage import gaussian_filter
    rng = np.random.RandomState(seed)
    pos = _grid_positions(grid)
    responses = np.zeros((n_stim, grid * grid))
    for s in range(n_stim):
        field = gaussian_filter(rng.randn(grid, grid), sigma=sigma, mode='wrap')
        responses[s] = field.ravel()
    return responses, pos


def make_random(n_stim=60, grid=16, seed=0):
    rng = np.random.RandomState(seed)
    pos = _grid_positions(grid)
    return rng.randn(n_stim, grid * grid), pos


def _assembly(responses, positions):
    n_stim, n_units = responses.shape
    return NeuroidAssembly(
        responses,
        coords={
            'stimulus_id': ('presentation', [f's{i}' for i in range(n_stim)]),
            'object_name': ('presentation', ['x'] * n_stim),
            'neuroid_id': ('neuroid', np.arange(n_units)),
            'tissue_x': ('neuroid', positions[:, 0]),
            'tissue_y': ('neuroid', positions[:, 1]),
        },
        dims=['presentation', 'neuroid'])


class TestSmoothness:
    def test_topographic_higher_than_random(self):
        tr, tp = make_topographic()
        rr, rp = make_random()
        topo = spatial_smoothness(tr, tp)
        rand = spatial_smoothness(rr, rp)
        assert topo > 0.2          # clearly topographic
        assert abs(rand) < 0.1     # no spatial structure
        assert topo > rand + 0.2

    def test_profile_decreases_with_distance(self):
        tr, tp = make_topographic()
        centers, corr, counts = correlation_distance_profile(tr, tp, n_bins=12)
        finite = np.isfinite(corr)
        # near-distance correlation exceeds far-distance correlation
        assert corr[finite][0] > corr[finite][-1]
        assert (counts > 0).sum() >= 8


class TestAlignment:
    def test_topo_vs_topo_beats_random_vs_topo(self):
        tr, tp = make_topographic(seed=0)
        tr2, tp2 = make_topographic(seed=1)         # independent topographic "brain"
        rr, rp = make_random(seed=2)
        topo_topo, _ = topographic_alignment(tr, tp, tr2, tp2)
        rand_topo, _ = topographic_alignment(rr, rp, tr2, tp2)
        assert topo_topo > rand_topo
        assert topo_topo > 0.5


class TestMetric:
    def test_metric_returns_score_with_profiles(self):
        tr, tp = make_topographic(seed=0)
        tr2, tp2 = make_topographic(seed=3)
        metric = TopographicMetric(n_bins=12)
        score = metric(_assembly(tr, tp), _assembly(tr2, tp2))
        assert float(score) > 0.5
        assert 'model_profile' in score.attrs
        assert 'brain_smoothness' in score.attrs

    def test_metric_requires_positions(self):
        tr, tp = make_topographic()
        bad = NeuroidAssembly(
            tr,
            coords={'stimulus_id': ('presentation', [f's{i}' for i in range(tr.shape[0])]),
                    'object_name': ('presentation', ['x'] * tr.shape[0]),
                    'neuroid_id': ('neuroid', np.arange(tr.shape[1]))},
            dims=['presentation', 'neuroid'])
        with pytest.raises(ValueError, match="tissue_x"):
            TopographicMetric()(bad, _assembly(*make_topographic(seed=1)))
