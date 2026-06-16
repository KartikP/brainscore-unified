"""The Metric seam is first-class: metrics register in metric_registry, load via
load_metric (with vision fallback), and satisfy the Metric(assembly1, assembly2) -> Score
contract. Exercised on the reference topographic-alignment metric.
"""
import numpy as np
import pytest

from brainscore_core.supported_data_standards.brainio.assemblies import NeuroidAssembly


def _topographic_assembly(seed=0, n_units=64, n_stim=30, shuffle_positions=False):
    """A (presentation, neuroid) assembly whose units lie on a 2-D sheet and whose
    responses vary smoothly with position — so nearby units are correlated (the
    signature a topographic metric detects). shuffle_positions breaks that link."""
    rng = np.random.RandomState(seed)
    side = int(round(n_units ** 0.5))
    n_units = side * side
    gx, gy = np.meshgrid(np.linspace(0, 1, side), np.linspace(0, 1, side))
    pos = np.stack([gx.ravel(), gy.ravel()], axis=1)            # (n_units, 2)
    # each stimulus = a smooth low-frequency field over the sheet
    resp = np.zeros((n_stim, n_units))
    for s in range(n_stim):
        freq = rng.randn(2) * 4.0
        phase = rng.uniform(0, 2 * np.pi)
        resp[s] = np.sin(pos @ freq + phase) + 0.05 * rng.randn(n_units)
    coord_pos = rng.permutation(pos) if shuffle_positions else pos
    return NeuroidAssembly(
        resp,
        coords={'stimulus_id': ('presentation', [f's{i}' for i in range(n_stim)]),
                'neuroid_id': ('neuroid', list(range(n_units))),
                'tissue_x': ('neuroid', coord_pos[:, 0]),
                'tissue_y': ('neuroid', coord_pos[:, 1])},
        dims=['presentation', 'neuroid'])


class TestMetricSeamFirstClass:
    def test_registry_populated(self):
        import brainscore
        assert 'topographic-alignment' in brainscore.metric_registry

    def test_load_metric_returns_metric(self):
        import brainscore
        from brainscore_core.metrics import Metric
        from brainscore.metrics import TopographicMetric
        metric = brainscore.load_metric('topographic-alignment')
        assert isinstance(metric, Metric) and isinstance(metric, TopographicMetric)

    def test_load_metric_forwards_args(self):
        import brainscore
        metric = brainscore.load_metric('topographic-alignment', 8)
        assert metric.n_bins == 8

    def test_unknown_metric_raises(self):
        import brainscore
        with pytest.raises(KeyError, match='not found'):
            brainscore.load_metric('not-a-real-metric')

    def test_returns_score(self):
        import brainscore
        from brainscore_core.metrics import Score
        metric = brainscore.load_metric('topographic-alignment')
        score = metric(_topographic_assembly(0), _topographic_assembly(1))
        assert isinstance(score, Score)
        assert np.isfinite(float(score))

    def test_topographic_layout_aligns_better_than_shuffled(self):
        # two genuinely topographic sheets share a decaying r(d) profile -> high alignment;
        # shuffling the model's unit positions flattens its profile -> lower alignment.
        import brainscore
        metric = brainscore.load_metric('topographic-alignment')
        brain = _topographic_assembly(1)
        aligned = float(metric(_topographic_assembly(0), brain))
        shuffled = float(metric(_topographic_assembly(0, shuffle_positions=True), brain))
        assert aligned > shuffled
