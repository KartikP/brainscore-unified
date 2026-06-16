"""Tests are the spec. A metric should: be loadable by identifier, return a Score,
score identical inputs near 1, and score unrelated inputs lower. Adapt to your metric.
"""
import numpy as np
import pytest


def _toy_assembly(seed=0):
    """Build a minimal (presentation, neuroid) assembly. Replace with a fixture
    that carries whatever coords your metric requires."""
    from brainscore_core.supported_data_standards.brainio.assemblies import NeuroidAssembly
    rng = np.random.RandomState(seed)
    data = rng.randn(20, 8)  # 20 stimuli x 8 units
    return NeuroidAssembly(
        data,
        coords={'stimulus_id': ('presentation', [f's{i}' for i in range(20)]),
                'neuroid_id': ('neuroid', list(range(8)))},
        dims=['presentation', 'neuroid'])


def test_registered_and_loadable():
    import brainscore
    assert 'your-metric' in brainscore.metric_registry
    metric = brainscore.load_metric('your-metric')
    assert callable(metric)


def test_identical_inputs_score_high():
    import brainscore
    metric = brainscore.load_metric('your-metric')
    a = _toy_assembly(0)
    score = metric(a, a)
    assert float(score) == pytest.approx(1.0, abs=0.05)


def test_unrelated_inputs_score_lower():
    import brainscore
    metric = brainscore.load_metric('your-metric')
    s_same = float(metric(_toy_assembly(0), _toy_assembly(0)))
    s_diff = float(metric(_toy_assembly(0), _toy_assembly(99)))
    assert s_diff < s_same
