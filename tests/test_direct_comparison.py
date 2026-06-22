"""Tests for the direct-comparison metric (whole-model→whole-brain encoders)."""
import numpy as np
import pytest

from brainscore_core.supported_data_standards.brainio.assemblies import NeuroidAssembly
from brainscore.metrics.direct_comparison import DirectComparisonMetric


def _assembly(data, stimulus_ids):
    # Real assemblies carry multiple presentation coords (a true MultiIndex with
    # a named 'stimulus_id' level); a 'repetition' coord keeps the index unique.
    n_pres, n_units = data.shape
    return NeuroidAssembly(
        data, dims=['presentation', 'neuroid'],
        coords={'stimulus_id': ('presentation', list(stimulus_ids)),
                'repetition': ('presentation', np.arange(n_pres)),
                'neuroid_id': ('neuroid', np.arange(n_units))})


def test_high_r_when_prediction_matches_and_null_near_zero():
    rng = np.random.default_rng(0)
    n_stim, n_units, reps = 30, 8, 3
    signal = rng.standard_normal((n_stim, n_units))
    pred = _assembly(signal, [f's{i}' for i in range(n_stim)])
    target = _assembly(
        np.repeat(signal, reps, axis=0) + 0.1 * rng.standard_normal((n_stim * reps, n_units)),
        np.repeat([f's{i}' for i in range(n_stim)], reps))

    score = DirectComparisonMetric()(pred, target)
    assert score.attrs['n_stimuli'] == n_stim          # reps averaged per stimulus
    assert score.attrs['n_units'] == n_units
    assert score.attrs['raw'] > 0.8                    # prediction tracks measured
    assert abs(score.attrs['null']) < 0.3              # clip-shuffle kills the signal
    assert score.attrs['raw'] > score.attrs['null']


def test_unit_count_mismatch_raises():
    a = _assembly(np.zeros((5, 4)), [f's{i}' for i in range(5)])
    b = _assembly(np.zeros((5, 6)), [f's{i}' for i in range(5)])
    with pytest.raises(ValueError, match="must share unit ordering"):
        DirectComparisonMetric()(a, b)


def test_no_shared_stimulus_raises():
    a = _assembly(np.zeros((3, 4)), ['a', 'b', 'c'])
    b = _assembly(np.zeros((3, 4)), ['x', 'y', 'z'])
    with pytest.raises(ValueError, match="no shared stimulus_id"):
        DirectComparisonMetric()(a, b)


def test_loadable_from_registry():
    import brainscore
    metric = brainscore.load_metric('direct-comparison')
    assert isinstance(metric, DirectComparisonMetric)
