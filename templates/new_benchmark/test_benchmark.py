"""Tests are the spec. A benchmark should: register + load by identifier, expose
identifier/ceiling/version, and produce a Score from a candidate.

These run offline against a tiny dummy candidate, so the wiring is exercised without
weights or data. Once your benchmark loads real measurements, enable the slow test at
the bottom. Adapt freely.
"""
import numpy as np
import pytest


class DummyCandidate:
    """The smallest thing that satisfies the interface a benchmark calls.

    A benchmark only ever touches a model through `start_recording` / `start_task` /
    `process`, so this is enough to exercise one. It also documents the contract your
    benchmark depends on: if you need more from a candidate than this, that is worth
    knowing early, because every real model will have to provide it too.
    """

    def __init__(self, n_neuroids=16, seed=0):
        self.n_neuroids, self.seed = n_neuroids, seed
        self.recorded = None

    def start_recording(self, recording_target, time_bins=None, **kwargs):
        self.recorded = recording_target

    def process(self, stimuli, **kwargs):
        from brainscore_core.supported_data_standards.brainio.assemblies import (
            NeuroidAssembly)
        stimulus_ids = list(stimuli['stimulus_id'].values)
        rng = np.random.RandomState(self.seed)
        # Two coords per dim on purpose — see the note in benchmark.py: a single
        # coord does not get promoted to a MultiIndex, and metrics then cannot find
        # stimulus_id. Real wrappers already return assemblies shaped this way.
        return NeuroidAssembly(
            rng.randn(len(stimulus_ids), self.n_neuroids),
            coords={'stimulus_id': ('presentation', stimulus_ids),
                    'presentation_index': ('presentation', list(range(len(stimulus_ids)))),
                    'neuroid_id': ('neuroid', list(range(self.n_neuroids))),
                    'region': ('neuroid', ['IT'] * self.n_neuroids)},
            dims=['presentation', 'neuroid'])


def test_registered_and_loads():
    import brainscore
    assert 'your-benchmark' in brainscore.benchmark_registry
    bench = brainscore.load_benchmark('your-benchmark')
    assert bench.identifier == 'your-benchmark'
    assert bench.ceiling is not None
    assert bench.version >= 1


def test_scores_a_candidate():
    import brainscore
    bench = brainscore.load_benchmark('your-benchmark')
    score = bench(DummyCandidate())
    assert np.isfinite(float(score)), 'benchmark returned a non-finite score'
    assert 'raw' in score.attrs and 'ceiling' in score.attrs, \
        'report the unceiled value and the ceiling used, so the number stays interpretable'


def test_asks_the_candidate_to_record():
    """The benchmark must configure the model before it processes anything."""
    import brainscore
    candidate = DummyCandidate()
    brainscore.load_benchmark('your-benchmark')(candidate)
    assert candidate.recorded is not None, \
        'benchmark called process() without start_recording() — the model had no ' \
        'active recording target'


@pytest.mark.slow
def test_scores_a_real_model():
    """TODO: enable once your benchmark loads real data.

    Point it at a small registered model and assert the score lands in a plausible
    range (0 to ~1.2 once ceiled).
    """
    pytest.skip('enable when real data and a registered model are available')
