"""Tests are the spec. A benchmark should: register + load by identifier, expose
identifier/ceiling/version, and produce a Score from a candidate.

These run offline against a tiny dummy candidate, so the wiring is exercised without
weights or data. Once your benchmark loads real measurements, enable the slow test at
the bottom. Adapt freely.
"""
import os

# Set BEFORE anything imports brainscore. The composition test below runs a real model,
# whose feature extraction otherwise writes to ~/.result_caching — and a template test
# has no business depending on the state of the user's home directory (it can be
# missing, read-only, or a symlink to an unmounted drive).
os.environ.setdefault('RESULTCACHING_DISABLE', '1')

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

        # Carry the stimulus set's presentation metadata onto the output. This is not
        # decoration: cross-validated metrics STRATIFY their folds on a presentation
        # coord (here `object_name`), and a prediction without it fails with
        # "Expected stratification coordinate object_name". Real wrappers do this for
        # you via _attach_stimulus_set_meta; a hand-rolled candidate must do it itself.
        coords = {'stimulus_id': ('presentation', stimulus_ids),
                  'presentation_index': ('presentation', list(range(len(stimulus_ids)))),
                  # Two coords per neuroid dim on purpose — a single coord is not
                  # promoted to a MultiIndex, and metrics then cannot find stimulus_id.
                  'neuroid_id': ('neuroid', list(range(self.n_neuroids))),
                  'region': ('neuroid', ['IT'] * self.n_neuroids)}
        for column in stimuli.columns:
            if column not in ('stimulus_id', 'image_file_name') and column not in coords:
                coords[column] = ('presentation', list(stimuli[column].values))

        return NeuroidAssembly(
            rng.randn(len(stimulus_ids), self.n_neuroids),
            coords=coords, dims=['presentation', 'neuroid'])


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


def _model_template_candidate():
    """Return the model template's candidate, or None if it is genuinely not installed.

    Resolution order matters. Once you follow the install instructions this file lives
    under `brainscore/benchmarks/<name>/`, where the sibling `new_model` package no
    longer exists — so a bare `import new_model` fails and, if that failure is swallowed,
    the composition check below silently turns into a no-op exactly for the person who
    did what the docs said. Ask the registry first, which is what "installed" means.

    Import errors from a model template that IS present are deliberately allowed to
    propagate: a broken model template must fail this test, not skip it.
    """
    import brainscore
    if 'your-model' in brainscore.model_registry:
        return brainscore.load_model('your-model')

    # In-repo layout only: templates/new_benchmark/ has templates/new_model/ beside it.
    import sys
    from pathlib import Path
    sibling = Path(__file__).resolve().parents[1] / 'new_model'
    if not sibling.is_dir():
        return None
    sys.path.insert(0, str(sibling.parent))
    import new_model  # noqa: F401  — registers 'your-model'; errors here are real
    return brainscore.load_model('your-model')


def test_composes_with_the_model_template():
    """The obvious first thing a newcomer tries: score MY model on MY benchmark.

    Regression test. This combination used to fail three separate ways — stimulus paths
    pointing at files that did not exist, a metric requiring matching unit counts, and a
    prediction missing the coord the CV splitter stratifies on. Every one of them failed
    only for a REAL candidate, so the other template tests stayed green throughout.
    """
    candidate = _model_template_candidate()
    if candidate is None:
        pytest.skip('model template is not installed; install templates/new_model per '
                    'its __init__.py to enable this composition check')

    import brainscore
    bench = brainscore.load_benchmark('your-benchmark')
    score = bench(candidate)
    assert np.isfinite(float(score))
    # Random weights against random targets: near zero, either sign. The point is that
    # it RUNS end to end, not that the number means anything.
    assert abs(float(score)) < 0.9, (
        f'implausible score {float(score)} for random weights vs random targets')


@pytest.mark.slow
def test_scores_a_real_model():
    """TODO: enable once your benchmark loads real data.

    Point it at a small registered model and assert the score lands in a plausible
    range (0 to ~1.2 once ceiled).
    """
    pytest.skip('enable when real data and a registered model are available')
