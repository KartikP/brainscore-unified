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

from brainscore_core.model_interface import Subject
from brainscore_core.supported_data_standards.brainio.assemblies import NeuroidAssembly
from brainscore_core.streaming import StreamEvent


class DummyCandidate(Subject):
    """The smallest subject this benchmark can score: it implements only participate.

    A benchmark reaches a subject through a session (`neural_response` builds one and
    calls `subject.participate(session)`), so this is enough to exercise one. No
    start_recording, no process: if your benchmark needs more from a subject than
    this, that is worth knowing early, because every subject will have to provide it.
    """

    identifier = 'dummy-candidate'
    in_channels = {'vision:image'}
    out_channels = {'neural:IT'}

    def __init__(self, n_neuroids=16, seed=0):
        self.n_neuroids, self.seed = n_neuroids, seed
        self.requested = None

    def participate(self, session):
        self.requested = tuple(session.requested_output_channels)
        rng = np.random.RandomState(self.seed)
        # One input event per stimulus; answer each with a one-stimulus response.
        while (event := session.next_input()) is not None:
            response = NeuroidAssembly(
                rng.randn(1, self.n_neuroids),
                coords={
                    # which stimulus this row answers (two coords per axis, so
                    # metrics can index them)
                    'stimulus_id': ('presentation', [event.meta['stimulus_id']]),
                    'stimulus_index': ('presentation', [event.meta['stimulus_index']]),
                    # which recorded unit each column is
                    'neuroid_id': ('neuroid', [f'IT.{unit}' for unit in range(self.n_neuroids)]),
                    'region': ('neuroid', ['IT'] * self.n_neuroids),
                },
                dims=['presentation', 'neuroid'],
            )
            session.respond(StreamEvent(
                channel='neural:IT',
                payload=response,
                t_ms=event.t_ms,
                meta=dict(event.meta),
            ))


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


def test_declares_inputs_outputs_and_protocol():
    """What the benchmark sends, asks for, and how its trials relate."""
    import brainscore
    bench = brainscore.load_benchmark('your-benchmark')
    assert set(bench.required_input_channels) == {'vision:image'}
    assert set(bench.requested_output_channels) == {'neural:IT'}
    assert bench.trial_memory == 'none'


def test_preflight_accepts_a_match_and_refuses_a_missing_input_or_output():
    import brainscore
    from brainscore_core.compatibility import CompatibilityError, check_compatibility
    bench = brainscore.load_benchmark('your-benchmark')
    assert check_compatibility(DummyCandidate(), bench).kind == 'full'

    class NoVision(DummyCandidate):
        in_channels = {'text'}

    class NoIT(DummyCandidate):
        out_channels = {'neural:V4'}

    with pytest.raises(CompatibilityError, match='takes in none of the inputs'):
        check_compatibility(NoVision(), bench)
    with pytest.raises(CompatibilityError, match='cannot produce output channels'):
        check_compatibility(NoIT(), bench)


def test_asks_the_candidate_for_its_region():
    """The session must tell the subject which response to produce."""
    import brainscore
    candidate = DummyCandidate()
    brainscore.load_benchmark('your-benchmark')(candidate)
    assert candidate.requested == ('neural:IT',), \
        'the session did not request the region this benchmark scores'


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
