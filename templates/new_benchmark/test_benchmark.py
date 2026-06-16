"""Tests are the spec. A benchmark should: register + load by identifier, expose
identifier/ceiling/version, and produce a Score from a candidate. A full data-backed run
belongs in a slow/integration test; here, check the contract and (optionally) run against a
tiny dummy candidate so the wiring is exercised offline. Adapt freely.
"""
import pytest


def test_registered_and_loads():
    import brainscore
    assert 'your-benchmark' in brainscore.benchmark_registry
    bench = brainscore.load_benchmark('your-benchmark')
    assert bench.identifier == 'your-benchmark'
    assert bench.ceiling is not None
    assert bench.version >= 1


@pytest.mark.slow
def test_scores_a_candidate():
    # TODO: load a real (small) model and assert 0 <= score <= ~1.2 (ceiled).
    import brainscore
    bench = brainscore.load_benchmark('your-benchmark')
    model = brainscore.load_model('your-model')
    score = bench(model)
    assert float(score) == pytest.approx(float(score))  # finite
