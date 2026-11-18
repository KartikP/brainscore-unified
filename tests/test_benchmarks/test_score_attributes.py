"""Every unified benchmark score carries the same attributes, written in one place."""
import ast
import math
from pathlib import Path

import pytest

from brainscore.benchmarks import benchmark_score, not_estimated
from brainscore_core.benchmarks import BenchmarkBase
from brainscore_core.metrics import Score
from tests.architecture_support import plugin_test

pytestmark = pytest.mark.unit

COMMON = ('raw', 'ceiling', 'normalized', 'metric', 'benchmark_version')


def _benchmark(ceiling):
    class Fixture(BenchmarkBase):
        def __call__(self, subject):
            raise AssertionError('not run')
    return Fixture(identifier='fixture', ceiling=ceiling, version=3, parent='behavioral')


def test_a_normalized_score_carries_raw_ceiling_and_metric():
    benchmark = _benchmark(Score(0.8))
    score = benchmark_score(benchmark, raw=0.6, metric='accuracy', n_items=5)
    assert all(key in score.attrs for key in COMMON)
    assert float(score) == 0.6 / 0.8
    assert isinstance(score.attrs['raw'], Score) and float(score.attrs['raw']) == 0.6
    assert score.attrs['ceiling'] is benchmark.ceiling
    assert score.attrs['normalized'] is True and score.attrs['metric'] == 'accuracy'
    assert score.attrs['benchmark_version'] == 3
    assert score.attrs['n_items'] == 5


def test_a_raw_score_carries_the_not_estimated_ceiling():
    score = benchmark_score(_benchmark(not_estimated()), raw=0.2, metric='r')
    assert float(score) == 0.2
    assert score.attrs['normalized'] is False
    assert math.isnan(float(score.attrs['ceiling']))
    assert score.attrs['ceiling'].attrs['status'] == 'not_estimated'


def test_another_reference_is_kept_as_a_score():
    score = benchmark_score(_benchmark(not_estimated()), raw=0.4, metric='r', ceiling=0.8)
    assert isinstance(score.attrs['ceiling'], Score) and float(score.attrs['ceiling']) == 0.8
    assert float(score) == 0.4 / 0.8 and score.attrs['normalized'] is True


def test_a_per_target_value_needs_a_measured_ceiling():
    score = benchmark_score(_benchmark(not_estimated()), value=0.7, raw=0.4, metric='r',
                            ceiling=Score(0.6))
    assert float(score) == 0.7 and score.attrs['normalized'] is True
    with pytest.raises(ValueError, match='measured ceiling'):
        benchmark_score(_benchmark(not_estimated()), value=0.7, raw=0.4, metric='r')


@pytest.mark.parametrize('ceiling,flag', [
    (not_estimated(), True), (Score(0.8), False), (Score(0.8), 1), (not_estimated(), 'yes')])
def test_the_normalized_flag_cannot_be_set(ceiling, flag):
    with pytest.raises(TypeError, match="normalized"):
        benchmark_score(_benchmark(ceiling), raw=0.4, metric='r', normalized=flag)


def test_benchmark_version_cannot_be_set():
    with pytest.raises(TypeError, match="benchmark_version"):
        benchmark_score(_benchmark(Score(0.8)), raw=0.4, metric='r', benchmark_version=9)


@pytest.mark.parametrize('ceiling', [Score(float('nan')), float('nan')])
def test_an_untagged_nan_ceiling_is_refused(ceiling):
    with pytest.raises(ValueError, match=r'not_estimated\(\)'):
        benchmark_score(_benchmark(not_estimated()), raw=0.4, metric='r', ceiling=ceiling)
    with pytest.raises(ValueError, match=r'not_estimated\(\)'):
        benchmark_score(_benchmark(Score(float('nan'))), raw=0.4, metric='r')


def test_a_measured_ceiling_of_one_stays_measured():
    score = benchmark_score(_benchmark(Score(1.0)), raw=0.4, metric='r')
    assert float(score) == 0.4 and score.attrs['normalized'] is True


def test_no_unified_benchmark_writes_the_common_attributes_by_hand():
    """The common attributes come from benchmark_score only, so they have one shape."""
    root = Path(__file__).parents[2] / 'brainscore'
    files = [*root.joinpath('benchmarks').rglob('*.py'), *root.joinpath('harnesses').rglob('*.py')]
    offenders = []
    for path in files:
        if path.name == '_score.py' or plugin_test(path):
            continue
        for node in ast.walk(ast.parse(path.read_text())):
            # score.attrs['raw'] = ... / score.attrs['ceiling'] = ...
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if (isinstance(target, ast.Subscript) and isinstance(target.slice, ast.Constant)
                            and target.slice.value in COMMON + ('ceiling_status',)
                            and isinstance(target.value, ast.Attribute)
                            and target.value.attr == 'attrs'):
                        offenders.append(f'{path.name}:{node.lineno}')
            # score.attrs.update(raw=...)
            if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                    and node.func.attr == 'update'
                    and any(k.arg in COMMON + ('ceiling_status',) for k in node.keywords)):
                offenders.append(f'{path.name}:{node.lineno}')
    assert offenders == []
