"""score() returns the N/A result for a pair that is not applicable, before any compute."""
import math

import pytest

import brainscore
from brainscore_core.contract import Subject
from brainscore_core.memory import MemoryError
from brainscore_core.metrics import Score

pytestmark = pytest.mark.unit


class _Recorder(Subject):
    """A subject that records every call to participate."""

    identifier = 'recorder'
    in_channels = frozenset()
    out_channels = frozenset()
    required_channels = frozenset()

    def __init__(self, in_channels, out_channels):
        self.in_channels = frozenset(in_channels)
        self.out_channels = frozenset(out_channels)
        self.calls = []

    def participate(self, session):
        self.calls.append(session)

    def reset(self):
        pass


class _Benchmark:
    identifier = 'toy-neural'
    required_input_channels = frozenset({'vision:image'})
    requested_output_channels = frozenset({'neural:IT'})

    def __init__(self):
        self.runs = 0

    def __call__(self, subject):
        self.runs += 1
        return Score(0.25)


def test_incompatible_pair_is_na_and_runs_nothing():
    subject, benchmark = _Recorder({'vision:image'}, {'behavior'}), _Benchmark()
    result = brainscore.score(subject, benchmark, check_mem=False)
    assert math.isnan(float(result))
    assert result.attrs['status'] == 'N/A'
    assert result.attrs['missing_channels'] == ['neural:IT']
    assert "'neural:IT'" in result.attrs['reason']
    assert result.attrs['model_identifier'] == 'recorder'
    assert result.attrs['benchmark_identifier'] == 'toy-neural'
    assert subject.calls == [] and benchmark.runs == 0


def test_na_comes_before_the_memory_check(monkeypatch):
    import brainscore_core.memory as memory
    monkeypatch.setattr(memory, 'check_memory', lambda *a, **k: pytest.fail('memory checked'))
    result = brainscore.score(_Recorder({'text'}, {'neural:IT'}), _Benchmark(), check_mem=True)
    assert result.attrs['status'] == 'N/A'
    assert result.attrs['missing_channels'] == ['vision:image']


def test_memory_failure_still_raises(monkeypatch):
    import brainscore_core.memory as memory

    def refuse(*args, **kwargs):
        raise MemoryError('not enough memory')
    monkeypatch.setattr(memory, 'check_memory', refuse)
    benchmark = _Benchmark()
    with pytest.raises(MemoryError, match='not enough memory'):
        brainscore.score(_Recorder({'vision:image'}, {'neural:IT'}), benchmark, check_mem=True)
    assert benchmark.runs == 0


def test_compatible_pair_score_is_unchanged():
    benchmark = _Benchmark()
    result = brainscore.score(_Recorder({'vision:image'}, {'neural:IT'}), benchmark, check_mem=False)
    assert float(result) == 0.25 and benchmark.runs == 1
    assert 'status' not in result.attrs
    assert result.attrs['compatibility']['match'] == 'full'
