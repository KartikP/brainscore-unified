"""One form for "no ceiling": ``not_estimated()``; no score value moves.

Each case below is a call that a unified benchmark made at commit 9442779 (the
frozen builder in ``tests/_benchmark_score_before_no_ceiling.py``) and the call it makes
now. The score value and the raw value are identical; only the ceiling attribute
(1.0 -> not estimated) and, for the template, the normalized flag change.
"""
import math
from types import SimpleNamespace

import numpy as np
import pytest

import brainscore
from brainscore.benchmarks import benchmark_score, not_estimated
from brainscore_core.metrics import Score
from tests._benchmark_score_before_no_ceiling import benchmark_score as old_benchmark_score
from tests._benchmark_score_before_no_ceiling import not_estimated as old_not_estimated
from tests.lahner_synthetic import Recorder, install

pytestmark = pytest.mark.unit

RAW = 0.1 + 0.2            # not exactly representable: a division by 1.0 must keep it
MEASURED = 0.7305          # a measured ceiling, any value below 1
PER_TARGET = 0.5123        # a median of per-target r / c


def _bench(ceiling):
    return SimpleNamespace(ceiling=ceiling, version=2)


def _array_raw():
    return Score(RAW)


# (case, old call, new call, ceiling after, normalized before, normalized after)
CASES = {
    # grid_game, human_replay, roar_yeatman2021: a measured ceiling divides.
    'measured': (
        lambda: old_benchmark_score(_bench(Score(MEASURED)), RAW / MEASURED, raw=RAW,
                                    metric='m', normalized=True),
        lambda: benchmark_score(_bench(Score(MEASURED)), raw=RAW, metric='m'),
        MEASURED, True, True),
    # Lahner2024 scalar path (10 variants): Score(1.0) meant "none" and divided by it.
    'one_divided': (
        lambda: old_benchmark_score(_bench(Score(1.0)), RAW / 1.0, raw=RAW, metric='m',
                                    normalized=False),
        lambda: benchmark_score(_bench(not_estimated()), raw=RAW, metric='m'),
        'not_estimated', False, False),
    # LeBel2023, induced dyslexia: Score(1.0) meant "none" and did not divide.
    'one_not_divided': (
        lambda: old_benchmark_score(_bench(Score(1.0)), RAW, raw=RAW, metric='m',
                                    normalized=False),
        lambda: benchmark_score(_bench(not_estimated()), raw=RAW, metric='m'),
        'not_estimated', False, False),
    # Algonauts2025, EWoK, CogGym (adapter_score): already not_estimated().
    'not_estimated': (
        lambda: old_benchmark_score(_bench(old_not_estimated()), RAW, raw=RAW, metric='m',
                                    normalized=False),
        lambda: benchmark_score(_bench(not_estimated()), raw=RAW, metric='m'),
        'not_estimated', False, False),
    # Lahner2024-fMRI-naturalistic-visualROI: a per-voxel normalization keeps value=.
    'per_target': (
        lambda: old_benchmark_score(_bench(Score(MEASURED)), PER_TARGET, raw=RAW, metric='m',
                                    normalized=True, ceiling=Score(0.69)),
        lambda: benchmark_score(_bench(Score(MEASURED)), value=PER_TARGET, raw=RAW,
                                metric='m', ceiling=Score(0.69)),
        0.69, True, True),
    # induced_dyslexia, literal: old line 206 (value=lesioned_acc, Score(1.0), not divided).
    'induced_dyslexia': (
        lambda: old_benchmark_score(_bench(Score(1.0)), 0.62, raw=0.62,
                                    metric='lesioned_reading_accuracy', normalized=False),
        lambda: benchmark_score(_bench(not_estimated()), raw=0.62,
                                metric='lesioned_reading_accuracy'),
        'not_estimated', False, False),
    # roar_yeatman2021, literal: old lines 258-262 (raw / float(human ceiling)).
    'roar_yeatman2021': (
        lambda: old_benchmark_score(_bench(Score(0.811)), 0.81 / float(Score(0.811)), raw=0.81,
                                    metric='accuracy', normalized=True),
        lambda: benchmark_score(_bench(Score(0.811)), raw=0.81, metric='accuracy'),
        0.811, True, True),
    # The benchmark template: Score(1.0) with normalized=True (a wrong flag); the
    # value is the same, the flag now says the score is raw.
    'template': (
        lambda: old_benchmark_score(_bench(Score(1.0)), float(_array_raw()) / 1.0,
                                    raw=_array_raw(), metric='m', normalized=True),
        lambda: benchmark_score(_bench(not_estimated()), raw=_array_raw(), metric='m'),
        'not_estimated', True, False),
}


def _same(a: Score, b: Score) -> bool:
    """Equal dtype and equal bytes: a float comparison would hide both."""
    a, b = np.asarray(a.values), np.asarray(b.values)
    return a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()


@pytest.mark.parametrize('case', sorted(CASES))
def test_the_value_and_raw_do_not_move(case):
    old_call, new_call, ceiling_after, normalized_before, normalized_after = CASES[case]
    old, new = old_call(), new_call()
    assert _same(new, old)                                 # bit for bit, same dtype
    assert _same(new.attrs['raw'], old.attrs['raw'])
    assert old.attrs['normalized'] is normalized_before
    assert new.attrs['normalized'] is normalized_after
    if ceiling_after == 'not_estimated':
        assert math.isnan(float(new.attrs['ceiling']))
        assert new.attrs['ceiling'].attrs['status'] == 'not_estimated'
    else:
        assert _same(new.attrs['ceiling'], old.attrs['ceiling'])
        assert float(new.attrs['ceiling']) == ceiling_after


def test_before_values_from_the_frozen_builder():
    """The numbers at 9442779, stated: the ceiling attribute was 1.0 for 'none'."""
    assert float(CASES['one_divided'][0]().attrs['ceiling']) == 1.0
    assert float(CASES['one_not_divided'][0]().attrs['ceiling']) == 1.0
    assert float(CASES['template'][0]().attrs['ceiling']) == 1.0
    assert float(CASES['measured'][0]()) == RAW / MEASURED


def _old_lahner_builder(benchmark, *, raw, metric, ceiling=None, value=None, **attrs):
    """The two Lahner calls at 9442779, with the old constructor ceiling (Score(1.0)
    where no ceiling was given), run through the frozen builder."""
    if value is not None:                                  # per-voxel, measured
        return old_benchmark_score(benchmark, value, raw=raw, metric=metric,
                                   normalized=True, ceiling=ceiling, **attrs)
    measured = not math.isnan(float(benchmark.ceiling))
    old_ceiling = benchmark.ceiling if measured else Score(1.0)
    return old_benchmark_score(benchmark, raw / float(old_ceiling), raw=raw, metric=metric,
                               normalized=float(old_ceiling) != 1.0, ceiling=old_ceiling,
                               **attrs)


@pytest.mark.parametrize('identifier,measured', [
    ('Lahner2024-fMRI-naturalistic', False),
    ('Lahner2024-fMRI-naturalistic-visualROI', True),
])
def test_lahner_old_and_new_benchmark_paths_agree(monkeypatch, tmp_path, identifier, measured):
    """The same synthetic data through the old and the new builder: value, raw,
    flag and (where measured) ceiling are byte-identical."""
    import brainscore.benchmarks.lahner2024.benchmark as lahner
    install(monkeypatch, tmp_path)
    new = brainscore.load_benchmark(identifier)(Recorder({'vision:video'}))
    with monkeypatch.context() as patch:
        patch.setattr(lahner, 'benchmark_score', _old_lahner_builder)
        old = brainscore.load_benchmark(identifier)(Recorder({'vision:video'}))
    assert _same(new, old)
    assert _same(new.attrs['raw'], old.attrs['raw'])
    assert new.attrs['normalized'] is old.attrs['normalized'] is measured
    np.testing.assert_array_equal(new.attrs['per_voxel_r'], old.attrs['per_voxel_r'])
    if measured:
        assert _same(new.attrs['ceiling'], old.attrs['ceiling'])
        assert old.attrs['ceiling_aggregation'] == 'per_voxel'
    else:
        assert float(old.attrs['ceiling']) == 1.0
        assert new.attrs['ceiling'].attrs['status'] == 'not_estimated'
