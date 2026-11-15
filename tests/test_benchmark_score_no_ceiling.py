"""One form for "no ceiling": ``not_estimated()``; no score value moves.

Each case below is a call that a unified benchmark made at commit 9442779 (the
frozen builder in ``tests/_benchmark_score_9442779.py``) and the call it makes
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
from tests._benchmark_score_9442779 import benchmark_score as old_benchmark_score
from tests._benchmark_score_9442779 import not_estimated as old_not_estimated
from tests.lahner_synthetic import Recorder, install

pytestmark = pytest.mark.unit

RAW = 0.1 + 0.2            # not exactly representable: a division by 1.0 must keep it
MEASURED = 0.7305          # the Lahner visual-region ceiling
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


@pytest.mark.parametrize('case', sorted(CASES))
def test_the_value_and_raw_do_not_move(case):
    old_call, new_call, ceiling_after, normalized_before, normalized_after = CASES[case]
    old, new = old_call(), new_call()
    assert float(new) == float(old)                       # bit for bit
    assert float(new.attrs['raw']) == float(old.attrs['raw'])
    assert old.attrs['normalized'] is normalized_before
    assert new.attrs['normalized'] is normalized_after
    if ceiling_after == 'not_estimated':
        assert math.isnan(float(new.attrs['ceiling']))
        assert new.attrs['ceiling'].attrs['status'] == 'not_estimated'
    else:
        assert float(new.attrs['ceiling']) == float(old.attrs['ceiling']) == ceiling_after


def test_before_values_from_the_frozen_builder():
    """The numbers at 9442779, stated: the ceiling attribute was 1.0 for 'none'."""
    assert float(CASES['one_divided'][0]().attrs['ceiling']) == 1.0
    assert float(CASES['one_not_divided'][0]().attrs['ceiling']) == 1.0
    assert float(CASES['template'][0]().attrs['ceiling']) == 1.0
    assert float(CASES['measured'][0]()) == RAW / MEASURED


def test_lahner_whole_cortex_value_equals_the_old_division_by_one(monkeypatch, tmp_path):
    install(monkeypatch, tmp_path)
    benchmark = brainscore.load_benchmark('Lahner2024-fMRI-naturalistic')
    score = benchmark(Recorder({'vision:video'}))
    raw = float(score.attrs['raw'])
    old = old_benchmark_score(_bench(Score(1.0)), raw / 1.0, raw=raw, metric='m',
                              normalized=False)
    assert float(score) == float(old)
    assert score.attrs['normalized'] is False
    assert score.attrs['ceiling'].attrs['status'] == 'not_estimated'


def test_lahner_visual_region_keeps_its_measured_ceiling(monkeypatch, tmp_path):
    install(monkeypatch, tmp_path)
    score = brainscore.load_benchmark('Lahner2024-fMRI-naturalistic-visualROI')(
        Recorder({'vision:video'}))
    assert score.attrs['normalized'] is True
    assert np.isfinite(float(score.attrs['ceiling']))
