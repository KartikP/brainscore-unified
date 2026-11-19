"""Lahner2024 TR time series: the feature reduction is fitted inside each fold, on the
stimuli shown in the training runs of that fold. Every test runs the benchmark call on
synthetic data behind the registered loaders (``tests/lahner_synthetic.py``): a subject
is scored through ``participate``; no model, no download.
"""
import numpy as np
import pytest
from sklearn.decomposition import TruncatedSVD

from brainscore.benchmarks.lahner2024.benchmark_timeresolved import (
    Lahner2024BOLDMoments_timeresolved, ParticipantRunBlocks, RunBlocks)
from brainscore.benchmarks.lahner2024 import benchmark_timeresolved as module
from brainscore.benchmarks.lahner2024.benchmark_timeresolved import FEATURE_DIM_CAP
from tests.lahner_synthetic import CLIPS, Recorder, install, tr_runs, unit_values

WIDE = FEATURE_DIM_CAP + 1
ID = 'Lahner2024-fMRI-naturalistic-timeresolved'
ONLY_IN_ONE_RUN = (CLIPS[4], CLIPS[5])   # shown only in the 'test' task runs of the synthetic events


def score(folds, subject):
    return Lahner2024BOLDMoments_timeresolved(folds=folds, identifier=ID)(subject)


def record_reductions(monkeypatch):
    """Wrap ``_reduce``; every call is kept as ``(features, train, result)``."""
    calls = []
    real = module._reduce

    def wrapper(features, train):
        result = real(features, train)
        calls.append((features, train.copy(), result))
        return result
    monkeypatch.setattr(module, '_reduce', wrapper)
    return calls


def old_reduce(features, train):
    """The version before the repair: the SVD of every stimulus, whatever the fold."""
    if features.shape[1] <= FEATURE_DIM_CAP:
        return features
    return TruncatedSVD(n_components=FEATURE_DIM_CAP, random_state=0).fit_transform(
        features).astype(np.float32)


def test_no_fitted_thing_sees_a_stimulus_shown_only_in_held_out_runs(monkeypatch, tmp_path):
    """Run 2 (a 'test' run) is held out in the last fold of RunBlocks(1); the training runs
    0 and 1 show four of the six clips. The reduction of that fold is the same when the
    features of the two other clips are replaced by random values."""
    install(monkeypatch, tmp_path, runs=tr_runs(('a', 'b', 'b')))
    calls = record_reductions(monkeypatch)
    score(RunBlocks(1), Recorder({'vision:video'}, width=WIDE))
    first = list(calls)
    del calls[:]
    changed = {clip: np.random.default_rng(7).standard_normal((1, WIDE)).astype(np.float32)
               for clip in ONLY_IN_ONE_RUN}
    score(RunBlocks(1), Recorder({'vision:video'}, width=WIDE, values=lambda clip, channel: (
        changed[clip] if clip in changed else unit_values(clip, channel, WIDE))))
    second = list(calls)

    assert len(first) == len(second)
    n_trained_on_four = 0
    for (_, train_a, result_a), (_, train_b, result_b) in zip(first, second):
        np.testing.assert_array_equal(train_a, train_b)
        if train_a.sum() == 4:   # the two changed clips are held out only
            n_trained_on_four += 1
            np.testing.assert_array_equal(result_a[train_a], result_b[train_b])
        else:                    # a fit that saw them does change: the check can fail
            assert not np.array_equal(result_a[train_a], result_b[train_b])
    assert n_trained_on_four == 1


def test_all_stimuli_in_training_is_the_old_fit_to_the_last_digit(monkeypatch, tmp_path):
    """Leave one participant out: the other participant saw every clip, so every fold
    fits on every stimulus, in the order of the full array: the old components, the old score."""
    runs = tr_runs(('a', 'a', 'b', 'b'))
    install(monkeypatch, tmp_path, runs=runs)
    calls = record_reductions(monkeypatch)
    new = score(RunBlocks(2), Recorder({'vision:video'}, width=WIDE))
    assert len(calls) == 1                       # one fit for the one distinct training set
    features, train, result = calls[0]
    assert train.all()
    np.testing.assert_array_equal(result, old_reduce(features, train))
    monkeypatch.setattr(module, '_reduce', old_reduce)
    old = score(RunBlocks(2), Recorder({'vision:video'}, width=WIDE))
    assert float(new.attrs['raw']) == float(old.attrs['raw'])
    assert float(new) == float(old)


def test_a_stream_at_or_below_the_cap_is_untouched(monkeypatch, tmp_path):
    install(monkeypatch, tmp_path, runs=tr_runs(('a', 'b', 'b')))
    features = np.random.default_rng(1).standard_normal((6, FEATURE_DIM_CAP)).astype(np.float32)
    assert module._reduce(features, np.array([True] * 4 + [False] * 2)) is features
    new = score(RunBlocks(1), Recorder({'vision:video'}, width=FEATURE_DIM_CAP))
    assert new.attrs['feature_projection_protocol'] == 'none'
    monkeypatch.setattr(module, '_reduce', lambda features, train: features)
    old = score(RunBlocks(1), Recorder({'vision:video'}, width=FEATURE_DIM_CAP))
    assert float(new.attrs['raw']) == float(old.attrs['raw'])


def test_one_fit_for_each_distinct_set_of_training_stimuli(monkeypatch, tmp_path):
    """Four folds within participants, two distinct training sets (four clips each,
    fewer than the components: scikit-learn then returns one component per clip)."""
    install(monkeypatch, tmp_path, runs=tr_runs(('a', 'a', 'b', 'b')))
    calls = record_reductions(monkeypatch)
    result = score(ParticipantRunBlocks(1), Recorder({'vision:video'}, width=WIDE))
    assert len(calls) == 2
    assert sorted(int(train.sum()) for _, train, _ in calls) == [4, 4]
    assert all(out.shape == (6, 4) for _, _, out in calls)
    assert result.attrs['feature_projection_protocol'] == module.TRAINING_SVD


def test_the_score_says_what_the_code_does(monkeypatch, tmp_path):
    install(monkeypatch, tmp_path, runs=tr_runs(('a', 'a', 'b', 'b')))
    assert module.TRAINING_SVD == 'truncated_svd_fit_on_training_stimuli_of_each_fold'
    assert not hasattr(module, 'TRANSDUCTIVE_SVD')
