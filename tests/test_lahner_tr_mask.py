"""Lahner2024 TR time series: a variant with a reliability mask takes the mask from the GLM
reliability of the BOLDMoments test-set videos and scores only the 'train' task runs. The
variants without that mask are unchanged. Synthetic data behind the registered loaders
(``tests/lahner_synthetic.py``); the subject is scored through ``participate``.
"""
import numpy as np
import pytest

import brainscore
import brainscore.benchmarks.lahner2024  # noqa: F401  (registers the benchmarks)
from brainscore.benchmarks.lahner2024 import (
    Lahner2024BOLDMoments_timeresolved, ParticipantRunBlocks, RunBlocks)
from brainscore.benchmarks.lahner2024.benchmark import presented_stimuli
from brainscore.benchmarks.lahner2024.benchmark_timeresolved import (
    _blocks_hold_out_whole_participants)
from brainscore.data.lahner2024 import MOTION_COLUMNS
from brainscore.temporal import contiguous_block_cv
from tests.lahner_synthetic import (
    CLIPS, GLM_CLIPS, GLM_REPS, N_TR, N_VOXELS, TEST_CLIPS, Recorder, glm_values, install,
    stimulus_set, tr_assembly, tr_events, tr_runs)

PREFIX = 'Lahner2024-fMRI-naturalistic-timeresolved'
AUDIOVISUAL = ('vision:image', 'vision:video', 'audio')
# a and b: train, test, train runs each (two scored runs a participant)
RUNS = tr_runs(('a', 'a', 'a', 'b', 'b', 'b'))


def events_with_test_set(runs, onsets=(0.0, 4.0, 8.0, 12.0)):
    """'train' runs show the training clips; 'test' runs show the test-set clips."""
    events = tr_events(runs, onsets)
    test = events['task'] == 'test'
    events.loc[test, 'stimulus_id'] = [
        TEST_CLIPS[CLIPS.index(clip) % len(TEST_CLIPS)] for clip in events.loc[test, 'stimulus_id']]
    return events


def mask_benchmark(**kwargs):
    kwargs.setdefault('folds', RunBlocks(2))
    return Lahner2024BOLDMoments_timeresolved(
        reliability_threshold=0.3, identifier=PREFIX + '-visualROI', **kwargs)


def glm_replaced(clips, seed):
    values = glm_values().copy()
    rows = np.array([clip in clips for _ in range(GLM_REPS) for clip in GLM_CLIPS])
    values[rows] = np.random.default_rng(seed).standard_normal((rows.sum(), N_VOXELS)) * 5
    return values


def mask_after(monkeypatch, tmp_path, name, **install_kwargs):
    install_kwargs.setdefault('runs', RUNS)
    install_kwargs.setdefault('events', events_with_test_set(RUNS))
    (tmp_path / name).mkdir()
    install(monkeypatch, tmp_path / name, **install_kwargs)
    return mask_benchmark().voxel_mask


def test_the_mask_comes_from_the_test_set_videos_and_nothing_else(monkeypatch, tmp_path):
    """Equal when the BOLD of every 'train' run changes and when the betas of the training
    videos change; different when the betas of the test-set videos change."""
    base = mask_after(monkeypatch, tmp_path, 'base')
    rng = np.random.default_rng(3)
    bold = 1000.0 + 50.0 * rng.standard_normal((len(RUNS), N_TR, N_VOXELS))
    train_bold = np.asarray(tr_assembly(RUNS).values).copy()
    for index, run in enumerate(RUNS):
        if run['task'] == 'train':
            train_bold[index] = bold[index]
    np.testing.assert_array_equal(base, mask_after(monkeypatch, tmp_path, 'bold', bold=train_bold))
    np.testing.assert_array_equal(base, mask_after(
        monkeypatch, tmp_path, 'betas', glm=glm_replaced(CLIPS, 5)))
    assert not np.array_equal(base, mask_after(
        monkeypatch, tmp_path, 'test', glm=glm_replaced(TEST_CLIPS, 6)))


def test_only_train_runs_are_scored_and_no_test_set_video_is_shown(monkeypatch, tmp_path):
    install(monkeypatch, tmp_path, runs=RUNS, events=events_with_test_set(RUNS))
    subject = Recorder({'vision:video'})
    benchmark = mask_benchmark()
    score = benchmark(subject)
    assert set(benchmark.scored_assembly['task'].values) == {'train'}
    assert score.attrs['n_runs'] == 4 and score.attrs['n_observations'] == 4 * N_TR
    assert len(benchmark.assembly['task']) == 6          # the assembly itself is whole
    shown = {clip for _, _, ids, _ in subject.seen for clip in ids}
    assert shown and not shown & set(TEST_CLIPS)
    assert score.attrs['cv_blocks_hold_out_whole_participants'] is True
    assert score.attrs['cv_block_runs'] == 2


def test_the_mask_variant_through_participate_and_the_registry_folds(monkeypatch, tmp_path):
    install(monkeypatch, tmp_path, runs=RUNS, events=events_with_test_set(RUNS))
    subject = Recorder({'vision:video', 'audio'}, regions=('IT', 'A1'))
    score = Lahner2024BOLDMoments_timeresolved(
        folds=RunBlocks(2), channels=AUDIOVISUAL, reliability_threshold=0.3,
        identifier=PREFIX + '-multimodal-visualROI')(subject)
    assert subject.seen and all(not set(ids) & set(TEST_CLIPS) for _, _, ids, _ in subject.seen)
    assert score.attrs['n_runs'] == 4
    folds = {name: brainscore.load_benchmark(PREFIX + name)._folds
             for name in ('', '-visualROI', '-multimodal-visualROI', '-improved-visualROI')}
    assert isinstance(folds['-visualROI'], RunBlocks) and folds['-visualROI'].n_runs == 40
    assert folds['-multimodal-visualROI'].n_runs == 40
    assert isinstance(folds['-improved-visualROI'], ParticipantRunBlocks)
    assert folds['-improved-visualROI'].n_runs == 10
    assert folds[''].n_runs == 52


def test_blocks_keep_their_meaning_after_the_run_filter():
    """40 'train' runs a participant: blocks of 40 hold out one participant each; blocks
    of 52 would not. Within a participant, blocks of 10 give four folds, not six."""
    participants = [f'p{i}' for i in range(10) for _ in range(40)]
    assert len(list(contiguous_block_cv(n_samples=400, block_size_samples=40))) == 10
    assert _blocks_hold_out_whole_participants(participants, 40) is True
    assert _blocks_hold_out_whole_participants(participants, 52) is False
    assert len(list(contiguous_block_cv(n_samples=40, block_size_samples=10))) == 4
    assert len(list(contiguous_block_cv(n_samples=52, block_size_samples=10))) == 6


def test_the_reduction_is_fitted_on_the_training_stimuli_of_the_filtered_folds(
        monkeypatch, tmp_path):
    """L4 on the filtered runs: the fit of each fold names only stimuli shown in the
    'train' runs of the fold, none of the test-set videos."""
    from brainscore.benchmarks.lahner2024 import benchmark_timeresolved as module
    install(monkeypatch, tmp_path, runs=RUNS, events=events_with_test_set(RUNS))
    calls = []
    real = module._reduce
    monkeypatch.setattr(module, '_reduce',
                        lambda features, train: calls.append(train.copy()) or real(features, train))
    mask_benchmark()(Recorder({'vision:video'}, width=module.FEATURE_DIM_CAP + 1))
    assert calls and all(0 < train.sum() <= len(CLIPS) for train in calls)


# ── Store key: the identifier names the set that is shown ─────────

def test_presented_stimuli_identifier_names_the_set():
    stimuli = stimulus_set()
    stimuli = presented_stimuli(stimuli, GLM_CLIPS)          # every row: unchanged
    assert stimuli.identifier == 'BOLDMoments' and len(stimuli) == len(GLM_CLIPS)
    train = presented_stimuli(stimuli, CLIPS)
    assert train.identifier == 'BOLDMoments-train' and len(train) == len(CLIPS)
    part = presented_stimuli(stimuli, CLIPS[:4])
    assert part.identifier not in ('BOLDMoments', 'BOLDMoments-train')
    assert part.identifier == presented_stimuli(stimuli, list(reversed(CLIPS[:4]))).identifier
    assert part.identifier != presented_stimuli(stimuli, CLIPS[1:5]).identifier
    assert set(part.stimulus_paths) == set(CLIPS[:4])


@pytest.mark.parametrize('onsets, identifier', [
    ((0.0, 4.0, 8.0, 12.0), None),                        # the train runs show 4 of 6 clips
    ((0.0, 4.0, 8.0, 12.0, 16.0, 20.0), 'BOLDMoments-train-videos'),   # they show all 6
])
def test_the_table_shown_has_the_identifier_of_its_set(monkeypatch, tmp_path, onsets, identifier):
    install(monkeypatch, tmp_path, runs=RUNS, events=events_with_test_set(RUNS, onsets))
    subject = Recorder({'vision:video'})
    mask_benchmark()(subject)
    seen = {table for *_, table in subject.seen}
    assert len(seen) == 1
    (table,) = seen
    if identifier:
        assert table == identifier
    else:
        assert table not in ('BOLDMoments-videos', 'BOLDMoments-train-videos')


# ── Variants without a reliability mask do not change ─────────────

# Measured at 231b496 (before this change) on the synthetic data of ``install``:
# (float(score), sum of per_voxel_r in float64), through ``repr``.
PINNED = {
    'plain': (0.002611062489449978, 32.40087760982715),
    'improved': (-0.0034763552248477936, -90.65468233136926),
    'multimodal': (0.0017713627312332392, 38.04770941377831),
    'auditory': (0.0010678978869691491, -4.130032965564169),
    'plain_wide': (-0.0002970074419863522, 15.58812397244742),
}
CASES = {
    'plain': (dict(folds=RunBlocks(2)), lambda: Recorder({'vision:video'})),
    'improved': (dict(folds=ParticipantRunBlocks(1), confounds=MOTION_COLUMNS),
                 lambda: Recorder({'vision:video'})),
    'multimodal': (dict(folds=RunBlocks(2), channels=AUDIOVISUAL),
                   lambda: Recorder({'vision:video', 'audio'}, regions=('IT', 'A1'))),
    'auditory': (dict(folds=RunBlocks(2), channels=AUDIOVISUAL,
                      voxel_mask_fn=lambda: np.arange(N_VOXELS) % 7 == 0),
                 lambda: Recorder({'vision:video', 'audio'}, regions=('IT', 'A1'))),
    'plain_wide': (dict(folds=RunBlocks(2)), lambda: Recorder({'vision:video'}, width=513)),
}


@pytest.mark.parametrize('name', CASES)
def test_variants_without_a_reliability_mask_keep_their_value(monkeypatch, tmp_path, name):
    install(monkeypatch, tmp_path)
    configuration, subject = CASES[name]
    score = Lahner2024BOLDMoments_timeresolved(
        **configuration, identifier=PREFIX)(subject())
    per_voxel = score.attrs['per_voxel_r']
    assert (float(score), float(np.nansum(per_voxel.astype('float64')))) == PINNED[name]
    assert score.attrs['n_runs'] == 4 and score.attrs['n_observations'] == 4 * N_TR
