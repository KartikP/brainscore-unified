"""Lahner2024 TR time series: task-keyed events, the version, score labels (3A)
and the fold rules. Every test runs the benchmark call on synthetic data behind
the registered loaders (``tests/lahner_synthetic.py``); no model, no download.
"""
import numpy as np
import pytest

import brainscore
import brainscore.benchmarks.lahner2024  # noqa: F401  (registers the benchmarks)
from brainscore.benchmarks.lahner2024 import (
    Lahner2024BOLDMoments_timeresolved, ParticipantRunBlocks, RunBlocks)
from brainscore.benchmarks.lahner2024.benchmark_timeresolved import (
    FEATURE_DIM_CAP, RUN_KEY, TR_SEC, TRANSDUCTIVE_SVD)
from brainscore.data.lahner2024 import MOTION_COLUMNS
from brainscore.temporal import double_gamma_hrf, hrf_convolve
from tests.lahner_synthetic import (
    CLIPS, N_TR, N_VOXELS, Recorder, install, tr_events, tr_runs, unit_values)

PREFIX = 'Lahner2024-fMRI-naturalistic'
TR_SUFFIXES = (
    '-timeresolved', '-timeresolved-visualROI',
    '-timeresolved-improved', '-timeresolved-improved-visualROI',
    '-timeresolved-multimodal', '-timeresolved-multimodal-visualROI',
    '-timeresolved-multimodal-auditoryROI',
)
TR_IDENTIFIERS = tuple(PREFIX + suffix for suffix in TR_SUFFIXES)
AUDIOVISUAL = ('vision:image', 'vision:video', 'audio')


def load(identifier, **configuration):
    return brainscore.load_benchmark(identifier, **configuration)


def bold_of(runs, events, width=4):
    """BOLD that is exactly the design of a video subject (one stream)."""
    kernel = double_gamma_hrf(duration_sec=32.0, sampling_rate_hz=1.0 / TR_SEC)
    bold = np.zeros((len(runs), N_TR, N_VOXELS))
    for index, run in enumerate(runs):
        impulses = np.zeros((N_TR, width), dtype=np.float32)
        own = events[(events['subject'] == run['subject']) & (events['task'] == run['task'])
                     & (events['session'] == run['session']) & (events['run'] == run['run'])]
        for _, event in own.iterrows():
            impulses[int(np.floor(event['onset_sec'] / TR_SEC))] += unit_values(
                event['stimulus_id'], 'neural:IT', width)[0]
        design = hrf_convolve(impulses, sampling_rate_hz=1.0 / TR_SEC, hrf=kernel)
        bold[index] = design[:, np.arange(N_VOXELS) % width]
    return bold + 1e-3 * np.random.default_rng(0).standard_normal(bold.shape)


# ── Each task sees only its own events ────────────────────────────


def test_each_task_sees_only_its_own_events(monkeypatch, tmp_path):
    """The two tasks share (subject, session, run); BOLD built from each task's own
    events is fitted, while a design that mixed the two tasks' events is not."""
    runs = tr_runs()
    events = tr_events(runs)
    install(monkeypatch, tmp_path, runs=runs, bold=bold_of(runs, events))
    own = float(Lahner2024BOLDMoments_timeresolved(folds=RunBlocks(2))(
        Recorder({'vision:video'})).attrs['raw'])
    assert own > 0.9
    mixed_events = events.assign(task='train')   # every run gets both tasks' events
    mixed = bold_of([dict(run, task='train') for run in runs], mixed_events)
    (tmp_path / 'mixed').mkdir()
    install(monkeypatch, tmp_path / 'mixed', runs=runs, bold=mixed, events=events)
    assert float(Lahner2024BOLDMoments_timeresolved(folds=RunBlocks(2))(
        Recorder({'vision:video'})).attrs['raw']) < own - 0.05


def test_run_without_events_raises(monkeypatch, tmp_path):
    runs = tr_runs()
    events = tr_events(runs)
    install(monkeypatch, tmp_path, runs=runs, events=events[events['task'] == 'train'])
    with pytest.raises(ValueError, match='No events for run'):
        Lahner2024BOLDMoments_timeresolved(folds=RunBlocks(2))(Recorder({'vision:video'}))


def test_events_without_task_column_raise(monkeypatch, tmp_path):
    runs = tr_runs()
    install(monkeypatch, tmp_path, runs=runs, events=tr_events(runs).drop(columns='task'))
    with pytest.raises(ValueError, match='lacks'):
        Lahner2024BOLDMoments_timeresolved(folds=RunBlocks(2))(Recorder({'vision:video'}))


# ── Version ───────────────────────────────────────────────────────

GLM_IDENTIFIERS = tuple(PREFIX + suffix for suffix in (
    '', '-visualROI', '-multimodal', '-multimodal-visualROI', '-multimodal-auditoryROI'))


@pytest.mark.parametrize('identifier', TR_IDENTIFIERS + GLM_IDENTIFIERS)
def test_every_registration_is_version_2(identifier):
    benchmark = load(identifier)
    assert benchmark.identifier == identifier
    assert benchmark.version == 2


def test_no_older_version_is_offered():
    with pytest.raises(TypeError):
        load(TR_IDENTIFIERS[0], version=1)


# ── Score labels (3A) and fold rules ──────────────────────────────

# The registry lines with folds small enough for four runs.
CONFIGURATIONS = {
    'one stream': dict(folds=RunBlocks(2)),
    'improved': dict(folds=ParticipantRunBlocks(1), confounds=MOTION_COLUMNS),
    'improved ROI': dict(folds=ParticipantRunBlocks(1), confounds=MOTION_COLUMNS,
                         reliability_threshold=0.3),
    'two streams': dict(folds=RunBlocks(2), channels=AUDIOVISUAL),
    'two streams auditory ROI': dict(folds=RunBlocks(2), channels=AUDIOVISUAL,
                                     voxel_mask_fn=lambda: np.arange(N_VOXELS) % 7 == 0),
}


@pytest.mark.parametrize('name', CONFIGURATIONS)
def test_scores_carry_method_labels(monkeypatch, tmp_path, name):
    install(monkeypatch, tmp_path)
    configuration = CONFIGURATIONS[name]
    subject = Recorder({'vision:video', 'audio'}, regions=('IT', 'A1'))
    score = Lahner2024BOLDMoments_timeresolved(**configuration)(subject)
    attrs = score.attrs
    assert attrs['encoding_model'] == 'clip_mean_onset_impulse_hrf'
    assert attrs['onset_rounding'] == 'floor_to_TR'
    assert attrs['event_duration_used'] is False
    assert attrs['tr_sec'] == TR_SEC
    assert attrs['stimulus_generalization'] == 'seen_stimuli'
    assert attrs['benchmark_version'] == 2
    assert attrs['inner_validation'] == 'none_fixed_alpha' and attrs['ridge_alpha'] == 1.0
    assert attrs['feature_projection_protocol'] == 'none'
    assert attrs['feature_dim_cap'] == FEATURE_DIM_CAP
    assert attrs['event_key'] == RUN_KEY == ('subject', 'session', 'task', 'run')
    assert 'known_defects' not in attrs
    improved = 'confounds' in configuration
    assert attrs['cv_mode'] == ('within_participant' if improved else 'participant_out')
    assert 'subject' not in attrs['cv_policy']
    assert attrs['motion_regressed'] is improved
    assert attrs['pipeline'] == ('tr_resolved_multimodal_concat' if 'channels' in configuration
                                 else 'continuous_time_encoding')


def test_participant_out_label_reports_whether_blocks_are_whole_participants(
        monkeypatch, tmp_path):
    install(monkeypatch, tmp_path, runs=tr_runs(('a', 'a', 'b', 'b')))
    score = Lahner2024BOLDMoments_timeresolved(folds=RunBlocks(2))(Recorder({'vision:video'}))
    assert score.attrs['cv_policy'] == 'contiguous_run_blocks_in_assembly_order'
    assert score.attrs['cv_blocks_hold_out_whole_participants'] is True
    assert 'cv_blocks_hold_out_whole_subjects' not in score.attrs


def test_interleaved_participants_are_reported(monkeypatch, tmp_path):
    install(monkeypatch, tmp_path, runs=tr_runs(('a', 'b', 'a', 'b')))
    score = Lahner2024BOLDMoments_timeresolved(folds=RunBlocks(2))(Recorder({'vision:video'}))
    assert score.attrs['cv_blocks_hold_out_whole_participants'] is False


@pytest.mark.parametrize('channels, label', [
    (('vision:image', 'vision:video'), TRANSDUCTIVE_SVD),
    (AUDIOVISUAL, TRANSDUCTIVE_SVD + '_per_modality'),
])
def test_transductive_projection_is_labelled(monkeypatch, tmp_path, channels, label):
    """Feature widths above the cap are projected with an SVD fit on all stimuli."""
    install(monkeypatch, tmp_path)
    subject = Recorder({'vision:video', 'audio'}, regions=('IT', 'A1'), width=FEATURE_DIM_CAP + 1)
    score = Lahner2024BOLDMoments_timeresolved(folds=RunBlocks(2), channels=channels)(subject)
    assert score.attrs['feature_projection_protocol'] == label
    assert len(CLIPS) <= FEATURE_DIM_CAP   # the projection keeps at most one column per clip
