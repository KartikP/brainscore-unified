"""Lahner2024 TR time series: task-keyed events (F6), versions, score labels (3A)
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
    CURRENT_VERSION, FEATURE_DIM_CAP, TR_SEC, TRANSDUCTIVE_SVD)
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
    """BOLD that is exactly the version-2 design of a video subject (one stream)."""
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


# ── F6: each task sees only its own events ────────────────────────


def test_each_task_sees_only_its_own_events(monkeypatch, tmp_path):
    """BOLD built from each task's own events: version 2 fits it, version 1 (both
    tasks' events in each run) does not."""
    runs = tr_runs()
    install(monkeypatch, tmp_path, runs=runs, bold=bold_of(runs, tr_events(runs)))
    scores = {version: float(Lahner2024BOLDMoments_timeresolved(
        folds=RunBlocks(2), version=version)(Recorder({'vision:video'})).attrs['raw'])
        for version in (1, 2)}
    assert scores[2] > 0.9
    assert scores[1] < scores[2] - 0.05


def test_versions_agree_on_runs_without_a_shared_key(monkeypatch, tmp_path):
    runs = [dict(run, run=f'{index:02d}') for index, run in enumerate(tr_runs())]
    install(monkeypatch, tmp_path, runs=runs)
    values = [float(Lahner2024BOLDMoments_timeresolved(folds=RunBlocks(2), version=version)(
        Recorder({'vision:video'}))) for version in (1, 2)]
    assert repr(values[0]) == repr(values[1])


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


# ── Versions ──────────────────────────────────────────────────────


@pytest.mark.parametrize('identifier', TR_IDENTIFIERS)
def test_tr_registrations_default_to_version_2(identifier):
    benchmark = load(identifier)
    assert benchmark.identifier == identifier
    assert benchmark.version == CURRENT_VERSION == 2


@pytest.mark.parametrize('identifier', TR_IDENTIFIERS)
def test_version_1_stays_loadable_under_the_same_identifier(identifier):
    benchmark = load(identifier, version=1)
    assert benchmark.identifier == identifier
    assert benchmark.version == 1


@pytest.mark.parametrize('identifier', TR_IDENTIFIERS)
def test_unknown_version_is_refused(identifier):
    with pytest.raises(ValueError, match='version must be one of'):
        load(identifier, version=3)


@pytest.mark.parametrize('identifier', (PREFIX, PREFIX + '-visualROI'))
def test_glm_beta_variants_are_unchanged(identifier):
    assert load(identifier).version == 1


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


@pytest.mark.parametrize('version', (1, 2))
@pytest.mark.parametrize('name', CONFIGURATIONS)
def test_scores_carry_method_labels(monkeypatch, tmp_path, name, version):
    install(monkeypatch, tmp_path)
    configuration = CONFIGURATIONS[name]
    subject = Recorder({'vision:video', 'audio'}, regions=('IT', 'A1'))
    score = Lahner2024BOLDMoments_timeresolved(version=version, **configuration)(subject)
    attrs = score.attrs
    assert attrs['encoding_model'] == 'clip_mean_onset_impulse_hrf'
    assert attrs['onset_rounding'] == 'floor_to_TR'
    assert attrs['event_duration_used'] is False
    assert attrs['tr_sec'] == TR_SEC
    assert attrs['stimulus_generalization'] == 'seen_stimuli'
    assert attrs['benchmark_version'] == version
    assert attrs['inner_validation'] == 'none_fixed_alpha' and attrs['ridge_alpha'] == 1.0
    assert attrs['feature_projection_protocol'] == 'none'
    assert attrs['feature_dim_cap'] == FEATURE_DIM_CAP
    if version == 1:
        assert attrs['event_key'] == ('subject', 'session', 'run')
        assert any(defect.startswith('F6') for defect in attrs['known_defects'])
    else:
        assert attrs['event_key'] == ('subject', 'session', 'task', 'run')
        assert attrs['known_defects'] == ()
    improved = 'confounds' in configuration
    assert attrs['cv_mode'] == ('within_subject' if improved else 'subject_out')
    assert attrs['motion_regressed'] is improved
    assert attrs['pipeline'] == ('tr_resolved_multimodal_concat' if 'channels' in configuration
                                 else 'continuous_time_encoding')


def test_subject_out_label_reports_whether_blocks_are_whole_participants(monkeypatch, tmp_path):
    install(monkeypatch, tmp_path, runs=tr_runs(('a', 'a', 'b', 'b')))
    score = Lahner2024BOLDMoments_timeresolved(folds=RunBlocks(2))(Recorder({'vision:video'}))
    assert score.attrs['cv_policy'] == 'contiguous_run_blocks_in_assembly_order'
    assert score.attrs['cv_blocks_hold_out_whole_subjects'] is True


def test_interleaved_participants_are_reported(monkeypatch, tmp_path):
    install(monkeypatch, tmp_path, runs=tr_runs(('a', 'b', 'a', 'b')))
    score = Lahner2024BOLDMoments_timeresolved(folds=RunBlocks(2))(Recorder({'vision:video'}))
    assert score.attrs['cv_blocks_hold_out_whole_subjects'] is False


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
