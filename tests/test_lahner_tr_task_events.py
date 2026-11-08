"""Lahner2024 TR-resolved protocol: task-keyed events (F6), versions, score
labels (3A) and the CV policy of the non-concat multimodal modes (F1).

Tiny synthetic arrays only; no model, no BOLD assembly, no download.
"""
import numpy as np
import pandas as pd
import pytest
import xarray as xr

import brainscore
import brainscore.benchmarks.lahner2024  # noqa: F401  (registers the factories)
from brainscore.benchmarks.lahner2024 import benchmark_timeresolved as tr
from brainscore.benchmarks.lahner2024.benchmark_timeresolved import TR_SEC
from brainscore.benchmarks.lahner2024.benchmark_timeresolved_multimodal import (
    Lahner2024BOLDMoments_timeresolved_multimodal,
)
from brainscore_core.temporal import double_gamma_hrf, hrf_convolve

PREFIX = 'Lahner2024-fMRI-naturalistic'
TR_SUFFIXES = (
    '-timeresolved', '-timeresolved-visualROI',
    '-timeresolved-improved', '-timeresolved-improved-visualROI',
    '-timeresolved-multimodal', '-timeresolved-multimodal-visualROI',
    '-timeresolved-multimodal-auditoryROI',
)
TR_IDENTIFIERS = tuple(PREFIX + suffix for suffix in TR_SUFFIXES)
NON_CONCAT_MODES = ('video_only', 'audio_only', 'per_modality', 'banded')
N_TR = 20


# ── Fixtures ──────────────────────────────────────────────────────


def collision_events():
    """Two tasks that share subject, session and run (as in BOLDMoments)."""
    return pd.DataFrame([
        dict(subject='01', session='01', run='01', task=task,
             stimulus_id=stim, onset_sec=0.0, duration_sec=3.0)
        for task, stim in [('train', 'train_clip'), ('test', 'test_clip')]])


def collision_assembly():
    return xr.DataArray(
        np.zeros((2, N_TR, 1), dtype=np.float32),
        dims=('presentation', 'time_bin', 'neuroid'),
        coords={key: ('presentation', values) for key, values in {
            'subject': ['01', '01'], 'session': ['01', '01'],
            'run': ['01', '01'], 'task': ['train', 'test']}.items()})


def unit_response():
    kernel = double_gamma_hrf(duration_sec=32.0, sampling_rate_hz=1.0 / TR_SEC)
    impulse = np.zeros((N_TR, 1), dtype=np.float32)
    impulse[0, 0] = 1.0
    return hrf_convolve(impulse, sampling_rate_hz=1.0 / TR_SEC, hrf=kernel)


def load(identifier, **configuration):
    return brainscore.load_benchmark(identifier, **configuration)


def capture_design(monkeypatch, benchmark):
    """Run the benchmark up to the design matrix and return it."""
    benchmark._events = collision_events()
    benchmark._assembly = collision_assembly()

    def features(candidate):
        benchmark._modality_per_feature = np.array(['video'])
        benchmark._feature_projection = 'none'
        return np.array([[1.0], [10.0]], dtype=np.float32), ['train_clip', 'test_clip']

    monkeypatch.setattr(benchmark, '_extract_per_stimulus_features', features)

    class StopAfterDesign(Exception):
        pass

    captured = []

    def capture(design, assembly):
        captured.append(design.copy())
        raise StopAfterDesign

    monkeypatch.setattr(benchmark, '_concatenate_with_mask', capture)
    with pytest.raises(StopAfterDesign):
        benchmark(None)
    return captured[0]


# ── F6: each task sees only its own events ────────────────────────


@pytest.mark.parametrize('identifier', TR_IDENTIFIERS)
def test_each_task_sees_only_its_own_events(monkeypatch, identifier):
    design = capture_design(monkeypatch, load(identifier))
    unit = unit_response()
    np.testing.assert_allclose(design[0], unit, atol=1e-6)          # train: amplitude 1
    np.testing.assert_allclose(design[1], unit * 10.0, atol=1e-6)   # test: amplitude 10


@pytest.mark.parametrize('mode', NON_CONCAT_MODES)
def test_non_concat_modes_see_only_their_own_task(monkeypatch, mode):
    benchmark = Lahner2024BOLDMoments_timeresolved_multimodal(mode=mode)
    design = capture_design(monkeypatch, benchmark)
    unit = unit_response()
    np.testing.assert_allclose(design[0], unit, atol=1e-6)
    np.testing.assert_allclose(design[1], unit * 10.0, atol=1e-6)


@pytest.mark.parametrize('identifier', TR_IDENTIFIERS)
def test_version_1_keeps_the_task_collision(monkeypatch, identifier):
    """Version 1 reproduces the historical design: both tasks get 1 + 10."""
    design = capture_design(monkeypatch, load(identifier, version=1))
    unit = unit_response()
    np.testing.assert_allclose(design[0], unit * 11.0, atol=1e-5)
    np.testing.assert_allclose(design[1], unit * 11.0, atol=1e-5)


def test_versions_agree_on_runs_without_a_shared_key():
    """Runs whose (subject, session, run) holds one task: identical designs."""
    events = pd.DataFrame([
        dict(subject='01', session='02', run='01', task='train',
             stimulus_id='a', onset_sec=1.0, duration_sec=3.0),
        dict(subject='01', session='02', run='01', task='train',
             stimulus_id='b', onset_sec=9.0, duration_sec=3.0),
        dict(subject='02', session='02', run='03', task='test',
             stimulus_id='a', onset_sec=4.5, duration_sec=3.0),
    ])
    assembly = xr.DataArray(
        np.zeros((2, N_TR, 1), dtype=np.float32),
        dims=('presentation', 'time_bin', 'neuroid'),
        coords={'subject': ('presentation', ['01', '02']),
                'session': ('presentation', ['02', '02']),
                'run': ('presentation', ['01', '03']),
                'task': ('presentation', ['train', 'test'])})
    features = np.array([[0.3, -1.2], [2.5, 0.7]], dtype=np.float32)
    designs = []
    for version in (1, 2):
        benchmark = tr.Lahner2024BOLDMoments_timeresolved(version=version)
        benchmark._events = events
        designs.append(benchmark._build_design(features, ['a', 'b'], assembly))
    np.testing.assert_array_equal(designs[0], designs[1])


def test_run_without_events_raises():
    benchmark = tr.Lahner2024BOLDMoments_timeresolved()
    benchmark._events = collision_events()
    with pytest.raises(ValueError, match='No events for run'):
        benchmark._events_for_run('01', '01', 'train', '02')


def test_events_without_task_column_raise():
    benchmark = tr.Lahner2024BOLDMoments_timeresolved()
    benchmark._events = collision_events().drop(columns='task')
    with pytest.raises(ValueError, match='lacks'):
        benchmark._build_design(np.ones((2, 1), np.float32),
                                ['train_clip', 'test_clip'], collision_assembly())


def test_selection_with_two_tasks_raises():
    """Defensive check: one run's selection must hold exactly one task."""
    benchmark = tr.Lahner2024BOLDMoments_timeresolved()
    benchmark._events = collision_events()
    groups = {('01', '01', 'train', '01'): collision_events()}  # both tasks
    with pytest.raises(ValueError, match='hold tasks'):
        benchmark._events_for_run('01', '01', 'train', '01', groups=groups)


def test_selector_keeps_event_order():
    events = pd.DataFrame([
        dict(subject='01', session='01', run='01', task='train', stimulus_id=s,
             onset_sec=t, duration_sec=3.0)
        for s, t in [('c', 8.0), ('a', 0.0), ('b', 4.0)]])
    benchmark = tr.Lahner2024BOLDMoments_timeresolved()
    benchmark._events = events
    selected = benchmark._events_for_run('01', '01', 'train', '01')
    assert selected['stimulus_id'].tolist() == ['c', 'a', 'b']


# ── Versions ──────────────────────────────────────────────────────


@pytest.mark.parametrize('identifier', TR_IDENTIFIERS)
def test_tr_registrations_default_to_version_2(identifier):
    benchmark = load(identifier)
    assert benchmark.identifier == identifier
    assert benchmark.version == tr.CURRENT_VERSION == 2


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


# ── Score labels (3A) and CV policy (F1) ──────────────────────────


def synthetic_assembly(subjects, tasks, n_tr=12, n_voxels=3, seed=0):
    rng = np.random.default_rng(seed)
    n_runs = len(subjects)
    return xr.DataArray(
        rng.standard_normal((n_runs, n_tr, n_voxels)).astype(np.float32),
        dims=('presentation', 'time_bin', 'neuroid'),
        coords={'subject': ('presentation', list(subjects)),
                'session': ('presentation', ['01'] * n_runs),
                'run': ('presentation', [f'{i:02d}' for i in range(n_runs)]),
                'task': ('presentation', list(tasks)),
                'n_valid_TR': ('presentation', [n_tr] * n_runs)})


def synthetic_events(assembly):
    rows = []
    for subject, run, task in zip(assembly['subject'].values, assembly['run'].values,
                                  assembly['task'].values):
        for k, onset in enumerate((0.0, 4.0, 8.0, 12.0)):
            rows.append(dict(subject=subject, session='01', run=run, task=task,
                             stimulus_id=f'clip{k % 3}', onset_sec=onset,
                             duration_sec=3.0))
    return pd.DataFrame(rows)


def score_synthetic(monkeypatch, benchmark, subjects=('a', 'a', 'b', 'b', 'c', 'c')):
    assembly = synthetic_assembly(subjects, ['train'] * len(subjects))
    benchmark._assembly = assembly
    benchmark._events = synthetic_events(assembly)
    benchmark._voxel_mask = np.ones(assembly.sizes['neuroid'], dtype=bool)
    rng = np.random.default_rng(1)
    if benchmark._apply_motion_regression:
        from brainscore.data.lahner2024 import MOTION_COLUMNS
        n_tr = assembly.sizes['time_bin']
        benchmark._motion = pd.concat([
            pd.DataFrame(dict(
                subject=subject, session='01', task=task, run=run,
                **{column: rng.standard_normal(n_tr) for column in MOTION_COLUMNS}))
            for subject, task, run in zip(assembly['subject'].values,
                                          assembly['task'].values,
                                          assembly['run'].values)])

    def features(candidate):
        benchmark._modality_per_feature = np.array(['video', 'video', 'audio'])
        benchmark._feature_projection = 'none'
        return rng.standard_normal((3, 3)).astype(np.float32), ['clip0', 'clip1', 'clip2']

    monkeypatch.setattr(benchmark, '_extract_per_stimulus_features', features)
    return benchmark(None)


METHOD_KEYS = ('encoding_model', 'feature_timing', 'onset_rounding',
               'event_duration_used', 'tr_sec', 'hrf', 'stimulus_generalization',
               'feature_projection_protocol', 'benchmark_version', 'event_key',
               'known_defects', 'cv_mode', 'cv_policy', 'cv_n_held_out_runs',
               'motion_regressed', 'inner_validation', 'ridge_alpha')


def assert_method_labels(score, version):
    for key in METHOD_KEYS:
        assert key in score.attrs, key
    assert score.attrs['encoding_model'] == 'clip_mean_onset_impulse_hrf'
    assert score.attrs['onset_rounding'] == 'floor_to_TR'
    assert score.attrs['event_duration_used'] is False
    assert score.attrs['tr_sec'] == TR_SEC
    assert score.attrs['stimulus_generalization'] == 'seen_stimuli'
    assert score.attrs['benchmark_version'] == version
    if version == 1:
        assert score.attrs['event_key'] == ('subject', 'session', 'run')
        assert any(defect.startswith('F6') for defect in score.attrs['known_defects'])
    else:
        assert score.attrs['event_key'] == ('subject', 'session', 'task', 'run')
        assert score.attrs['known_defects'] == ()


@pytest.mark.parametrize('version', (1, 2))
@pytest.mark.parametrize('identifier', TR_IDENTIFIERS)
def test_default_paths_carry_method_labels(monkeypatch, identifier, version):
    benchmark = load(identifier, version=version)
    benchmark._cv_n_held_out_runs = 2
    benchmark._within_subject_n_held_out = 1
    score = score_synthetic(monkeypatch, benchmark)
    assert_method_labels(score, version)
    assert score.attrs['inner_validation'] == 'none_fixed_alpha'
    assert score.attrs['feature_projection_protocol'] == 'none'
    expected_cv = 'within_subject' if '-improved' in identifier else 'subject_out'
    assert score.attrs['cv_mode'] == expected_cv
    assert score.attrs['motion_regressed'] is ('-improved' in identifier)


def test_subject_out_label_reports_whether_blocks_are_whole_subjects(monkeypatch):
    grouped = tr.Lahner2024BOLDMoments_timeresolved(cv_n_held_out_runs=2)
    score = score_synthetic(monkeypatch, grouped, subjects=('a', 'a', 'b', 'b', 'c', 'c'))
    assert score.attrs['cv_policy'] == 'contiguous_run_blocks_in_assembly_order'
    assert score.attrs['cv_blocks_hold_out_whole_subjects'] is True

    interleaved = tr.Lahner2024BOLDMoments_timeresolved(cv_n_held_out_runs=2)
    score = score_synthetic(monkeypatch, interleaved, subjects=('a', 'b', 'c', 'a', 'b', 'c'))
    assert score.attrs['cv_blocks_hold_out_whole_subjects'] is False


@pytest.mark.parametrize('mode', NON_CONCAT_MODES)
def test_non_concat_modes_label_their_actual_cv(monkeypatch, mode):
    benchmark = Lahner2024BOLDMoments_timeresolved_multimodal(
        mode=mode, cv_n_held_out_runs=2, apply_motion_regression=False)
    score = score_synthetic(monkeypatch, benchmark)
    assert_method_labels(score, 2)
    assert score.attrs['cv_mode'] == 'pooled_run_kfold'
    assert score.attrs['cv_policy'] == 'shuffled_kfold_over_pooled_runs'
    assert score.attrs['cv_n_splits'] == 3
    assert score.attrs['cv_blocks_hold_out_whole_subjects'] is False
    assert score.attrs['motion_regressed'] is False
    if mode == 'banded':
        assert score.attrs['inner_validation'] == (
            'random_rows_20pct_runs_shared_with_inner_training')
        assert score.attrs['ridge_alpha'] == 'per_group_from_grid'
    else:
        assert score.attrs['inner_validation'] == 'none_fixed_alpha'
        assert score.attrs['ridge_alpha'] == 1.0


@pytest.mark.parametrize('mode', NON_CONCAT_MODES)
@pytest.mark.parametrize('cv_mode', ('subject_out', 'within_subject'))
def test_non_concat_modes_refuse_subject_policies(mode, cv_mode):
    with pytest.raises(ValueError, match='does not implement cv_mode'):
        Lahner2024BOLDMoments_timeresolved_multimodal(mode=mode, cv_mode=cv_mode)


@pytest.mark.parametrize('mode', NON_CONCAT_MODES)
@pytest.mark.parametrize('cv_mode', (None, 'pooled_run_kfold'))
def test_non_concat_modes_accept_their_policy(mode, cv_mode):
    benchmark = Lahner2024BOLDMoments_timeresolved_multimodal(mode=mode, cv_mode=cv_mode)
    assert benchmark._cv_mode == 'pooled_run_kfold'


@pytest.mark.parametrize('cv_mode', ('subject_out', 'within_subject'))
def test_concat_mode_keeps_subject_policies(cv_mode):
    benchmark = Lahner2024BOLDMoments_timeresolved_multimodal(mode='concat', cv_mode=cv_mode)
    assert benchmark._cv_mode == cv_mode


def test_concat_mode_rejects_pooled_policy():
    with pytest.raises(ValueError, match="cv_mode must be"):
        Lahner2024BOLDMoments_timeresolved_multimodal(
            mode='concat', cv_mode='pooled_run_kfold')


def test_transductive_projection_is_labelled():
    """Feature widths above the cap are projected with an SVD fit on all stimuli."""
    benchmark = tr.Lahner2024BOLDMoments_timeresolved()
    benchmark._feature_projection = tr.TRANSDUCTIVE_SVD
    attrs = benchmark._method_attrs()
    assert attrs['feature_projection_protocol'] == (
        'transductive_truncated_svd_fit_on_all_stimuli')
    assert attrs['feature_dim_cap'] == tr.FEATURE_DIM_CAP
