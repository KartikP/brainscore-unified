"""Structural tests for the Algonauts 2025 benchmark scaffold.

Data is on EC2 (~100 GB DataLad pull); these tests don't load any of
it. They verify:
- All 14 registry entries exist (subject x split, and two more forms of sub01).
- Constructor validates subject ∈ {1, 2, 3, 5} and the form.
- ``__call__`` raises NotImplementedError until the assembly
  preparation pipeline runs on EC2.
- The data plugin raises FileNotFoundError with a clear pointer to
  prepare_assembly.py when data hasn't been built yet.
"""
import pytest

import brainscore


# ── Registry presence ─────────────────────────────────────────────


def test_friends_benchmarks_registered_per_subject():
    for sub in (1, 2, 3, 5):
        assert (f'Algonauts2025-friends-sub{sub:02d}'
                in brainscore.benchmark_registry)


def test_friends_s7_benchmarks_registered_per_subject():
    for sub in (1, 2, 3, 5):
        assert (f'Algonauts2025-friends-s7-sub{sub:02d}'
                in brainscore.benchmark_registry)


def test_ood_benchmarks_registered_per_subject():
    for sub in (1, 2, 3, 5):
        assert (f'Algonauts2025-ood-sub{sub:02d}'
                in brainscore.benchmark_registry)


def test_total_algonauts_benchmark_count():
    """Three splits × four subjects = 12 entries, plus the video and video-audio
    variants of sub01."""
    entries = [k for k in brainscore.benchmark_registry
               if k.startswith('Algonauts2025-')]
    assert len(entries) == 14, (
        f"expected 14 Algonauts entries, got {len(entries)}: {sorted(entries)}")


# ── Constructor validation ────────────────────────────────────────


def test_invalid_participant_raises():
    from brainscore.benchmarks.algonauts2025.benchmark import (
        Algonauts2025Friends)
    with pytest.raises(ValueError, match="participants are"):
        Algonauts2025Friends(participant=4)


# ── Scaffold contract ─────────────────────────────────────────────


def test_held_out_splits_raise_when_scored_directly():
    """Held-out (S7, OOD) splits have no ground truth — scoring them directly
    raises a clear error pointing at the Codabench prediction path
    (generate_predictions -> submit_codabench), rather than silently scoring."""
    from brainscore.benchmarks.algonauts2025.benchmark import (
        Algonauts2025FriendsS7, Algonauts2025OOD)
    for cls in (Algonauts2025FriendsS7, Algonauts2025OOD):
        b = cls(participant=1)
        with pytest.raises(ValueError, match="prediction target.*generate_predictions"):
            b(subject=None)


def test_assembly_load_raises_clear_error_when_data_missing(tmp_path):
    """If the assembly hasn't been built yet, the load path should
    raise FileNotFoundError pointing the user at prepare_assembly.py."""
    from brainscore.benchmarks.algonauts2025.benchmark import (
        Algonauts2025Friends)
    b = Algonauts2025Friends(participant=1, assembly_root=tmp_path)
    with pytest.raises(FileNotFoundError, match="prepare_assembly"):
        _ = b.assembly


def test_default_alpha_grid_logarithmic():
    """One penalty per decade from 1e-2 to 1e8, so 1e5 (the best video penalty in
    the earlier research note) is inside the grid, not at its edge."""
    import numpy as np
    from brainscore.benchmarks.algonauts2025.benchmark import (
        _Algonauts2025Base)
    grid = np.asarray(_Algonauts2025Base.BANDED_ALPHA_GRID)
    np.testing.assert_allclose(np.diff(np.log10(grid)), 1.0)
    assert grid[0] == 1e-2 and grid[-1] == 1e8 and len(grid) == 11
    assert 1e5 in grid and grid[0] < 1e5 < grid[-1]


def test_the_window_stays_inside_a_run_and_the_lags_are_right(tmp_path):
    import numpy as np
    from brainscore.benchmarks.algonauts2025.benchmark import Algonauts2025Friends
    b = Algonauts2025Friends(participant=1, stimulus_window=3, hrf_delay=2,
                             assembly_root=tmp_path)
    run = np.array([0] * 8 + [1] * 8)
    X = np.arange(16, dtype=float)[:, None] + 1   # the value of row t is t + 1
    stacked = b._apply_stimulus_window_and_hrf(X, run)
    for t in range(16):
        start = 0 if t < 8 else 8
        # Columns hold rows t-4, t-3, t-2 (window 3, delay 2); a lag before the run is 0.
        expected = [X[t - lag, 0] if t - lag >= start else 0.0 for lag in (4, 3, 2)]
        np.testing.assert_array_equal(stacked[t], expected)


def test_exclusions_at_a_run_start_cover_the_window(tmp_path):
    from brainscore.benchmarks.algonauts2025.benchmark import Algonauts2025Friends
    b = Algonauts2025Friends(participant=1, assembly_root=tmp_path)
    assert b._excluded_samples_start == 3 + 5 - 1          # hrf_delay + window - 1
    assert b.version == 2
    with pytest.raises(ValueError, match='at least hrf_delay'):
        Algonauts2025Friends(participant=1, excluded_samples_start=5, assembly_root=tmp_path)


def test_a_run_out_of_scan_order_is_refused(tmp_path):
    import numpy as np
    import xarray as xr
    from brainscore.benchmarks.algonauts2025.benchmark import Algonauts2025Friends
    b = Algonauts2025Friends(participant=1, assembly_root=tmp_path)
    b._assembly = xr.DataArray(np.zeros((3, 1)), dims=('presentation', 'neuroid'), coords={
        'stimulus_id': ('presentation', ['m'] * 3), 'run': ('presentation', ['1'] * 3),
        't_within_run': ('presentation', [0, 2, 1])})
    with pytest.raises(ValueError, match='not in t_within_run order'):
        b._runs_and_keep()


def test_a_missing_tr_is_filled_from_its_own_run_only(tmp_path):
    import numpy as np
    import pandas as pd
    import xarray as xr
    from brainscore.benchmarks.algonauts2025.benchmark import Algonauts2025Friends
    b = Algonauts2025Friends(participant=1, assembly_root=tmp_path)
    b._assembly = xr.DataArray(np.zeros((4, 1)), dims=('presentation', 'neuroid'), coords={
        'stimulus_id': ('presentation', ['a', 'a', 'b', 'b']),
        'run': ('presentation', ['1'] * 4), 't_within_run': ('presentation', [0, 1, 0, 1])})
    # Run a lacks its second TR; run b lacks its first.
    table = pd.DataFrame({'stimulus_id': ['a_t0', 'b_t1'], 'frame_id': ['a_t0', 'b_t1'],
                          'clip_id': ['a', 'b'], 't_within_run': [0, 1]})
    X = b._align_features_to_assembly(np.array([[1.0], [2.0]]), ['a_t0', 'b_t1'], table)
    np.testing.assert_array_equal(X[:, 0], [1.0, 1.0, 2.0, 2.0])


class _Coord:
    def __init__(self, values):
        self.values = values
    def __len__(self):
        return len(self.values)


class _MockAssembly:
    """Stands in for the loaded fMRI assembly: stimulus_id + run coords, no data."""
    def __init__(self, stim_ids, runs):
        self._stim = stim_ids
        self._run = runs
    def __getitem__(self, key):
        return _Coord(self._stim if key == 'stimulus_id' else self._run)


def _mock_assembly_two_runs(tr_per_run=20):
    # two runs -> unique (stim, run) blocks; used to exercise the exclusion math
    stim = ['v1'] * tr_per_run + ['v2'] * tr_per_run
    run = ['r1'] * tr_per_run + ['r2'] * tr_per_run
    return _MockAssembly(stim, run), 2 * tr_per_run, 2  # assembly, n_trs, n_runs


def test_execution_plan_declares_TR_cardinality_and_a_band_bound(tmp_path):
    """The benchmark declares its memory-execution shape: one extraction row per TR
    (not per video), a metric width of one reduced band per requested region,
    post-exclusion metric rows, a ridge category, no re-run ceiling, and a probe
    frame (raw width probed through one requested region)."""
    from brainscore.benchmarks.algonauts2025.benchmark import (
        Algonauts2025Friends, _Algonauts2025Base)
    from brainscore_core.execution_plan import ExecutionPlan

    b = Algonauts2025Friends(participant=1, assembly_root=tmp_path)
    assembly, n_trs, n_runs = _mock_assembly_two_runs(tr_per_run=20)
    b._assembly = assembly
    plan = b.execution_plan
    assert isinstance(plan, ExecutionPlan)
    assert plan.n_extraction_presentations == n_trs                 # 40 TRs, not videos
    assert plan.feature_width is None                               # probed (model-dependent)
    assert plan.metric_feature_width == (
        3 * b._stimulus_window * _Algonauts2025Base.LAYER_DIM_CAP)    # V1, V4, IT: 3 * 5 * 100
    # metric fits over TRs MINUS per-run excluded samples: 40 - (7+5)*2 = 16
    excluded = b._excluded_samples_start + b._excluded_samples_end
    assert plan.resolved_metric_observations == n_trs - excluded * n_runs
    assert plan.metric_category == 'ridge'                          # not the mis-detected PLS
    assert plan.runs_ceiling_metric is False                       # constant Score(1.0) ceiling
    assert plan.recording_target is None                           # a requested region
    assert plan.probe_stimuli is not None                          # image frame, not a video row


def test_metric_observations_clamps_per_short_run(tmp_path):
    """Short-run case: a run shorter than the excluded span retains 0 rows,
    not a negative count. Runs of 6 and 20 with 7 excluded at the start and 5
    at the end retain max(6-12,0) + max(20-12,0) = 0 + 8 = 8 (not 26 - 24 = 2)."""
    from brainscore.benchmarks.algonauts2025.benchmark import Algonauts2025Friends

    b = Algonauts2025Friends(participant=1, assembly_root=tmp_path)
    stim = ['short'] * 6 + ['long'] * 20
    run = ['r1'] * 6 + ['r2'] * 20
    b._assembly = _MockAssembly(stim, run)
    assert b._excluded_samples_start == 7 and b._excluded_samples_end == 5
    assert b.execution_plan.resolved_metric_observations == 8


def test_check_memory_probes_a_requested_region_with_a_frame_and_sizes_off_TR_count(tmp_path):
    """check_memory takes the DECLARED path: it records a requested region in the
    model's region-map order and probes an image frame an image-only model accepts,
    then sizes off the TR count — not the video count — and reports
    DECLARED-grounded (not APPROXIMATE)."""
    import logging
    from unittest.mock import patch
    from brainscore.benchmarks.algonauts2025.benchmark import Algonauts2025Friends
    from brainscore_core import memory as mem

    recorded = {}

    import numpy as np
    from brainscore_core.model_interface import UnifiedModel
    from brainscore_core.supported_data_standards.brainio.assemblies import NeuroidAssembly

    class _ImageOnlyModel(UnifiedModel):  # a typed subject: probed through participate
        identifier = 'img-only'
        region_layer_map = {'V1': 'early', 'IT': 'late'}  # IT is NOT first
        supported_modalities = {'vision'}
        def start_recording(self, target, time_bins=None, recording_type=None):
            recorded['target'] = target
        def process(self, stimuli):
            # an image-only model chokes on a raw video row; the probe passes it
            # the declared image frame, which it accepts
            if not hasattr(stimuli, 'stimulus_paths'):
                raise RuntimeError("cannot process a raw video row")
            return NeuroidAssembly(np.zeros((1, 768)), dims=['presentation', 'neuroid'],
                                   coords={'stimulus_id': ('presentation', [0]),
                                           'neuroid_id': ('neuroid', list(range(768)))})

    b = Algonauts2025Friends(participant=1, assembly_root=tmp_path)
    assembly, n_trs, _ = _mock_assembly_two_runs(tr_per_run=20)
    b._assembly = assembly

    with patch.object(mem, 'get_host_available_memory', return_value=64_000_000_000), \
         patch('psutil.Process') as proc, \
         mem_caplog(logging.INFO) as records:
        proc.return_value.memory_info.return_value.rss = 500_000_000
        mem.check_memory(_ImageOnlyModel(), b)
    msgs = [r.getMessage() for r in records]
    assert recorded['target'] == 'V1'                        # requested, first in the map
    assert any('DECLARED-grounded' in m for m in msgs)
    assert any(f'{n_trs} presentations' in m for m in msgs)  # TR count, not videos
    assert not any('APPROXIMATE' in m for m in msgs)


from contextlib import contextmanager

@contextmanager
def mem_caplog(level):
    import logging
    logger = logging.getLogger('brainscore_core.memory')
    records = []

    class _H(logging.Handler):
        def emit(self, record):
            records.append(record)

    h = _H(); h.setLevel(level)
    old_level = logger.level
    logger.setLevel(level); logger.addHandler(h)
    try:
        yield records
    finally:
        logger.removeHandler(h); logger.setLevel(old_level)


# ── Video form ────────────────────────────────────────────────────

import shutil

needs_ffmpeg = pytest.mark.skipif(
    shutil.which('ffmpeg') is None or shutil.which('ffprobe') is None,
    reason='ffmpeg and ffprobe are needed')


def test_video_variant_is_registered_for_sub01_train_only():
    entries = sorted(k for k in brainscore.benchmark_registry
                     if k.startswith('Algonauts2025-') and k.endswith('-video'))
    assert entries == ['Algonauts2025-friends-sub01-video']


def test_video_variant_identifier_and_declarations(tmp_path):
    from brainscore.benchmarks.algonauts2025.benchmark import Algonauts2025Friends
    from tests.benchmark_declarations import assert_declares, assert_preflight
    video = brainscore.benchmark_registry['Algonauts2025-friends-sub01-video']()
    assert video.identifier == 'Algonauts2025-friends-sub01-video'
    assert_declares(video, {'vision:video'}, {'neural:V1', 'neural:V4', 'neural:IT'})
    assert_preflight(Algonauts2025Friends(participant=1, form='video', assembly_root=tmp_path))
    frame = brainscore.benchmark_registry['Algonauts2025-friends-sub01']()
    assert frame.identifier == 'Algonauts2025-friends-sub01'
    assert_declares(frame, {'vision:image'}, {'neural:V1', 'neural:V4', 'neural:IT'})


def test_invalid_form_raises():
    from brainscore.benchmarks.algonauts2025.benchmark import Algonauts2025Friends
    with pytest.raises(ValueError, match="form must be one of"):
        Algonauts2025Friends(participant=1, form='audio')


@needs_ffmpeg
def test_video_execution_plan_probes_a_blank_clip(tmp_path):
    from brainscore.benchmarks.algonauts2025.benchmark import (
        Algonauts2025Friends, CLIP_FRAMES)
    import subprocess
    b = Algonauts2025Friends(participant=1, form='video', assembly_root=tmp_path)
    assembly, n_trs, _ = _mock_assembly_two_runs(tr_per_run=20)
    b._assembly = assembly
    plan = b.execution_plan
    assert plan.n_extraction_presentations == n_trs
    assert plan.recording_target is None
    probe = plan.probe_stimuli
    assert len(probe) == 1 and 'video_path' in probe.columns
    assert 'image_file_name' not in probe.columns
    path = probe['video_path'].iloc[0]
    frames = subprocess.run(
        ['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-count_frames',
         '-show_entries', 'stream=nb_read_frames,width,height', '-of', 'csv=p=0', path],
        capture_output=True, text=True, check=True).stdout.strip()
    assert frames == f'224,224,{CLIP_FRAMES}'


def test_frame_form_plan_still_probes_an_image(tmp_path):
    from brainscore.benchmarks.algonauts2025.benchmark import Algonauts2025Friends
    b = Algonauts2025Friends(participant=1, assembly_root=tmp_path)
    b._assembly = _mock_assembly_two_runs(tr_per_run=20)[0]
    probe = b.execution_plan.probe_stimuli
    assert 'image_file_name' in probe.columns and 'video_path' not in probe.columns
