"""Structural tests for the Algonauts 2025 benchmark scaffold.

Data is on EC2 (~100 GB DataLad pull); these tests don't load any of
it. They verify:
- All 12 (subject × split) registry entries exist.
- Constructor validates subject ∈ {1, 2, 3, 5} and mode ∈ VALID_MODES.
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
    """Three splits × four subjects = 12 entries."""
    entries = [k for k in brainscore.benchmark_registry
               if k.startswith('Algonauts2025-')]
    assert len(entries) == 12, (
        f"expected 12 Algonauts entries, got {len(entries)}: {sorted(entries)}")


# ── Constructor validation ────────────────────────────────────────


def test_invalid_subject_raises():
    from brainscore.benchmarks.algonauts2025.benchmark import (
        Algonauts2025Friends)
    with pytest.raises(ValueError, match="subjects are"):
        Algonauts2025Friends(subject=4)


def test_invalid_mode_raises():
    from brainscore.benchmarks.algonauts2025.benchmark import (
        Algonauts2025Friends)
    with pytest.raises(ValueError, match="mode must be one of"):
        Algonauts2025Friends(subject=1, mode='not_a_mode')


def test_valid_modes_construct():
    from brainscore.benchmarks.algonauts2025.benchmark import (
        Algonauts2025Friends, _Algonauts2025Base)
    for mode in _Algonauts2025Base.VALID_MODES:
        b = Algonauts2025Friends(subject=1, mode=mode)
        assert b._mode == mode


# ── Scaffold contract ─────────────────────────────────────────────


def test_held_out_splits_raise_when_scored_directly():
    """Held-out (S7, OOD) splits have no ground truth — scoring them directly
    raises a clear error pointing at the Codabench prediction path
    (generate_predictions -> submit_codabench), rather than silently scoring."""
    from brainscore.benchmarks.algonauts2025.benchmark import (
        Algonauts2025FriendsS7, Algonauts2025OOD)
    for cls in (Algonauts2025FriendsS7, Algonauts2025OOD):
        b = cls(subject=1)
        with pytest.raises(ValueError, match="no ground truth|Codabench"):
            b(candidate=None)


def test_assembly_load_raises_clear_error_when_data_missing(tmp_path):
    """If the assembly hasn't been built yet, the load path should
    raise FileNotFoundError pointing the user at prepare_assembly.py."""
    from brainscore.benchmarks.algonauts2025.benchmark import (
        Algonauts2025Friends)
    b = Algonauts2025Friends(subject=1, assembly_root=tmp_path)
    with pytest.raises(FileNotFoundError, match="prepare_assembly"):
        _ = b.assembly


def test_default_alpha_grid_logarithmic():
    """Banded ridge α grid should span enough decades to push
    uninformative modalities to ~zero contribution."""
    from brainscore.benchmarks.algonauts2025.benchmark import (
        _Algonauts2025Base)
    grid = _Algonauts2025Base.BANDED_ALPHA_GRID
    assert min(grid) <= 1.0
    assert max(grid) >= 10000.0
    assert len(grid) >= 5


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


def test_execution_plan_declares_TR_cardinality_metric_cap_and_target(tmp_path):
    """The benchmark declares its memory-execution shape: one extraction row per TR
    (not per video), a metric-width cap, post-exclusion metric rows, a ridge
    category, no re-run ceiling, and an IT-target probe frame (raw width probed)."""
    from brainscore.benchmarks.algonauts2025.benchmark import (
        Algonauts2025Friends, _Algonauts2025Base)
    from brainscore_core.execution_plan import ExecutionPlan

    b = Algonauts2025Friends(subject=1, assembly_root=tmp_path)
    assembly, n_trs, n_runs = _mock_assembly_two_runs(tr_per_run=20)
    b._assembly = assembly
    plan = b.execution_plan
    assert isinstance(plan, ExecutionPlan)
    assert plan.n_extraction_presentations == n_trs                 # 40 TRs, not videos
    assert plan.feature_width is None                               # probed (model-dependent)
    assert plan.metric_feature_width == (
        b._stimulus_window * _Algonauts2025Base.FEATURE_DIM_CAP)     # 5 * 1000
    # metric fits over TRs MINUS per-run excluded samples: 40 - (5+5)*2 = 20
    excluded = b._excluded_samples_start + b._excluded_samples_end
    assert plan.resolved_metric_observations == n_trs - excluded * n_runs
    assert plan.metric_category == 'ridge'                          # not the mis-detected PLS
    assert plan.runs_ceiling_metric is False                       # constant Score(1.0) ceiling
    assert plan.recording_target == 'IT'                           # the real target region
    assert plan.probe_stimuli is not None                          # image frame, not a video row


def test_check_memory_probes_IT_with_a_frame_and_sizes_off_TR_count(tmp_path):
    """check_memory takes the DECLARED path: it records IT (not the first region)
    and probes an image frame an image-only model accepts, then sizes off the TR
    count — not the video count — and reports DECLARED-grounded (not APPROXIMATE)."""
    import logging
    from unittest.mock import patch
    from brainscore.benchmarks.algonauts2025.benchmark import Algonauts2025Friends
    from brainscore_core import memory as mem

    recorded = {}

    class _ImageOnlyModel:
        identifier = 'img-only'
        region_layer_map = {'V1': 'early', 'IT': 'late'}  # IT is NOT first
        def start_recording(self, target, time_bins=None, recording_type=None):
            recorded['target'] = target
        def process(self, stimuli):
            # an image-only model chokes on a raw video row; the probe passes it
            # the declared image frame, which it accepts
            if not hasattr(stimuli, 'stimulus_paths'):
                raise RuntimeError("cannot process a raw video row")
            return type('R', (), {'shape': (1, 768)})()

    b = Algonauts2025Friends(subject=1, assembly_root=tmp_path)
    assembly, n_trs, _ = _mock_assembly_two_runs(tr_per_run=20)
    b._assembly = assembly

    with patch.object(mem, 'get_host_available_memory', return_value=64_000_000_000), \
         patch('psutil.Process') as proc, \
         mem_caplog(logging.INFO) as records:
        proc.return_value.memory_info.return_value.rss = 500_000_000
        mem.check_memory(_ImageOnlyModel(), b)
    msgs = [r.getMessage() for r in records]
    assert recorded['target'] == 'IT'                        # declared target, not V1
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
