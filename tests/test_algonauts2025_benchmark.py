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


def test_execution_plan_declares_TR_cardinality_and_metric_cap():
    """The benchmark declares a reliable ExecutionPlan for the memory pre-flight:
    one extraction row per TR (not per video), a metric-width cap of
    stimulus_window x FEATURE_DIM_CAP, and the raw width left to the probe."""
    from brainscore.benchmarks.algonauts2025.benchmark import (
        Algonauts2025Friends, _Algonauts2025Base)
    from brainscore_core.execution_plan import ExecutionPlan

    class _MockAssembly:
        def __init__(self, n_trs):
            self._n = n_trs
        def __getitem__(self, key):
            assert key == 'stimulus_id'
            return list(range(self._n))

    b = Algonauts2025Friends(subject=1)
    b._assembly = _MockAssembly(162_671)  # avoid loading real data
    plan = b.execution_plan
    assert isinstance(plan, ExecutionPlan)
    assert plan.n_extraction_presentations == 162_671          # TRs, not videos
    assert plan.feature_width is None                          # probed (model-dependent)
    assert plan.metric_feature_width == (
        b._stimulus_window * _Algonauts2025Base.FEATURE_DIM_CAP)  # 5 * 1000
    assert plan.resolved_metric_observations == 162_671        # metric fits per-TR


def test_check_memory_takes_the_reliable_path_on_algonauts():
    """check_memory reads the declared plan and reports a RELIABLE (not
    APPROXIMATE) estimate, driven by the TR count rather than the video count."""
    import logging
    from unittest.mock import patch
    from brainscore.benchmarks.algonauts2025.benchmark import Algonauts2025Friends
    from brainscore_core import memory as mem

    class _MockAssembly:
        def __init__(self, n_trs):
            self._n = n_trs
        def __getitem__(self, key):
            return list(range(self._n))
        def __len__(self):
            return self._n

    class _TinyModel:
        identifier = 'tiny'
        region_layer_map = {}
        def process(self, stimuli):
            return type('R', (), {'shape': (1, 768)})()

    b = Algonauts2025Friends(subject=1)
    b._assembly = _MockAssembly(162_671)
    b._stimulus_set = _MockAssembly(300)  # 300 videos — the WRONG count to size off

    with patch.object(mem, 'get_host_available_memory', return_value=64_000_000_000), \
         patch('psutil.Process') as proc, \
         mem_caplog(logging.INFO) as records:
        proc.return_value.memory_info.return_value.rss = 500_000_000
        mem.check_memory(_TinyModel(), b)
    msgs = [r.getMessage() for r in records]
    assert any('RELIABLE' in m for m in msgs)
    assert any('162671 presentations' in m for m in msgs)  # TR count, not 300 videos
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
