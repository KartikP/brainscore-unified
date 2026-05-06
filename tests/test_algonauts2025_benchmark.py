"""Structural tests for the Algonauts 2025 benchmark scaffold.

Data is on EC2 (~100 GB DataLad pull); these tests don't load any of
it. They verify:
- All 12 (subject × split) registry entries exist.
- Constructor validates subject ∈ {1, 2, 3, 5} and mode ∈ VALID_MODES.
- ``__call__`` raises NotImplementedError until the assembly
  preparation pipeline runs on EC2.
- ``_load_assembly`` raises FileNotFoundError with a clear pointer to
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


def test_call_raises_not_implemented():
    """Until the actual scoring pipeline is built, __call__ must
    refuse to silently produce a wrong result."""
    from brainscore.benchmarks.algonauts2025.benchmark import (
        Algonauts2025Friends)
    b = Algonauts2025Friends(subject=1)
    with pytest.raises(NotImplementedError, match="scoring not yet"):
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
