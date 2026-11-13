"""Structural tests for Lahner2024-fMRI-naturalistic-timeresolved-multimodal.

Skips anything that needs a real forward pass or the 11 GB TR-resolved
assembly download — those run on EC2, not local.

Validates: mode validation, factory functions, voxel-mask
injection (auditory-ROI variant),
``_modality_per_feature`` tagging logic.
"""
import pytest

import brainscore


# ── Mode validation ───────────────────────────────────────────────


def test_invalid_mode_raises():
    """The constructor rejects unknown modes early."""
    from brainscore.benchmarks.lahner2024.benchmark_timeresolved_multimodal \
        import Lahner2024BOLDMoments_timeresolved_multimodal
    with pytest.raises(ValueError, match="mode must be one of"):
        Lahner2024BOLDMoments_timeresolved_multimodal(mode='unknown')


def test_valid_modes_construct():
    """Each mode should construct a benchmark without I/O work
    (lazy-loaded assembly, voxel mask)."""
    from brainscore.benchmarks.lahner2024.benchmark_timeresolved_multimodal \
        import Lahner2024BOLDMoments_timeresolved_multimodal
    for mode in ('concat', 'per_modality', 'video_only', 'audio_only',
                 'banded'):
        b = Lahner2024BOLDMoments_timeresolved_multimodal(mode=mode)
        assert b._mode == mode


# ── Factory wiring ────────────────────────────────────────────────


def test_visualROI_registry_line_uses_reliability():
    """The visualROI registry line should set the reliability threshold (0.3) and
    NOT install a custom voxel_mask_fn."""
    import brainscore
    b = brainscore.benchmark_registry[
        'Lahner2024-fMRI-naturalistic-timeresolved-multimodal-visualROI']()
    assert b._reliability_threshold == 0.3
    assert b._voxel_mask_fn is None


def test_auditoryROI_registry_line_installs_destrieux_mask():
    """The auditoryROI registry line should install the Destrieux mask function and
    leave reliability_threshold unset."""
    import brainscore
    from brainscore.benchmarks.lahner2024.auditory_roi import (
        build_auditory_mask)
    b = brainscore.benchmark_registry[
        'Lahner2024-fMRI-naturalistic-timeresolved-multimodal-auditoryROI']()
    assert b._reliability_threshold is None
    assert b._voxel_mask_fn is build_auditory_mask


# ── Banded-ridge α grid contract ──────────────────────────────────


def test_banded_alpha_grid_logarithmic():
    """The banded grid must span at least four decades to give per-
    modality α tuning enough range to push uninformative modalities
    toward zero contribution."""
    from brainscore.benchmarks.lahner2024.benchmark_timeresolved_multimodal \
        import Lahner2024BOLDMoments_timeresolved_multimodal
    grid = Lahner2024BOLDMoments_timeresolved_multimodal.BANDED_ALPHA_GRID
    assert min(grid) <= 1.0
    assert max(grid) >= 10000.0
    assert len(grid) >= 5  # at least 5 points to interpolate


# ── Modality-tagging contract ─────────────────────────────────────


def test_modality_per_feature_unset_initially():
    """Before any feature extraction, the modality-per-feature array
    should be None — only the multimodal extraction populates it."""
    from brainscore.benchmarks.lahner2024.benchmark_timeresolved_multimodal \
        import Lahner2024BOLDMoments_timeresolved_multimodal
    b = Lahner2024BOLDMoments_timeresolved_multimodal()
    assert b._modality_per_feature is None
