"""Structural tests for Lahner2024-fMRI-naturalistic-timeresolved-multimodal.

Skips anything that needs a real forward pass or the 11 GB TR-resolved
assembly download — those run on EC2, not local.

Validates: registry, mode validation, factory functions, voxel-mask
injection (auditory-ROI variant), null-control model registrations,
``_modality_per_feature`` tagging logic.
"""
import pytest

import brainscore


# ── Registry presence ─────────────────────────────────────────────


def test_tr_multimodal_benchmarks_registered():
    for ident in (
        'Lahner2024-fMRI-naturalistic-timeresolved-multimodal',
        'Lahner2024-fMRI-naturalistic-timeresolved-multimodal-visualROI',
        'Lahner2024-fMRI-naturalistic-timeresolved-multimodal-auditoryROI',
    ):
        assert ident in brainscore.benchmark_registry, (
            f"missing benchmark registration: {ident}")


def test_null_control_model_combos_registered():
    """All 4 signal/null permutations of vjepa1+wav2vec2."""
    for ident in (
        'vjepa1-wav2vec2',
        'random-vjepa1-wav2vec2',
        'vjepa1-random-wav2vec2',
        'random-vjepa1-random-wav2vec2',
    ):
        assert ident in brainscore.model_registry, (
            f"missing model registration: {ident}")


def test_random_wav2vec2_audio_null_registered():
    """Standalone audio-only null."""
    assert 'random-wav2vec2-base' in brainscore.model_registry


def test_clip_wav2vec2_combos_registered():
    """All 4 signal/null permutations for CLIP+Wav2Vec2."""
    for ident in (
        'clip-wav2vec2',
        'random-clip-wav2vec2',
        'clip-random-wav2vec2',
        'random-clip-random-wav2vec2',
    ):
        assert ident in brainscore.model_registry, (
            f"missing CLIP+Wav2Vec2 combo: {ident}")


def test_blip2_wav2vec2_registered():
    assert 'blip2-wav2vec2' in brainscore.model_registry


def test_qwen_vl_wav2vec2_registered():
    assert 'qwen2.5-vl-wav2vec2' in brainscore.model_registry


def test_total_multimodal_combo_count():
    """Sanity: 11 A+V combos total spanning V-JEPA / CLIP / BLIP-2 /
    Qwen-VL families plus the standalone random-Wav2Vec2 audio null."""
    combos = [k for k in brainscore.model_registry
              if 'wav2vec' in k.lower()]
    assert len(combos) == 11, (
        f"expected 11 wav2vec2-related registrations, got {len(combos)}: "
        f"{sorted(combos)}")


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


def test_visualROI_factory_uses_reliability():
    """visualROI factory should set the reliability threshold (0.3) and
    NOT install a custom voxel_mask_fn."""
    from brainscore.benchmarks.lahner2024.benchmark_timeresolved_multimodal \
        import (
            Lahner2024BOLDMoments_timeresolved_multimodal_visualROI)
    b = Lahner2024BOLDMoments_timeresolved_multimodal_visualROI()
    assert b._reliability_threshold == 0.3
    assert b._voxel_mask_fn is None


def test_auditoryROI_factory_installs_destrieux_mask():
    """auditoryROI factory should install the Destrieux mask function and
    leave reliability_threshold unset."""
    from brainscore.benchmarks.lahner2024.benchmark_timeresolved_multimodal \
        import (
            Lahner2024BOLDMoments_timeresolved_multimodal_auditoryROI)
    from brainscore.benchmarks.lahner2024.auditory_roi import (
        build_auditory_mask)
    b = Lahner2024BOLDMoments_timeresolved_multimodal_auditoryROI()
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
