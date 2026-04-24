"""Unit tests for V-JEPA2 registration.

Covers the pieces that don't require downloading the ~1.2GB checkpoint:
- registry entry is populated
- post-hook reshape produces the expected (B, T, H) shape
- shape inference fallback handles non-default seq_lens
- wiring constants match the ViT-L fpc64-256 checkpoint

Full end-to-end scoring on Lahner2024 is run on EC2 (not in unit tests).
"""

import numpy as np
import pytest

import brainscore
from brainscore.models.vjepa.model import (
    REGION_LAYER_MAP,
    NUM_INPUT_FRAMES,
    NUM_TEMPORAL_STEPS,
    NUM_SPATIAL_PATCHES,
    _vjepa_post_hook,
)


def test_vjepa_registered():
    assert 'vjepa2-vitl' in brainscore.model_registry


def test_vjepa_constants():
    assert NUM_INPUT_FRAMES == 64
    assert NUM_TEMPORAL_STEPS == 32  # tubelet_size=2
    assert NUM_SPATIAL_PATCHES == 256  # (256/16)**2
    # Sanity: ViT-L layer count must accommodate the IT mapping.
    assert REGION_LAYER_MAP['IT'] == 'encoder.layer.16'  # aligned with v1 sweep peak
    for region in ('V1', 'V2', 'V4', 'IT'):
        assert region in REGION_LAYER_MAP


def test_post_hook_default_seqlen():
    B, H = 2, 1024
    seq_len = NUM_TEMPORAL_STEPS * NUM_SPATIAL_PATCHES  # 8192
    arr = np.random.randn(B, seq_len, H).astype(np.float32)
    out = _vjepa_post_hook(arr)
    assert out.shape == (B, NUM_TEMPORAL_STEPS, H)


def test_post_hook_spatial_mean_semantics():
    """Constant along the spatial axis should survive mean unchanged."""
    B, H = 1, 8
    arr = np.zeros((B, NUM_TEMPORAL_STEPS * NUM_SPATIAL_PATCHES, H),
                   dtype=np.float32)
    # For temporal step t, fill all spatial patches with value t.
    reshaped = arr.reshape(B, NUM_TEMPORAL_STEPS, NUM_SPATIAL_PATCHES, H)
    for t in range(NUM_TEMPORAL_STEPS):
        reshaped[:, t, :, :] = float(t)
    flat = reshaped.reshape(B, -1, H)
    out = _vjepa_post_hook(flat)
    for t in range(NUM_TEMPORAL_STEPS):
        assert np.allclose(out[0, t, :], float(t))


def test_post_hook_preserves_temporal_order():
    """Two videos with permuted temporal content must produce different outputs."""
    B, H = 1, 16
    shape = (B, NUM_TEMPORAL_STEPS, NUM_SPATIAL_PATCHES, H)
    a = np.random.randn(*shape).astype(np.float32)
    b = a[:, ::-1, :, :].copy()  # reverse time
    out_a = _vjepa_post_hook(a.reshape(B, -1, H))
    out_b = _vjepa_post_hook(b.reshape(B, -1, H))
    assert not np.allclose(out_a, out_b)
    # And reversing out_a should match out_b
    assert np.allclose(out_a[:, ::-1, :], out_b)


def test_post_hook_inferred_seqlen_spatial_multiple():
    """If seq_len is a multiple of NUM_SPATIAL_PATCHES but not the expected
    product, post-hook should infer temporal from spatial."""
    B, H = 1, 32
    t_steps = 16  # different from default 32
    seq_len = t_steps * NUM_SPATIAL_PATCHES
    arr = np.random.randn(B, seq_len, H).astype(np.float32)
    out = _vjepa_post_hook(arr)
    assert out.shape == (B, t_steps, H)


def test_post_hook_bad_seqlen_raises():
    arr = np.random.randn(1, 13, 8).astype(np.float32)  # prime, not a multiple
    with pytest.raises(ValueError, match="unable to infer"):
        _vjepa_post_hook(arr)


def test_preprocess_pads_short_videos_to_64():
    """Lahner2024 has some 45-frame clips at 15 FPS. V-JEPA expects 64 frames
    per video, and variable T across videos breaks the cross-video stack in
    VideoWrapper._package. Preprocessing must pad to exactly NUM_INPUT_FRAMES.
    """
    import torch
    from brainscore.models.vjepa.model import _make_preprocessing

    class _FakeProcessor:
        """Mimics VJEPA2VideoProcessor — returns (1, T, C, H, W)."""
        def __call__(self, frames, return_tensors='pt'):
            T = len(frames)
            return {'pixel_values_videos': torch.zeros(1, T, 3, 256, 256)}

    preprocess = _make_preprocessing(_FakeProcessor())
    # Short video: 45 frames
    short = [np.zeros((224, 224, 3), dtype=np.uint8) for _ in range(45)]
    out_short = preprocess(short)
    assert out_short.shape == (NUM_INPUT_FRAMES, 3, 256, 256)
    # Long video: 181 frames
    long_ = [np.zeros((224, 224, 3), dtype=np.uint8) for _ in range(181)]
    out_long = preprocess(long_)
    assert out_long.shape == (NUM_INPUT_FRAMES, 3, 256, 256)
    # Exact video: 64 frames — unchanged
    exact = [np.zeros((224, 224, 3), dtype=np.uint8) for _ in range(64)]
    out_exact = preprocess(exact)
    assert out_exact.shape == (NUM_INPUT_FRAMES, 3, 256, 256)
    # Empty → error
    with pytest.raises(ValueError, match="empty frame list"):
        preprocess([])
