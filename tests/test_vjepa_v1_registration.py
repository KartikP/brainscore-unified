"""Unit tests for V-JEPA v1 ViT-L/16 registration.

Covers the pieces that don't require downloading the ~1.2GB checkpoint:
- registry entry is populated
- post-hook reshape produces the expected (B, 8, 1024) layout
- frame padding handles short/long/exact/empty videos
- permute adapter flips (B, T, C, H, W) -> (B, C, T, H, W)
- constants match Meta's ViT-L vitl16 checkpoint layout
"""

import numpy as np
import pytest
import torch
import torch.nn as nn

import brainscore
from brainscore.models.vjepa_v1.model import (
    REGION_LAYER_MAP,
    NUM_INPUT_FRAMES,
    NUM_TEMPORAL_STEPS,
    NUM_SPATIAL_PATCHES,
    PATCH_SIZE,
    IMG_SIZE,
    _vjepa_v1_post_hook,
    _make_preprocessing,
    _BCTPermuteAdapter,
)


def test_vjepa_v1_registered():
    assert 'vjepa1-vitl' in brainscore.model_registry


def test_constants_match_vitl16():
    assert NUM_INPUT_FRAMES == 16
    assert NUM_TEMPORAL_STEPS == 8  # tubelet_size=2
    assert NUM_SPATIAL_PATCHES == 196  # (224/16)^2
    assert PATCH_SIZE == 16
    assert IMG_SIZE == 224
    assert REGION_LAYER_MAP['IT'] == 'backbone.blocks.16'  # per-voxel sweep peak
    for region in ('V1', 'V2', 'V4', 'IT'):
        assert region in REGION_LAYER_MAP


def test_post_hook_default_seqlen():
    B, H = 2, 1024
    seq_len = NUM_TEMPORAL_STEPS * NUM_SPATIAL_PATCHES  # 1568
    arr = np.random.randn(B, seq_len, H).astype(np.float32)
    out = _vjepa_v1_post_hook(arr)
    assert out.shape == (B, NUM_TEMPORAL_STEPS, H)


def test_post_hook_spatial_mean_semantics():
    B, H = 1, 8
    arr = np.zeros((B, NUM_TEMPORAL_STEPS * NUM_SPATIAL_PATCHES, H),
                   dtype=np.float32)
    reshaped = arr.reshape(B, NUM_TEMPORAL_STEPS, NUM_SPATIAL_PATCHES, H)
    for t in range(NUM_TEMPORAL_STEPS):
        reshaped[:, t, :, :] = float(t)
    out = _vjepa_v1_post_hook(reshaped.reshape(B, -1, H))
    for t in range(NUM_TEMPORAL_STEPS):
        assert np.allclose(out[0, t, :], float(t))


def test_post_hook_preserves_temporal_order():
    B, H = 1, 16
    shape = (B, NUM_TEMPORAL_STEPS, NUM_SPATIAL_PATCHES, H)
    a = np.random.randn(*shape).astype(np.float32)
    b = a[:, ::-1, :, :].copy()
    out_a = _vjepa_v1_post_hook(a.reshape(B, -1, H))
    out_b = _vjepa_v1_post_hook(b.reshape(B, -1, H))
    assert not np.allclose(out_a, out_b)
    assert np.allclose(out_a[:, ::-1, :], out_b)


def test_post_hook_bad_seqlen_raises():
    arr = np.random.randn(1, 13, 8).astype(np.float32)  # prime, not a multiple
    with pytest.raises(ValueError, match="unable to infer"):
        _vjepa_v1_post_hook(arr)


def test_preprocess_pads_short_videos_to_16():
    preprocess = _make_preprocessing()
    short = [np.zeros((112, 112, 3), dtype=np.uint8) for _ in range(10)]
    out = preprocess(short)
    assert out.shape == (NUM_INPUT_FRAMES, 3, IMG_SIZE, IMG_SIZE)
    # Long — 45 frames (common Lahner short-clip case)
    long_ = [np.zeros((180, 180, 3), dtype=np.uint8) for _ in range(45)]
    assert preprocess(long_).shape == (NUM_INPUT_FRAMES, 3, IMG_SIZE, IMG_SIZE)
    # Exact — 16 frames
    exact = [np.zeros((224, 224, 3), dtype=np.uint8) for _ in range(16)]
    assert preprocess(exact).shape == (NUM_INPUT_FRAMES, 3, IMG_SIZE, IMG_SIZE)
    with pytest.raises(ValueError, match="empty frame list"):
        preprocess([])


def test_preprocess_normalization_and_dtype():
    preprocess = _make_preprocessing()
    # All-127 uint8 should map to near-zero after ImageNet mean/std norm
    frames = [np.full((200, 200, 3), 127, dtype=np.uint8) for _ in range(16)]
    out = preprocess(frames)
    assert out.dtype == torch.float32
    # 127/255 = 0.498; (0.498 - mean) / std is small in magnitude
    assert float(out.abs().max()) < 2.0
    # shape (T, C, H, W) with C=3
    assert out.shape[1] == 3


def test_bct_permute_adapter_flips_axes():
    """Verify adapter permutes (B,T,C,H,W) -> (B,C,T,H,W) before backbone."""
    captured = {}

    class _Spy(nn.Module):
        def forward(self, x, *args, **kwargs):
            captured['shape'] = tuple(x.shape)
            # return something with a predictable shape so hooks don't break
            B = x.shape[0]
            return torch.zeros(B, 1568, 1024)

    adapter = _BCTPermuteAdapter(_Spy())
    btc_input = torch.randn(2, 16, 3, 224, 224)  # (B, T, C, H, W)
    _ = adapter(btc_input)
    assert captured['shape'] == (2, 3, 16, 224, 224)  # (B, C, T, H, W)


def test_bct_permute_adapter_preserves_values():
    """Permute should be axis-permutation only — no dropped/added data."""
    captured = {}

    class _Spy(nn.Module):
        def forward(self, x, *args, **kwargs):
            captured['x'] = x
            return torch.zeros(x.shape[0], 1568, 1024)

    adapter = _BCTPermuteAdapter(_Spy())
    btc = torch.arange(2 * 16 * 3 * 4 * 4).float().reshape(2, 16, 3, 4, 4)
    adapter(btc)
    expected_bct = btc.permute(0, 2, 1, 3, 4)
    assert torch.equal(captured['x'], expected_bct)
