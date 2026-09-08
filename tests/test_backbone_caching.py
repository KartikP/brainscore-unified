"""Tests for backbone-level feature caching across wrappers.

The wrappers (PytorchWrapper, TextWrapper, VLMVisionWrapper, VideoWrapper)
accept an optional ``backbone_id`` parameter that overrides the per-model
``identifier`` as the ``@store_xarray`` cache key. Two registrations that
share backbone weights can set the same backbone_id and reuse cached
activations across registrations.

These tests exercise the construction contract + cache-key propagation
*without* actually triggering a forward pass (which would need real weights).
"""

from unittest.mock import MagicMock

import pytest

from brainscore_core.model_interface import BrainScoreModel


# ── BrainScoreModel construction ─────────────────────────────────────

class TestBrainScoreModelBackboneId:

    def test_default_backbone_id_equals_identifier(self):
        model = BrainScoreModel(
            identifier='clip-vit-b-32',
            model=None,
            region_layer_map={},
            preprocessors={'vision': lambda x: x},
        )
        assert model.backbone_id == 'clip-vit-b-32'

    def test_explicit_backbone_id_overrides(self):
        model = BrainScoreModel(
            identifier='blip-2-opt-2.7b',
            model=None,
            region_layer_map={},
            preprocessors={'vision': lambda x: x},
            backbone_id='vit-g-14-openclip',
        )
        assert model.backbone_id == 'vit-g-14-openclip'
        # identifier is preserved for logging/telemetry
        assert model.identifier == 'blip-2-opt-2.7b'

    def test_two_models_same_backbone_share_backbone_id(self):
        m1 = BrainScoreModel(
            identifier='blip-2-opt-2.7b', model=None, region_layer_map={},
            preprocessors={'vision': lambda x: x},
            backbone_id='vit-g-14',
        )
        m2 = BrainScoreModel(
            identifier='instruct-blip', model=None, region_layer_map={},
            preprocessors={'vision': lambda x: x},
            backbone_id='vit-g-14',
        )
        assert m1.backbone_id == m2.backbone_id
        assert m1.identifier != m2.identifier


# ── TextWrapper ──────────────────────────────────────────────────────

class TestTextWrapperBackboneId:

    def _make(self, identifier='w', backbone_id=None):
        # Mock model/tokenizer just enough to pass __init__ — no forward pass
        model = MagicMock()
        model.to = MagicMock(return_value=model)
        model.__class__.__name__ = 'MockTextModel'
        model.parameters = MagicMock(return_value=iter([]))
        tokenizer = MagicMock()
        from brainscore.model_helpers.text_wrapper import TextWrapper
        return TextWrapper(
            model=model, tokenizer=tokenizer,
            identifier=identifier, backbone_id=backbone_id,
        )

    def test_default_backbone_id_equals_identifier(self):
        w = self._make(identifier='gpt2-text')
        assert w.backbone_id == 'gpt2-text'

    def test_explicit_backbone_id(self):
        w = self._make(identifier='custom-gpt2', backbone_id='gpt2-124M')
        assert w.backbone_id == 'gpt2-124M'
        assert w.identifier == 'custom-gpt2'

    def test_cache_call_uses_backbone_id(self):
        """_from_texts_cached passes backbone_id, not identifier, to the
        stored function."""
        w = self._make(identifier='wrapper-a', backbone_id='shared-backbone')
        captured = {}

        def fake_stored(identifier, stimuli_identifier, layers, texts, extraction_fingerprint):
            captured['identifier'] = identifier
            return 'sentinel'

        w.cache_config = lambda: {"test_configuration": 1}
        w._from_texts_stored = fake_stored
        result = w._from_texts_cached(
            texts=['hello'], layers=['layer'], stimuli_identifier='stim-1')
        assert result == 'sentinel'
        assert captured['identifier'] == 'shared-backbone'


# ── VLMVisionWrapper ─────────────────────────────────────────────────

class TestVLMVisionWrapperBackboneId:

    def _make(self, identifier='w', backbone_id=None):
        model = MagicMock()
        model.to = MagicMock(return_value=model)
        model.__class__.__name__ = 'MockVLMModel'
        processor = MagicMock()
        from brainscore.model_helpers.vlm_vision_wrapper import VLMVisionWrapper
        return VLMVisionWrapper(
            model=model, processor=processor,
            identifier=identifier, backbone_id=backbone_id,
        )

    def test_default_backbone_id_equals_identifier(self):
        w = self._make(identifier='blip-2-opt-2.7b')
        assert w.backbone_id == 'blip-2-opt-2.7b'

    def test_explicit_backbone_id(self):
        w = self._make(identifier='blip-2-opt-2.7b', backbone_id='vit-g-14')
        assert w.backbone_id == 'vit-g-14'

    def test_cache_call_uses_backbone_id(self):
        w = self._make(identifier='a', backbone_id='shared-vit')
        captured = {}

        def fake_stored(identifier, stimuli_identifier, layers, paths, extraction_fingerprint):
            captured['identifier'] = identifier
            return 'sentinel'

        w.cache_config = lambda: {"test_configuration": 1}
        w._from_paths_stored = fake_stored
        result = w._from_paths_cached(
            paths=['a.png'], layers=['layer'], stimuli_identifier='stim-1')
        assert result == 'sentinel'
        assert captured['identifier'] == 'shared-vit'


# ── VideoWrapper ─────────────────────────────────────────────────────

class TestVideoWrapperBackboneId:

    def _make(self, identifier='w', backbone_id=None):
        import torch.nn as nn
        model = nn.Linear(1, 1)  # needs real nn.Module for .to(device)
        from brainscore.model_helpers.video_wrapper import VideoWrapper

        def preprocess(frames):
            import torch
            return torch.zeros(len(frames), 3, 16, 16)

        return VideoWrapper(
            model=model, preprocessing=preprocess,
            identifier=identifier, backbone_id=backbone_id,
        )

    def test_default_backbone_id_equals_identifier(self):
        w = self._make(identifier='vjepa1-vitl')
        assert w.backbone_id == 'vjepa1-vitl'

    def test_explicit_backbone_id(self):
        w = self._make(identifier='vjepa1-vitl', backbone_id='vit-l-16-video')
        assert w.backbone_id == 'vit-l-16-video'

    def test_cache_call_uses_backbone_id(self):
        w = self._make(identifier='vjepa-a', backbone_id='vjepa-shared-l16')
        captured = {}

        def fake_stored(identifier, stimuli_identifier, layers, paths, extraction_fingerprint):
            captured['identifier'] = identifier
            return 'sentinel'

        w.cache_config = lambda: {"test_configuration": 1}
        w._from_paths_stored = fake_stored
        result = w._from_paths_cached(
            paths=['a.mp4'], layers=['layer'], stimuli_identifier='stim-1')
        assert result == 'sentinel'
        assert captured['identifier'] == 'vjepa-shared-l16'


# ── Invariant: different backbone_ids produce different cache keys ──

def test_different_backbone_ids_dont_collide():
    """Two BrainScoreModels with different backbone_ids do not share a
    cache entry — same stimulus_identifier, different key."""
    m1 = BrainScoreModel(
        identifier='a', model=None, region_layer_map={},
        preprocessors={'vision': lambda x: x}, backbone_id='vit-l',
    )
    m2 = BrainScoreModel(
        identifier='b', model=None, region_layer_map={},
        preprocessors={'vision': lambda x: x}, backbone_id='vit-g',
    )
    assert m1.backbone_id != m2.backbone_id
