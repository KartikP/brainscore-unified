"""Tests for backbone-level feature caching across wrappers.

The wrappers (PytorchWrapper, TextWrapper, VLMVisionWrapper, VideoWrapper)
accept an optional ``backbone_id`` parameter that overrides the per-model
``identifier`` as the ``@store_xarray`` cache key. Two registrations that
share backbone weights can set the same backbone_id and reuse cached
activations across registrations.

These tests check the construction contract (the attribute each wrapper keeps).
That two registrations with one backbone_id share one cache entry is tested on a
real temporary cache in test_extraction_fingerprints.py.
"""

from unittest.mock import MagicMock

import pytest


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
