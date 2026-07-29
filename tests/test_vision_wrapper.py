"""VisionWrapper facade: dispatch + delegation + escape hatch.

These verify the facade's job (pick the right strategy, delegate to it), not the
underlying extraction -- each strategy already has its own end-to-end tests.
"""
import pytest
import torch.nn as nn

from brainscore.model_helpers.vision_wrapper import VisionWrapper


def _tiny():
    return nn.Conv2d(3, 4, 3)


def _preproc(x):
    return x


class _VideoNet(nn.Module):
    def forward(self, x):
        return x


def test_defaults_to_frame_for_a_plain_model():
    w = VisionWrapper(_tiny(), _preproc, identifier='m')
    assert w.kind == 'frame'
    assert type(w.strategy).__name__ == 'PytorchWrapper'


def test_temporal_detected_from_class_name():
    w = VisionWrapper(_VideoNet(), _preproc, identifier='m')
    assert w.kind == 'temporal'
    assert type(w.strategy).__name__ == 'VideoWrapper'


def test_temporal_detected_from_a_temporal_kwarg():
    w = VisionWrapper(_tiny(), _preproc, identifier='m', num_frames=16)
    assert w.kind == 'temporal'


def test_vlm_detected_from_a_processor():
    w = VisionWrapper(_tiny(), processor=object(), identifier='m')
    assert w.kind == 'vlm'
    assert type(w.strategy).__name__ == 'VLMVisionWrapper'


def test_explicit_kind_overrides_auto_detection():
    # a temporal-looking model, but the user pins frame -> the override wins
    w = VisionWrapper(_VideoNet(), _preproc, identifier='m', kind='frame')
    assert w.kind == 'frame'
    assert type(w.strategy).__name__ == 'PytorchWrapper'


def test_vlm_kind_without_processor_errors_clearly():
    with pytest.raises(ValueError, match='processor'):
        VisionWrapper(_tiny(), _preproc, kind='vlm')


def test_invalid_kind_errors():
    with pytest.raises(ValueError, match='kind must be'):
        VisionWrapper(_tiny(), _preproc, kind='bogus')


def test_identifier_get_and_set_delegate_to_strategy():
    w = VisionWrapper(_tiny(), _preproc, identifier='m')
    assert w.identifier == 'm'
    w.identifier = 'renamed'
    assert w.strategy.identifier == 'renamed'


def test_unknown_attribute_delegates_to_strategy():
    w = VisionWrapper(_tiny(), _preproc, identifier='m')
    # PytorchWrapper stores the model; accessing it through the facade should work
    assert w._model is w.strategy._model
