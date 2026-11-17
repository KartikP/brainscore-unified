"""Every wrapper keeps a model where a device map placed it.

A model too large for one accelerator is loaded with a device map: its layers
are spread over several devices, some possibly offloaded and left on ``meta``.
A wrapper must not move such a model, and must send its inputs to the device of
the model's parameters. A model on one device is moved exactly as before.
"""
import pytest
import torch

pytestmark = pytest.mark.unit


class _Processor:
    sampling_rate = 16000


def _text(model, device=None):
    from brainscore.model_helpers.text_wrapper import TextWrapper

    class _Tokenizer:
        pad_token, eos_token = '<pad>', '<eos>'
    return TextWrapper(model, _Tokenizer(), identifier='stand-in', device=device)


def _picture(model, device=None):
    from brainscore.model_helpers.vlm_vision_wrapper import VLMVisionWrapper
    return VLMVisionWrapper(model, _Processor(), identifier='stand-in', device=device)


def _clip(model, device=None):
    from brainscore.model_helpers.video_wrapper import VideoWrapper
    return VideoWrapper(model, preprocessing=lambda frames: None, identifier='stand-in',
                        device=device)


def _sound(model, device=None):
    from brainscore.model_helpers.audio_wrapper import AudioWrapper
    return AudioWrapper(model, _Processor(), identifier='stand-in', device=device)


WRAPPERS = [_text, _picture, _clip, _sound]


class _StandIn(torch.nn.Module):
    """Two layers; ``.to()`` is recorded, and refused when a layer is on ``meta``."""

    def __init__(self, offloaded: bool):
        super().__init__()
        self.first = torch.nn.Linear(4, 4)
        self.second = torch.nn.Linear(4, 4)
        if offloaded:
            self.second.to_empty(device='meta')
        self.moved_to = []

    def to(self, *args, **kwargs):
        if any(p.device.type == 'meta' for p in self.parameters()):
            raise NotImplementedError('Cannot copy out of meta tensor; no data!')
        self.moved_to.append(args[0] if args else kwargs.get('device'))
        return self


@pytest.mark.parametrize('make', WRAPPERS)
def test_a_placed_model_is_not_moved_and_inputs_follow_its_parameters(make):
    model = _StandIn(offloaded=True)
    wrapper = make(model)
    assert wrapper._model is model and not model.moved_to
    assert wrapper._device == model.first.weight.device


@pytest.mark.parametrize('make', WRAPPERS)
def test_a_model_on_one_device_is_moved_as_before(make):
    """The two lines this rule replaced: ``select_device(device)``, then ``model.to`` it."""
    from brainscore.model_helpers._device import select_device
    model = _StandIn(offloaded=False)
    wrapper = make(model, device='cpu')
    assert model.moved_to == [select_device('cpu')]
    assert wrapper._device == select_device('cpu')
