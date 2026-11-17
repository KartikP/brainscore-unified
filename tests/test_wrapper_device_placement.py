"""Every wrapper keeps a model where a device map placed it.

A model too large for one accelerator is loaded with a device map: its layers
are spread over several devices, some possibly offloaded and left on ``meta``.
A wrapper must not move such a model, and must send its inputs to the device of
the model's parameters. A model on one device is moved exactly as before.
"""
import types

import pytest
import torch

pytestmark = pytest.mark.unit


class _Processor:
    sampling_rate = 16000

    def cache_config(self):
        return {'stand-in': 'processor'}


def _text(model, device=None):
    from brainscore.model_helpers.text_wrapper import TextWrapper

    class _Tokenizer:
        pad_token, eos_token = '<pad>', '<eos>'

        def cache_config(self):
            return {'stand-in': 'tokenizer'}
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


class _Plain(torch.nn.Module):
    """Real weights on the CPU; ``.to()`` works as torch's own."""

    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        self.first = torch.nn.Linear(4, 4)
        self.second = torch.nn.Linear(4, 4)


def _fingerprint(wrapper):
    from brainscore_core.extraction_cache import extraction_fingerprint
    return extraction_fingerprint({'configuration': wrapper.cache_config(), 'inputs': ['a']})


def _base_place_model(model, device=None):
    """The two lines every changed wrapper ran before the rule (base ``3688d83f``)."""
    from brainscore.model_helpers._device import select_device
    target = select_device(device)
    return model.to(target), target


@pytest.mark.parametrize('device', ['cpu', 'cpu:0'])
@pytest.mark.parametrize('make', WRAPPERS)
def test_a_model_on_one_device_keeps_its_base_cache_key(make, device, monkeypatch):
    tip = make(_Plain(), device=device)
    monkeypatch.setattr('brainscore.model_helpers._device.place_model', _base_place_model)
    base = make(_Plain(), device=device)
    assert _fingerprint(tip) is not None and _fingerprint(tip) == _fingerprint(base)
    assert str(tip._device) == str(base._device)


@pytest.mark.parametrize('device', [None, 'cpu', 'cpu:0'])
@pytest.mark.parametrize('make', WRAPPERS)
def test_a_model_placed_by_a_map_only_is_not_moved_and_gives_one_device_form(make, device):
    """A map on one device is a placement too; ``cpu`` and ``cpu:0`` give one key."""
    model = _StandIn(offloaded=False)
    model.hf_device_map = {'': 'cpu'}
    wrapper = make(model, device=device)
    assert not model.moved_to
    assert str(wrapper._device) == 'cpu'


# -- The device the inputs of a placed model go to ------------------------------

class _OffloadedFirst(torch.nn.Module):
    """A network whose first registered parameter is offloaded (``meta``); ``net`` takes the input.

    Records the devices of the tensors that reach the forward.
    """

    def __init__(self, net):
        super().__init__()
        self.offloaded = torch.nn.Linear(4, 4).to_empty(device='meta')
        self.net = net
        self.received = set()

    def forward(self, *args, **kwargs):
        self.received |= {value.device for value in (*args, *kwargs.values()) if torch.is_tensor(value)}
        return self.net(*args, **kwargs)


def _run_text(model, device=None):
    from tests.test_wrapper_cache_guard import _TextNet, _tokenizer
    from brainscore.model_helpers.text_wrapper import TextWrapper
    network = _OffloadedFirst(_TextNet())
    TextWrapper(network, _tokenizer(), identifier='stand-in', device=device)(
        ['the cat sat', 'dog ran'], layers=['net.proj'])
    return network


def _run_sound(device=None):
    from tests.test_wrapper_cache_guard import _AudioNet, _AudioProcessor, _audio_files, _load_npy, _stimuli
    from brainscore.model_helpers.audio_wrapper import AudioWrapper
    network = _OffloadedFirst(_AudioNet())
    AudioWrapper(network, _AudioProcessor(), identifier='stand-in', max_duration_sec=4.0,
                 audio_loader=_load_npy, device=device)(
        _stimuli('placed-sound', audio_path=_audio_files()), layers=['net.head'])
    return network


def _run_clip(device=None):
    from tests.test_wrapper_cache_guard import _VideoNet, _frames_to_tensor, _video_files, _stimuli
    from brainscore.model_helpers.video_wrapper import VideoWrapper
    network = _OffloadedFirst(_VideoNet())
    VideoWrapper(network, _frames_to_tensor, identifier='stand-in', num_frames=4, device=device)(
        _stimuli('placed-clip', video_path=_video_files()), layers=['net.head'])
    return network


def _run_picture(device=None):
    from tests.test_wrapper_cache_guard import _PatchNet, _PatchProcessor, _image_files, _stimuli
    from brainscore.model_helpers.vlm_vision_wrapper import VLMVisionWrapper
    network = _OffloadedFirst(_PatchNet())
    VLMVisionWrapper(network, _PatchProcessor(), identifier='stand-in', device=device)(
        _stimuli('placed-picture', image_file_name=_image_files()), layers=['net.blocks.0'])
    return network


RUNS = [lambda device=None: _run_text(None, device), _run_sound, _run_clip, _run_picture]


@pytest.mark.parametrize('device', [None, 'cpu:0'])
@pytest.mark.parametrize('run', RUNS)
def test_the_forward_of_a_placed_model_receives_inputs_on_its_input_device(run, device, tmp_path, monkeypatch):
    """Not on ``meta``, where an offloaded parameter registered first would send them."""
    monkeypatch.chdir(tmp_path)
    network = run(device)
    assert network.received == {torch.device('cpu')}
    assert network.offloaded.weight.device.type == 'meta'      # not moved


class _Spread(torch.nn.Module):
    """Reports parameters on the stated devices (no second device is needed on this machine)."""

    def __init__(self, devices, entry=None):
        super().__init__()
        self._devices, self._entry = [torch.device(d) for d in devices], entry

    def parameters(self, recurse=True):
        return iter([types.SimpleNamespace(device=d) for d in self._devices])

    def to(self, *args, **kwargs):
        raise AssertionError('a placed model is not moved')

    def get_input_embeddings(self):
        if self._entry is None:
            raise NotImplementedError
        return self._entry


def test_the_input_device_is_the_device_of_the_input_module():
    from brainscore.model_helpers._device import input_device
    assert str(input_device(_Spread(['cuda:0', 'cuda:1'], entry=_Spread(['cuda:1'])))) == 'cuda:1'


@pytest.mark.parametrize('devices', [['meta'], ['cuda:0', 'cuda:1']])
@pytest.mark.parametrize('make', WRAPPERS)
def test_an_input_device_that_cannot_be_told_is_refused_at_construction(make, devices):
    with pytest.raises(ValueError, match='State the input device'):
        make(_Spread(devices))


@pytest.mark.parametrize('make', WRAPPERS)
def test_a_stated_device_is_used_for_the_inputs_of_a_placed_model(make):
    model = _Spread(['cuda:0', 'cuda:1'])
    assert str(make(model, device='cuda:1')._device) == 'cuda:1'


def test_the_reply_path_sends_inputs_where_the_wrapper_does():
    """One rule for where inputs go: the reply path and a wrapper agree on one placed model."""
    from brainscore.model_helpers.causal_lm import _CausalLMParts
    from tests.test_wrapper_cache_guard import _TextNet, _tokenizer
    from brainscore.model_helpers.text_wrapper import TextWrapper
    network = _OffloadedFirst(_TextNet())
    parts = _CausalLMParts(network, _tokenizer())
    assert parts.device == TextWrapper(network, _tokenizer(), identifier='stand-in')._device == torch.device('cpu')


@pytest.mark.parametrize('placed', [True, False])
def test_a_stated_meta_device_is_refused_at_construction(placed):
    for make in WRAPPERS:
        model = _StandIn(offloaded=False)
        if placed:
            model.hf_device_map = {'': 'cpu'}
        with pytest.raises(ValueError, match="'meta' holds no values"):
            make(model, device='meta')
        assert not model.moved_to
