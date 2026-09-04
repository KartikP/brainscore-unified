"""TextWrapper must not relocate a model that accelerate has already placed.

A model too large for one accelerator is loaded with ``device_map='auto'``, which
may leave layers offloaded to host memory or on the meta device. Calling ``.to()``
on such a model raises "Cannot copy out of meta tensor", which is what blocked
registering a 27B model through the ordinary interface.
"""

import pytest
import torch

pytestmark = pytest.mark.unit


class _Tokenizer:
    pad_token = '<pad>'
    eos_token = '<eos>'


class _Mapped(torch.nn.Module):
    """Stands in for an accelerate-placed model."""

    def __init__(self):
        super().__init__()
        self.layers = torch.nn.ModuleList([torch.nn.Linear(4, 4)])
        self.hf_device_map = {'layers.0': 0, 'layers.1': 'cpu'}
        self.moved = False

    def to(self, *args, **kwargs):
        self.moved = True
        raise NotImplementedError('Cannot copy out of meta tensor; no data!')


class _Plain(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.ModuleList([torch.nn.Linear(4, 4)])
        self.moved = False

    def to(self, *args, **kwargs):
        self.moved = True
        return self


def test_device_mapped_model_is_not_moved():
    from brainscore.model_helpers.text_wrapper import TextWrapper

    model = _Mapped()
    wrapper = TextWrapper(model=model, tokenizer=_Tokenizer(), identifier='mapped')
    assert not model.moved, 'a device-mapped model must be left where accelerate put it'
    assert wrapper._device == next(model.parameters()).device


def test_plain_model_is_still_moved():
    """The ordinary path is unchanged: models without a map are placed as before."""
    from brainscore.model_helpers.text_wrapper import TextWrapper

    model = _Plain()
    TextWrapper(model=model, tokenizer=_Tokenizer(), identifier='plain')
    assert model.moved


class _SubmoduleOfMapped(torch.nn.Module):
    """A submodule of a device-mapped model: no hf_device_map of its own.

    This is the realistic case — registrations hand the wrapper the text tower,
    while accelerate recorded its plan on the parent.
    """

    def __init__(self, devices):
        super().__init__()
        self.embed_tokens = torch.nn.Embedding(8, 4)
        self.layers = torch.nn.ModuleList([torch.nn.Linear(4, 4) for _ in devices])
        self.moved = False

    def to(self, *args, **kwargs):
        self.moved = True
        raise NotImplementedError('Cannot copy out of meta tensor; no data!')


def test_submodule_with_meta_parameters_is_not_moved():
    """Offloaded layers show up as meta tensors and must be left alone."""
    from brainscore.model_helpers.text_wrapper import TextWrapper

    model = _SubmoduleOfMapped(['cpu', 'meta'])
    model.layers[1].to_empty(device='meta')
    assert not hasattr(model, 'hf_device_map')      # the realistic situation
    TextWrapper(model=model, tokenizer=_Tokenizer(), identifier='submodule')
    assert not model.moved


def test_inputs_go_to_the_embedding_device():
    """Inputs must land where input_ids are consumed, not on an arbitrary layer."""
    from brainscore.model_helpers.text_wrapper import TextWrapper

    model = _SubmoduleOfMapped(['cpu', 'meta'])
    model.layers[1].to_empty(device='meta')
    wrapper = TextWrapper(model=model, tokenizer=_Tokenizer(), identifier='sub2')
    assert wrapper._device == next(model.embed_tokens.parameters()).device


def test_hook_handles_bfloat16_activations():
    """Large models load in bfloat16, which numpy cannot represent.

    Without promotion the forward hook raises "Got unsupported ScalarType
    BFloat16", which blocked recording from a 27B model entirely.
    """
    import numpy as np
    from brainscore.model_helpers.text_wrapper import TextWrapper

    net = torch.nn.Sequential(torch.nn.Linear(4, 6)).to(torch.bfloat16)
    wrapper = TextWrapper(model=net, tokenizer=_Tokenizer(), identifier='bf16')
    store = {}
    handle = wrapper._register_hook(wrapper._model[0], 'l0', store)
    wrapper._model(torch.randn(2, 4, dtype=torch.bfloat16, device=wrapper._device))
    handle.remove()

    assert store['l0'].dtype == np.float32
    assert store['l0'].shape == (2, 6)
    assert np.isfinite(store['l0']).all()
