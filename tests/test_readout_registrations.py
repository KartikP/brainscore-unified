"""The registered readout-only models give choices in an experiment (stage 2, C3 review Y01).

Each test runs the shipped builder body of CLIP, random ViT or GPT-2 with only
the network, tokenizer, processor and extractors replaced by stand-ins, then
runs an ``Experiment`` with ``TaskProtocol``, a ``Choice`` task with fitting
stimuli and ``RecordBehavior``. The readout derives the choice by argmax, as
the registration states.
"""
from types import SimpleNamespace

import numpy as np
import pytest

from brainscore_core.contract import TaskContext
from brainscore_core.responses import Choice
from brainscore.experiments import Experiment, RecordBehavior, TaskProtocol
from tests.brainscore_model.test_behavior_paths import FITTING, STIMULI, _features

pytestmark = pytest.mark.unit


class _Extractor:
    """Stands in for a wrapper: the features of test_behavior_paths."""

    def __init__(self, *args, **kwargs):
        self.identifier = kwargs.get('identifier', 'stand-in')

    def __call__(self, stimuli, layers, **kwargs):
        return _features(stimuli, layers, **kwargs)


def _network(*args, **kwargs):
    return SimpleNamespace(vision_model=object(), text_model=object())


def _loader(make):
    return SimpleNamespace(from_pretrained=lambda *args, **kwargs: make())


def _clip(monkeypatch, module_name):
    import importlib
    module = importlib.import_module(module_name)
    monkeypatch.setattr(module, 'CLIPProcessor', _loader(lambda: SimpleNamespace(tokenizer=object())))
    monkeypatch.setattr(module, 'pin_image_processor', lambda processor, name: processor)
    monkeypatch.setattr(module, 'PytorchWrapper', _Extractor)
    monkeypatch.setattr(module, 'TextWrapper', _Extractor)
    if module_name.endswith('random_vit_b_32.model'):
        monkeypatch.setattr(module, 'CLIPConfig', _loader(object))
        monkeypatch.setattr(module, 'CLIPModel', _network)
    else:
        monkeypatch.setattr(module, 'CLIPModel', _loader(_network))
    return module


def _gpt2(monkeypatch):
    import transformers
    import brainscore.model_helpers.text_wrapper as text_wrapper
    from brainscore.models.gpt2 import model as module
    monkeypatch.setattr(transformers, 'GPT2Model', _loader(object), raising=False)
    monkeypatch.setattr(transformers, 'GPT2Tokenizer', _loader(lambda: SimpleNamespace(eos_token='e')),
                        raising=False)
    monkeypatch.setattr(text_wrapper, 'TextWrapper', _Extractor)
    return module


BUILDERS = {
    'clip-vit-b-32': lambda mp: _clip(mp, 'brainscore.models.clip_vit_b_32.model'),
    'random-vit-b-32': lambda mp: _clip(mp, 'brainscore.models.random_vit_b_32.model'),
    'gpt2': _gpt2,
}


@pytest.mark.parametrize('identifier', sorted(BUILDERS))
def test_a_registered_readout_model_gives_choices_in_an_experiment(identifier, monkeypatch, tmp_path):
    subject = BUILDERS[identifier](monkeypatch).get_model(identifier)
    assert 'behavior:choice' in subject.out_channels
    task = TaskContext(instruction='Which?', response=Choice(['cat', 'dog']), fitting_stimuli=FITTING)
    result = Experiment(subject=subject, protocol=TaskProtocol(STIMULI, task),
                        tools=[RecordBehavior()], output_dir=tmp_path / identifier).run()
    choice = result.behavior(measure='choice')
    np.testing.assert_array_equal(choice.values, [[1, 0], [0, 1], [1, 0]])
    record = choice.attrs['response']
    assert (record['origin'], record['conversion'], record['parameters']['option_order']) == \
        ('derived', 'argmax', 'readout')
