"""A layer whose module runs more than once in one forward pass is refused, not cut to its last call.

A recording keeps one value for each layer. A shared or recurrent block that runs twice for one
input has no one value; the recording refuses it at the first input, before any score. Several
forward passes of the model for one input (generation, one pass for each token) stay allowed.
"""
import copy

import pytest
import torch

from brainscore import BrainScoreModel
from brainscore.model_helpers.text_wrapper import TextWrapper
from brainscore_core.contract import TaskContext
from brainscore_core.responses import Text
from brainscore_core.streaming_helpers import stimulus_session
from brainscore_core.supported_data_standards.brainio.stimuli import StimulusSet
from tests.test_model_helpers.test_wrapper_cache_guard import _tokenizer

pytestmark = pytest.mark.unit


class _Looped(torch.nn.Module):
    """An embedding and one block; the block runs ``loops`` times in each forward pass."""

    def __init__(self, loops):
        super().__init__()
        self.loops = loops
        self.embed = torch.nn.Embedding(9, 4)
        self.block = torch.nn.Linear(4, 4)

    def forward(self, input_ids=None, attention_mask=None, **_):
        hidden = self.embed(input_ids)
        for _ in range(self.loops):
            hidden = self.block(hidden)
        return hidden


def _subject(loops, reply=None):
    network, tokenizer = _Looped(loops), _tokenizer()
    wrapper = TextWrapper(network, tokenizer, identifier=f'looped-{loops}', device='cpu')
    parts = {} if reply is None else {'reply': reply(network, tokenizer)}
    return BrainScoreModel(f'looped-{loops}', extractor={'text': wrapper},
                           region_layer_map={'R': 'block'}, **parts)


def _stimuli():
    stimuli = StimulusSet({'stimulus_id': ['a', 'b'], 'text': ['the cat sat', 'the dog ran']})
    stimuli.identifier = 'looped-texts'
    return stimuli


def _two_passes(network, tokenizer):
    """A reply that runs the model twice for one input, as generation does."""
    def reply(prompt):
        ids = torch.tensor([tokenizer('the cat')['input_ids']])
        network(input_ids=ids)
        network(input_ids=ids)
        return 'yes'
    return reply


def _passive(subject):
    session = stimulus_session(_stimuli(), ['neural:R'])
    subject.participate(session)
    return session


def _answering(subject):
    session = stimulus_session(_stimuli(), ['behavior:text', 'neural:R'], TaskContext(response=Text()))
    subject.participate(session)
    return session


@pytest.mark.parametrize('run', [_passive, _answering], ids=['passive', 'during-an-answer'])
def test_a_block_that_runs_twice_in_one_pass_is_refused_with_the_remedy(run):
    subject = _subject(loops=2, reply=_two_passes)
    with pytest.raises(ValueError, match='more than once in one forward pass') as error:
        run(subject)
    assert 'block' in str(error.value) and 'runs once' in str(error.value)


@pytest.mark.parametrize('run', [_passive, _answering], ids=['passive', 'during-an-answer'])
def test_a_block_that_runs_once_in_each_of_several_passes_records(run):
    session = run(_subject(loops=1, reply=_two_passes))
    assert session.collect('neural:R').sizes['presentation'] == 2


# One pass of a repeated module is named '<layer>-t<n>', counted from 0 as the old vision time steps.

class _Unrolled(torch.nn.Module):
    """The same network as ``_Looped(3)`` with each pass of the block as its own copy."""

    def __init__(self, looped):
        super().__init__()
        self.embed = looped.embed
        self.steps = torch.nn.ModuleList([copy.deepcopy(looped.block) for _ in range(3)])

    def forward(self, input_ids=None, attention_mask=None, **_):
        hidden = self.embed(input_ids)
        for step in self.steps:
            hidden = step(hidden)
        return hidden


def _recorded(network, layers, identifier):
    wrapper = TextWrapper(network, _tokenizer(), identifier=identifier, device='cpu')
    subject = BrainScoreModel(identifier, extractor={'text': wrapper},
                              region_layer_map={f'R{i}': layer for i, layer in enumerate(layers)})
    session = stimulus_session(_stimuli(), [f'neural:R{i}' for i in range(len(layers))])
    subject.participate(session)
    return [session.collect(f'neural:R{i}').values for i in range(len(layers))]


def test_each_pass_is_recorded_apart_and_equals_a_hook_on_that_call():
    torch.manual_seed(0)
    looped = _Looped(3)
    passes = _recorded(looped, ['block-t0', 'block-t1', 'block-t2'], 'looped-passes')
    calls = _recorded(_Unrolled(looped), ['steps.0', 'steps.1', 'steps.2'], 'unrolled-calls')
    for kept, hooked in zip(passes, calls):
        assert (kept == hooked).all()
    assert not (passes[0] == passes[1]).all()


def test_a_pass_above_the_count_is_refused_with_the_remedy():
    with pytest.raises(ValueError, match='-t0') as error:
        _recorded(_Looped(3), ['block-t3'], 'looped-above')
    assert 'block-t3' in str(error.value) and '3' in str(error.value)


def test_the_plain_name_of_a_repeated_module_names_the_pass_form_in_its_remedy():
    with pytest.raises(ValueError, match='more than once in one forward pass') as error:
        _recorded(_Looped(3), ['block'], 'looped-plain')
    assert '-t<n>' in str(error.value)


def test_a_once_run_module_takes_t0_as_its_one_call_and_refuses_t1():
    torch.manual_seed(0)
    once = _Looped(1)
    assert (_recorded(once, ['block-t0'], 'once-t0')[0] == _recorded(once, ['block'], 'once-plain')[0]).all()
    with pytest.raises(ValueError, match='block-t1'):
        _recorded(once, ['block-t1'], 'once-t1')


@pytest.mark.parametrize('layer', ['block-t1', 'block:input'])
def test_an_extractor_without_record_refuses_a_call_or_input_name_at_construction(layer):
    """The vision PytorchWrapper keeps one value for each module and reads no site."""
    from brainscore_vision.model_helpers.activations.pytorch import PytorchWrapper
    wrapper = PytorchWrapper(identifier='looped-vision', model=_Looped(2), preprocessing=lambda paths: paths)
    with pytest.raises(ValueError, match='one value for each module') as error:
        BrainScoreModel('looped-vision', extractor={'vision': wrapper}, region_layer_map={'R': layer})
    assert layer in str(error.value)


def test_a_wrapper_refuses_a_plain_data_frame():
    # A plain DataFrame iterates over its column names: 20 rows would give 2 stimuli.
    import pandas as pd
    wrapper = TextWrapper(_Looped(1), _tokenizer(), identifier='plain-frame', device='cpu')
    table = pd.DataFrame({'stimulus_id': [f's{i}' for i in range(20)], 'text': ['the cat sat'] * 20})
    with pytest.raises(TypeError, match='StimulusSet'):
        wrapper(table, ['block'])
