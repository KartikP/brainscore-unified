"""A layer whose module runs more than once in one forward pass is refused, not cut to its last call.

A recording keeps one value for each layer. A shared or recurrent block that runs twice for one
input has no one value; the recording refuses it at the first input, before any score. Several
forward passes of the model for one input (generation, one pass for each token) stay allowed.
"""
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
