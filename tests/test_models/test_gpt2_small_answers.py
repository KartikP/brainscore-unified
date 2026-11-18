"""``gpt2-small`` answers with its readout and with its likelihood.

The registration is built through ``brainscore.load_model`` with stand-ins for the
network, the tokenizer and the text extractor; the likelihood part is the shipped
one (``_CausalLMParts``) over a small stand-in language model.
"""
from types import SimpleNamespace

import numpy as np
import pytest

import brainscore
from brainscore_core.contract import TaskContext
from brainscore_core.responses import Choice
from brainscore_core.streaming_helpers import stimulus_session
from tests.brainscore_model.test_answer_ownership import OPTIONS
from tests.brainscore_model.test_behavior_paths import FITTING, STIMULI, _features
from tests.test_benchmarks.test_ewok import native, prepared  # noqa: F401 (fixtures: synthetic EWoK items)

pytestmark = pytest.mark.unit

VOCABULARY = 31


class _Tokenizer:
    """One token for each character."""
    eos_token, pad_token, eos_token_id, pad_token_id = 'e', 'e', 0, 0

    def __call__(self, text, add_special_tokens=False):
        return {'input_ids': [1 + ord(character) % (VOCABULARY - 1) for character in text]}


def _language_model(forbidden):
    """A causal language model whose logits depend on the token ids; it has a ``transformer``.
    It fails if it runs when ``forbidden``."""
    import torch

    class LanguageModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(1))
            self.transformer = SimpleNamespace(name='transformer')

        def forward(self, input_ids, attention_mask=None):
            if forbidden:
                raise AssertionError('the likelihood ran')
            logits = torch.nn.functional.one_hot((input_ids * 7) % VOCABULARY, VOCABULARY).float()
            return SimpleNamespace(logits=logits + self.weight)
    return LanguageModel()


class _Extractor:
    """Stands in for the text extractor; it fails if it runs when ``forbidden``."""

    def __init__(self, forbidden, *args, **kwargs):
        self.forbidden, self.identifier = forbidden, kwargs.get('identifier', 'stand-in')

    def __call__(self, stimuli, layers, **kwargs):
        if self.forbidden:
            raise AssertionError('the readout extracted features')
        return _features(stimuli, layers, **kwargs)


def no_real_loads(monkeypatch):
    """Fail the test if a real checkpoint, tokenizer or the network is reached."""
    import socket
    from transformers import modeling_utils, tokenization_utils_base

    def refuse(*args, **kwargs):
        pytest.fail(f'a real loader or the network was reached: {args[:2]!r}')
    monkeypatch.setattr(socket.socket, 'connect', refuse)
    monkeypatch.setattr(modeling_utils.PreTrainedModel, 'from_pretrained', refuse)
    monkeypatch.setattr(tokenization_utils_base.PreTrainedTokenizerBase, 'from_pretrained', refuse)


def gpt2_stand_ins(monkeypatch, language_model, tokenizer):
    """The GPT-2 classes the registration loads give stand-ins (their own ``from_pretrained``)."""
    from transformers.models.gpt2 import modeling_gpt2, tokenization_gpt2
    no_real_loads(monkeypatch)
    monkeypatch.setattr(modeling_gpt2.GPT2LMHeadModel, 'from_pretrained', lambda *args, **kwargs: language_model())
    monkeypatch.setattr(tokenization_gpt2.GPT2Tokenizer, 'from_pretrained', lambda *args, **kwargs: tokenizer())


def _load(monkeypatch, forbid=None):
    """``brainscore.load_model('gpt2-small')`` with stand-ins; ``forbid`` names the part
    (``'likelihood'`` or ``'readout'``) that fails if it runs."""
    import brainscore.model_helpers.text_wrapper as text_wrapper
    gpt2_stand_ins(monkeypatch, lambda: _language_model(forbid == 'likelihood'), _Tokenizer)
    monkeypatch.setattr(text_wrapper, 'TextWrapper',
                        lambda *args, **kwargs: _Extractor(forbid == 'readout', *args, **kwargs))
    return brainscore.load_model('gpt2-small')


@pytest.fixture
def gpt2_small(monkeypatch):
    return _load(monkeypatch)


def test_declares_the_log_probabilities_and_keeps_its_readout_outputs(gpt2_small):
    assert {'neural:language_system', 'behavior:choice', 'behavior:probabilities', 'behavior:rating',
            'behavior:log_probabilities'} <= set(gpt2_small.out_channels)
    # The likelihood part takes the task's instruction.
    assert {'text', 'instruction'} <= set(gpt2_small.in_channels)


def test_a_lexical_decision_choice_is_answered_by_the_readout(monkeypatch):
    """The form of Yeatman2021-lexical_decision-text: an instruction, two labels, fitting stimuli."""
    task = TaskContext(instruction='Is this a real word or a pseudo word?', response=Choice(['cat', 'dog']),
                       fitting_stimuli=FITTING)
    session = stimulus_session(STIMULI, ['behavior:choice'], task)
    _load(monkeypatch, forbid='likelihood').participate(session)
    choice = session.collect('behavior:choice')
    assert choice.attrs['response']['method'] == 'readout'
    np.testing.assert_array_equal(choice.values, [[1, 0], [0, 1], [1, 0]])


def test_the_log_probabilities_of_a_verbatim_choice_come_from_the_likelihood(monkeypatch):
    session = stimulus_session(OPTIONS, ['behavior:log_probabilities'],
                               TaskContext(response=Choice('options', verbatim=True)))
    _load(monkeypatch, forbid='readout').participate(session)
    log_probabilities = session.collect('behavior:log_probabilities')
    assert log_probabilities.attrs['response']['method'] == 'likelihood'
    assert set(log_probabilities['response_state'].values) == {'valid'}
    assert np.all(log_probabilities.values < 0)


def test_ewok_log_probabilities_are_applicable_and_scored(gpt2_small, prepared, tmp_path, monkeypatch):  # noqa: F811
    monkeypatch.setenv('RESULTCACHING_HOME', str(tmp_path / 'cache'))
    benchmark = brainscore.load_benchmark('EWoK-core-1.0-logprobs', root=prepared, output_dir=tmp_path / 'scored')
    value = brainscore.score(gpt2_small, benchmark, check_mem=False)
    assert value.attrs.get('status') != 'N/A'
    assert 0 <= float(value) <= 1


def test_log_probabilities_of_a_choice_that_is_not_verbatim_are_refused_before_the_likelihood(monkeypatch):
    """GPT-2 has no chat template: it gives log probabilities only for a verbatim choice."""
    from brainscore_core.compatibility import ExecutionNotFeasibleError
    session = stimulus_session(OPTIONS, ['behavior:log_probabilities'],
                               TaskContext(instruction='Which?', response=Choice('options')))
    with pytest.raises(ExecutionNotFeasibleError, match='verbatim'):
        _load(monkeypatch, forbid='likelihood').participate(session)
