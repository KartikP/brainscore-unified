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
from tests.test_ewok import native, prepared  # noqa: F401 (fixtures: synthetic EWoK items)

pytestmark = pytest.mark.unit

VOCABULARY = 31


class _Tokenizer:
    """One token for each character."""
    eos_token, pad_token, eos_token_id, pad_token_id = 'e', 'e', 0, 0

    def __call__(self, text, add_special_tokens=False):
        return {'input_ids': [1 + ord(character) % (VOCABULARY - 1) for character in text]}


def _language_model(calls):
    """A causal language model whose logits depend on the token ids; it has a ``transformer``."""
    import torch

    class LanguageModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(1))
            self.transformer = SimpleNamespace(name='transformer')

        def forward(self, input_ids, attention_mask=None):
            calls.append('likelihood')
            logits = torch.nn.functional.one_hot((input_ids * 7) % VOCABULARY, VOCABULARY).float()
            return SimpleNamespace(logits=logits + self.weight)
    return LanguageModel()


class _Extractor:
    """Stands in for the text extractor; records what it is asked to extract."""

    def __init__(self, calls, *args, **kwargs):
        self.calls, self.identifier = calls, kwargs.get('identifier', 'stand-in')

    def __call__(self, stimuli, layers, **kwargs):
        self.calls.append(('extract', list(stimuli['stimulus_id']), list(layers)))
        return _features(stimuli, layers, **kwargs)


@pytest.fixture
def gpt2_small(monkeypatch):
    """``brainscore.load_model('gpt2-small')`` with stand-ins; its calls in ``.calls``."""
    import transformers
    import brainscore.model_helpers.text_wrapper as text_wrapper
    calls = []
    monkeypatch.setattr(transformers, 'GPT2LMHeadModel',
                        SimpleNamespace(from_pretrained=lambda *args, **kwargs: _language_model(calls)),
                        raising=False)
    monkeypatch.setattr(transformers, 'GPT2Tokenizer',
                        SimpleNamespace(from_pretrained=lambda *args, **kwargs: _Tokenizer()), raising=False)
    monkeypatch.setattr(text_wrapper, 'TextWrapper', lambda *args, **kwargs: _Extractor(calls, *args, **kwargs))
    subject = brainscore.load_model('gpt2-small')
    subject.calls = calls
    return subject


def test_declares_the_log_probabilities_and_keeps_its_readout_outputs(gpt2_small):
    assert {'neural:language_system', 'behavior:choice', 'behavior:probabilities', 'behavior:rating',
            'behavior:log_probabilities'} <= set(gpt2_small.out_channels)
    # The likelihood part takes the task's instruction; GPT-2 does not write.
    assert {'text', 'instruction'} <= set(gpt2_small.in_channels)
    assert 'behavior:text' not in gpt2_small.out_channels


def test_a_lexical_decision_choice_is_answered_by_the_readout(gpt2_small):
    """The form of Yeatman2021-lexical_decision-text: an instruction, two labels, fitting stimuli."""
    task = TaskContext(instruction='Is this a real word or a pseudo word?', response=Choice(['cat', 'dog']),
                       fitting_stimuli=FITTING)
    session = stimulus_session(STIMULI, ['behavior:choice'], task)
    gpt2_small.participate(session)
    choice = session.collect('behavior:choice')
    assert choice.attrs['response']['method'] == 'readout'
    np.testing.assert_array_equal(choice.values, [[1, 0], [0, 1], [1, 0]])
    # The readout's fit and its answer: two extractions at its layer, no likelihood pass.
    assert gpt2_small.calls == [('extract', list(FITTING['stimulus_id']), ['h.11']),
                                ('extract', list(STIMULI['stimulus_id']), ['h.11'])]


def test_the_log_probabilities_of_a_verbatim_choice_come_from_the_likelihood(gpt2_small):
    session = stimulus_session(OPTIONS, ['behavior:log_probabilities'],
                               TaskContext(response=Choice('options', verbatim=True)))
    gpt2_small.participate(session)
    log_probabilities = session.collect('behavior:log_probabilities')
    assert log_probabilities.attrs['response']['method'] == 'likelihood'
    assert set(log_probabilities['response_state'].values) == {'valid'}
    assert np.all(log_probabilities.values < 0)
    assert 'likelihood' in gpt2_small.calls and not any(call[0] == 'extract' for call in gpt2_small.calls
                                                         if isinstance(call, tuple))


def test_ewok_log_probabilities_are_applicable_and_scored(gpt2_small, prepared, tmp_path, monkeypatch):  # noqa: F811
    monkeypatch.setenv('RESULTCACHING_HOME', str(tmp_path / 'cache'))
    benchmark = brainscore.load_benchmark('EWoK-core-1.0-logprobs', root=prepared, output_dir=tmp_path / 'scored')
    value = brainscore.score(gpt2_small, benchmark, check_mem=False)
    assert value.attrs.get('status') != 'N/A'
    assert 0 <= float(value) <= 1
    assert value.attrs['items'] == 3 and value.attrs['n_trials'] == 6
