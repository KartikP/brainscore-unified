"""The old reading-times task answered by a likelihood part in the segment layout (ADR-0075).

A tiny GPT-2 on a word vocabulary (no weights, no download) answers through the legacy
facade and ``participate``. The old language helper (``HuggingfaceSubject``) on the same
network and tokenizer is the reference: the values are equal to the last bit, also after
the passage passes the model's window.
"""
import numpy as np
import pandas as pd
import pytest

from brainscore import BrainScoreModel
from brainscore_core.compatibility import ExecutionNotFeasibleError
from brainscore_core.contract import TaskContext
from brainscore_core.legacy_facade import LegacyBenchmarkFacade
from brainscore_core.responses import Segments
from brainscore_core.streaming_helpers import stimulus_session
from brainscore_core.supported_data_standards.brainio.stimuli import StimulusSet

pytestmark = pytest.mark.unit

torch = pytest.importorskip('torch')

F6 = 'behavior:log_probabilities'
WORDS = ['[UNK]', '[PAD]', 'the', 'dog', 'cat', 'ran', 'sat', 'and', 'a', 'big', 'small', 'home',
         ',', '.', 'it', 'was']
# 30 parts, more tokens than the window of 16 positions: the cache slides.
PASSAGE = ('the big dog ran home , and the small cat sat . it was a big cat , '
           'and the dog sat . a small dog ran home .').split()


@pytest.fixture
def tiny(monkeypatch):
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast
    raw = Tokenizer(models.WordLevel(vocab={w: i for i, w in enumerate(WORDS)}, unk_token='[UNK]'))
    raw.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=raw, unk_token='[UNK]', pad_token='[PAD]',
                                        eos_token='[PAD]', model_max_length=16, truncation_side='left')
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(3)
        network = GPT2LMHeadModel(GPT2Config(vocab_size=len(WORDS), n_positions=16, n_embd=16,
                                             n_layer=2, n_head=2)).eval()
    # The old helper moves its network to a GPU when one is there; keep both on the CPU.
    monkeypatch.setattr(torch.backends.mps, 'is_available', lambda: False)
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    monkeypatch.setattr('brainscore.model_helpers._device.select_device', lambda device=None: 'cpu')
    return network, tokenizer


def _old_reading_times(network, tokenizer, parts):
    from brainscore_language.artificial_subject import ArtificialSubject
    from brainscore_language.model_helpers.huggingface import HuggingfaceSubject
    old = HuggingfaceSubject(model_id='tiny', region_layer_mapping={}, model=network, tokenizer=tokenizer)
    old.start_behavioral_task(ArtificialSubject.Task.reading_times)
    return old.digest_text(parts)['behavior']


def _subject(likelihood):
    return BrainScoreModel('tiny-lm', in_channels={'text'}, likelihood=likelihood)


def _facade_reading_times(subject, parts):
    facade = LegacyBenchmarkFacade(subject)
    facade.start_behavioral_task('reading_times')
    return facade, facade.digest_text(parts)['behavior']


@pytest.mark.parametrize('parts', [PASSAGE, ['the dog', 'ran home .', 'the', 'cat']])
def test_the_values_equal_the_old_helper_to_the_last_bit(tiny, parts):
    from brainscore.model_helpers.causal_lm import _CausalLMParts
    network, tokenizer = tiny
    old = _old_reading_times(network, tokenizer, parts)
    _, new = _facade_reading_times(_subject(_CausalLMParts(network, tokenizer).likelihood), parts)
    assert new.values.dtype == old.values.dtype
    np.testing.assert_array_equal(new.values, old.values)      # NaN at the same places
    for coordinate in ('stimulus', 'context', 'part_number'):
        assert list(new[coordinate].values) == list(old[coordinate].values)


def test_the_chat_parts_likelihood_gives_the_same_values(tiny):
    from brainscore.model_helpers.replies import chat_parts
    network, tokenizer = tiny
    old = _old_reading_times(network, tokenizer, PASSAGE)
    _, new = _facade_reading_times(_subject(chat_parts(network, tokenizer)[1]), PASSAGE)
    np.testing.assert_array_equal(new.values, old.values)


def test_the_stored_answer_keeps_the_raw_first_value_with_its_reason(tiny):
    from brainscore.model_helpers.causal_lm import _CausalLMParts
    network, tokenizer = tiny
    subject = _subject(_CausalLMParts(network, tokenizer).likelihood)
    stimuli = StimulusSet(pd.DataFrame({'sentence': ['the dog ran'], 'segments': [('the', 'dog', 'ran')],
                                        'stimulus_id': ['s0']}))
    stimuli.identifier = 'segments-one'
    session = stimulus_session(stimuli, [F6], TaskContext(response=Segments('segments')))
    subject.participate(session)
    answer = session.collect(F6)
    assert list(answer['choice'].values) == ['1', '2', '3']
    assert np.isnan(answer.values[0, 0]) and np.isfinite(answer.values[0, 1:]).all()
    assert (answer.values[0, 1:] <= 0).all()
    assert list(answer['response_state'].values) == ['invalid']
    assert list(answer['response_reason'].values) == ['position 1: no conditioning context']
    parameters = answer.attrs['response']['parameters']
    assert parameters['layout'] == 'segment' and parameters['units'] == 'bits'
    assert parameters['tokenization']['truncation'] == 16


def test_a_likelihood_without_the_segment_layout_is_refused_before_any_call(tiny):
    calls = []

    def continuation(prompt, candidates):
        calls.append(prompt)
        return {candidate: -1.0 for candidate in candidates}

    facade = LegacyBenchmarkFacade(_subject(continuation))
    facade.start_behavioral_task('reading_times')
    with pytest.raises(ExecutionNotFeasibleError, match="states no 'segment' layout"):
        facade.digest_text(['the', 'dog'])
    assert calls == []


def test_a_likelihood_mapping_names_known_layouts_only():
    with pytest.raises(ValueError, match="'continuation' \\(required\\)"):
        _subject({'segment': lambda prompt: ([], {})})
    with pytest.raises(ValueError, match='got'):
        _subject({'continuation': lambda p, c: {}, 'segments': lambda prompt: ([], {})})


def test_an_empty_part_is_refused_before_a_forward_pass(tiny):
    from brainscore.model_helpers.causal_lm import _CausalLMParts
    network, tokenizer = tiny
    calls = []
    network.register_forward_hook(lambda *args: calls.append(1))
    with pytest.raises(ValueError, match='an empty part cannot be scored'):
        _facade_reading_times(_subject(_CausalLMParts(network, tokenizer).likelihood), ['the', '', 'dog'])
    assert calls == []


def test_a_model_with_no_likelihood_part_is_refused_before_compute():
    answered = []

    def answer(stimulus_row, instruction, label_set):
        answered.append(stimulus_row)
        return 'x'

    facade = LegacyBenchmarkFacade(BrainScoreModel('answers-only', in_channels={'text'}, answer=answer))
    facade.start_behavioral_task('reading_times')      # the old route: bare behavior
    with pytest.raises((ValueError, ExecutionNotFeasibleError), match='states no response'):
        facade.digest_text(['the', 'dog'])
    assert answered == []
