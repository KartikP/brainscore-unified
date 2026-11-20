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
    monkeypatch.setattr('brainscore.model_helpers.device.select_device', lambda device=None: 'cpu')
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
    from brainscore.model_helpers.causal_lm import CausalLMParts
    network, tokenizer = tiny
    old = _old_reading_times(network, tokenizer, parts)
    _, new = _facade_reading_times(_subject(CausalLMParts(network, tokenizer, sliding_window=True).likelihood), parts)
    assert new.values.dtype == old.values.dtype
    np.testing.assert_array_equal(new.values, old.values)      # NaN at the same places
    for coordinate in ('stimulus', 'context', 'part_number'):
        assert list(new[coordinate].values) == list(old[coordinate].values)


def test_the_chat_parts_likelihood_gives_the_same_values(tiny):
    from brainscore.model_helpers.replies import chat_parts
    network, tokenizer = tiny
    old = _old_reading_times(network, tokenizer, PASSAGE[:12])      # within the window of 16
    _, new = _facade_reading_times(_subject(chat_parts(network, tokenizer)[1]), PASSAGE[:12])
    np.testing.assert_array_equal(new.values, old.values)


def test_the_stored_answer_keeps_the_raw_first_value_with_its_reason(tiny):
    from brainscore.model_helpers.causal_lm import CausalLMParts
    network, tokenizer = tiny
    subject = _subject(CausalLMParts(network, tokenizer, sliding_window=True).likelihood)
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
    reason, = answer['response_reason'].values
    assert 'position 1' in reason and 'no conditioning context' in reason
    assert 'position 2' not in reason
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
    with pytest.raises(ValueError, match='callable'):
        _subject({'continuation': lambda p, c: {}, 'segment': 'not a part'})
    with pytest.raises(ValueError, match="likelihood\\['continuation'\\] is a callable part"):
        _subject({'continuation': None, 'segment': lambda prompt: ([], {})})


def test_an_empty_part_is_refused_before_a_forward_pass(tiny):
    from brainscore.model_helpers.causal_lm import CausalLMParts
    network, tokenizer = tiny
    calls = []
    network.register_forward_hook(lambda *args: calls.append(1))
    with pytest.raises(ValueError, match='an empty part cannot be scored'):
        _facade_reading_times(_subject(CausalLMParts(network, tokenizer, sliding_window=True).likelihood), ['the', '', 'dog'])
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


def _full_prefix(network, tokenizer, parts):
    """The reference: one full forward pass over the whole text, the log probability in bits
    of each part's new tokens given everything before them (NaN for a part with none)."""
    import math
    from brainscore_core.text import prepare_context
    encoded = tokenizer(prepare_context(parts), return_tensors='pt')
    tokens = encoded['input_ids'][0]
    with torch.no_grad():
        logp = network(**encoded).logits[0].log_softmax(-1)
    values, start = [], 0
    for i in range(len(parts)):
        end = len(tokenizer(prepare_context(parts[:i + 1]))['input_ids'])
        span = range(max(start, 1), end)
        values.append(float(sum(logp[k - 1, tokens[k]] for k in span)) / math.log(2) if span else np.nan)
        start = end
    return np.array(values)


@pytest.fixture
def tiny_qwen_vl(tiny):
    from transformers import Qwen2_5_VLConfig, Qwen2_5_VLForConditionalGeneration
    import copy
    tokenizer = copy.deepcopy(tiny[1])
    tokenizer.model_input_names = ['input_ids', 'attention_mask']   # as a Qwen tokenizer gives
    config = Qwen2_5_VLConfig(
        text_config=dict(vocab_size=len(WORDS), hidden_size=16, intermediate_size=24, num_hidden_layers=2,
                         num_attention_heads=2, num_key_value_heads=1, max_position_embeddings=64,
                         rope_scaling={'type': 'mrope', 'mrope_section': [2, 1, 1]}),
        vision_config=dict(depth=1, hidden_size=16, intermediate_size=24, num_heads=2, out_hidden_size=16,
                           fullatt_block_indexes=[0]))
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(7)
        network = Qwen2_5_VLForConditionalGeneration(config).eval().double()
    return network, tokenizer


@pytest.mark.parametrize('kind', ['gpt2', 'qwen2.5-vl'])
def test_the_cached_values_equal_one_full_prefix_pass(tiny, tiny_qwen_vl, kind):
    """The new tokens of each part get the positions after the cached ones (in float64, so a
    wrong position shows far above rounding)."""
    from brainscore.model_helpers.replies import chat_parts
    from transformers import GPT2Config, GPT2LMHeadModel
    if kind == 'gpt2':
        _, tokenizer = tiny
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(3)
            network = GPT2LMHeadModel(GPT2Config(vocab_size=len(WORDS), n_positions=64, n_embd=16,
                                                 n_layer=2, n_head=2)).eval().double()
    else:
        network, tokenizer = tiny_qwen_vl
    parts = PASSAGE[:12]
    _, new = _facade_reading_times(_subject(chat_parts(network, tokenizer)[1]), parts)
    np.testing.assert_allclose(new.values, -_full_prefix(network, tokenizer, parts), rtol=1e-10)


def test_a_float64_model_reduces_and_returns_float64_as_the_old_helper(tiny):
    from brainscore.model_helpers.causal_lm import CausalLMParts
    network, tokenizer = tiny
    network = network.double()
    old = _old_reading_times(network, tokenizer, PASSAGE)
    _, new = _facade_reading_times(_subject(CausalLMParts(network, tokenizer, sliding_window=True).likelihood), PASSAGE)
    assert new.values.dtype == old.values.dtype == np.float64
    np.testing.assert_array_equal(new.values, old.values)


def test_a_first_part_that_fills_the_window_is_scored_as_the_old_helper(tiny):
    from brainscore.model_helpers.causal_lm import CausalLMParts
    network, tokenizer = tiny
    parts = [' '.join(PASSAGE[:16]), 'the', 'dog']        # 16 tokens: the whole window
    old = _old_reading_times(network, tokenizer, parts)
    _, new = _facade_reading_times(_subject(CausalLMParts(network, tokenizer, sliding_window=True).likelihood), parts)
    assert np.isfinite(new.values).all()
    np.testing.assert_array_equal(new.values, old.values)


def test_a_schedule_that_cannot_fit_the_window_is_refused_before_a_forward_pass(tiny):
    from brainscore.model_helpers.causal_lm import CausalLMParts
    network, tokenizer = tiny
    calls = []
    network.register_forward_hook(lambda *args: calls.append(1))
    with pytest.raises(ExecutionNotFeasibleError, match='part 3 has 16 new tokens'):
        _facade_reading_times(_subject(CausalLMParts(network, tokenizer, sliding_window=True).likelihood),
                              ['the', 'dog', ' '.join(PASSAGE[:16])])
    assert calls == []


def test_the_chat_likelihood_keeps_its_two_argument_call(tiny):
    from brainscore.model_helpers.replies import chat_parts
    from brainscore.model_helpers.causal_lm import CausalLMParts
    from brainscore_core.responses import Choice, Prompt
    network, tokenizer = tiny
    prompt = Prompt((('text', 'the dog'),), response=Choice(['ran', 'sat'], verbatim=True))
    for likelihood in (chat_parts(network, tokenizer)[1], CausalLMParts(network, tokenizer, sliding_window=True).likelihood):
        assert set(likelihood(prompt, ['ran', 'sat'])) == {'ran', 'sat'}
        assert callable(likelihood['segment'])


def _segments_row(texts, parts):
    stimuli = StimulusSet(pd.DataFrame({'sentence': texts, 'segments': parts,
                                        'stimulus_id': [f's{i}' for i in range(len(texts))]}))
    stimuli.identifier = f'segments-{len(texts)}'
    return stimuli


def _recording_segment(calls):
    def segment(prompt):
        calls.append(prompt)
        return [None] + [-1.0] * (len(prompt.response.segments) - 1), {
            'layout': 'segment', 'context': 'prefix', 'aggregation': 'sum', 'units': 'bits',
            'tokenization': {'join': 'stand-in', 'special_tokens': 'none', 'template': 'none',
                             'truncation': 'none'}}
    return segment


@pytest.mark.parametrize('settings', [dict(instruction='Read this.'), dict(trial_memory='history')])
def test_a_segments_trial_that_shows_more_than_its_text_is_refused_before_any_call(settings):
    calls = []
    memory = 'trial_history' if settings.get('trial_memory') else None
    subject = BrainScoreModel('tiny-lm', in_channels={'text'}, memory=memory,
                              likelihood={'continuation': lambda p, c: {}, 'segment': _recording_segment(calls)})
    session = stimulus_session(_segments_row(['the dog'], [('the', 'dog')]), [F6],
                               TaskContext(response=Segments('segments'), **settings))
    with pytest.raises(ExecutionNotFeasibleError, match='an instruction|earlier trials'):
        subject.participate(session)
    assert calls == []


@pytest.mark.parametrize('memory, settings', [('trial_history', {}), (None, dict(present=('text',)))])
def test_independent_trials_with_their_text_only_are_answered(memory, settings):
    """A subject that can keep earlier trials, on independent trials, and a task that names the
    text as its one part, are valid Segments requests."""
    calls = []
    subject = BrainScoreModel('tiny-lm', in_channels={'text'}, memory=memory,
                              likelihood={'continuation': lambda p, c: {}, 'segment': _recording_segment(calls)})
    session = stimulus_session(_segments_row(['the dog'], [('the', 'dog')]), [F6],
                               TaskContext(response=Segments('segments'), **settings))
    subject.participate(session)
    assert len(calls) == 1
    np.testing.assert_array_equal(session.collect(F6).values[0, 1:], [-1.0])


def test_a_rotary_model_with_a_text_longer_than_its_window_is_refused_before_a_forward_pass(tiny_qwen_vl):
    """Its slid cache would keep keys rotated at their old positions."""
    from brainscore.model_helpers.replies import chat_parts
    network, tokenizer = tiny_qwen_vl
    network.config.get_text_config().max_position_embeddings = 8
    calls = []
    network.register_forward_hook(lambda *args: calls.append(1))
    with pytest.raises(ExecutionNotFeasibleError, match='does not state that its cache can slide'):
        _facade_reading_times(_subject(chat_parts(network, tokenizer)[1]), PASSAGE[:12])
    assert calls == []


def test_the_window_behavior_is_the_registration_statement_not_a_configuration_guess(tiny):
    """GPT-2 with an unused rope_theta in its configuration still has learned positions: stated
    as sliding, it is scored as the old helper scores it; not stated, its long text is refused."""
    from brainscore.model_helpers.causal_lm import CausalLMParts
    network, tokenizer = tiny
    network.config.rope_theta = 10000.0
    old = _old_reading_times(network, tokenizer, PASSAGE)
    _, new = _facade_reading_times(_subject(CausalLMParts(network, tokenizer, sliding_window=True).likelihood),
                                   PASSAGE)
    np.testing.assert_array_equal(new.values, old.values)
    with pytest.raises(ExecutionNotFeasibleError, match='does not state that its cache can slide'):
        _facade_reading_times(_subject(CausalLMParts(network, tokenizer).likelihood), PASSAGE)


def test_each_cached_run_holds_only_its_new_tokens(tiny):
    from brainscore.model_helpers.causal_lm import CausalLMParts
    network, tokenizer = tiny
    sizes = []

    def record(module, args, kwargs):
        ids = kwargs['input_ids']
        sizes.append((ids.untyped_storage().nbytes(), ids.numel() * ids.element_size()))

    network.register_forward_pre_hook(record, with_kwargs=True)
    _facade_reading_times(_subject(CausalLMParts(network, tokenizer, sliding_window=True).likelihood), PASSAGE[:10])
    assert len(sizes) == 10 and all(stored == held for stored, held in sizes)


@pytest.mark.parametrize('memory, streaming', [(None, False), ('trial_history', False), (None, True)])
def test_a_session_of_several_segments_trials_is_refused_before_any_call(memory, streaming):
    """No row is scored before a later row could be refused: the whole table, a subject that
    takes the table one trial at a time, and a streamed table."""
    calls = []
    subject = BrainScoreModel('tiny-lm', in_channels={'text'}, memory=memory,
                              likelihood={'continuation': lambda p, c: {}, 'segment': _recording_segment(calls)})
    session = stimulus_session(_segments_row(['the dog', 'the dog'], [('the', 'dog'), ('the', 'dog')]), [F6],
                               TaskContext(response=Segments('segments')), streaming=streaming)
    with pytest.raises(ExecutionNotFeasibleError, match='one Segments trial in a session'):
        subject.participate(session)
    assert calls == []


def test_a_second_segments_trial_on_a_raw_stream_is_refused_when_it_arrives():
    """A caller-built stream presents no known table: the first trial is answered, the second
    is refused before its compute."""
    from brainscore_core.streaming import Session
    calls = []
    subject = _subject({'continuation': lambda p, c: {}, 'segment': _recording_segment(calls)})
    task = TaskContext(response=Segments('segments'))
    table = _segments_row(['the dog', 'the cat'], [('the', 'dog'), ('the', 'cat')])
    built = stimulus_session(table, [F6], task, streaming=True)
    session = Session(built.source, [F6], task, stimuli=built.stimuli)
    with pytest.raises(ExecutionNotFeasibleError, match='one Segments trial in a session'):
        subject.participate(session)
    assert len(calls) == 1


def test_an_instruction_cannot_stand_in_for_the_text():
    calls = []
    subject = _subject({'continuation': lambda p, c: {}, 'segment': _recording_segment(calls)})
    session = stimulus_session(_segments_row(['the dog'], [('the', 'dog')]), [F6],
                               TaskContext(response=Segments('segments'), instruction='the dog', present=()))
    with pytest.raises(ExecutionNotFeasibleError, match='an instruction'):
        subject.participate(session)
    assert calls == []


class _EditedTokens:
    """A stand-in tokenizer: the real one, then ``edit`` on the list of ids of each text."""

    def __init__(self, tokenizer, edit):
        self._tokenizer, self._edit = tokenizer, edit

    def __call__(self, text, **kwargs):
        encoded = self._tokenizer(text, **kwargs)
        ids = self._edit(encoded['input_ids'][0].tolist())
        encoded['input_ids'] = torch.tensor([ids])
        encoded['attention_mask'] = torch.ones(1, len(ids), dtype=torch.long)
        return encoded


def _join_the_dog(ids):          # a merge across the parts: 'the', 'dog' become one id
    out, i = [], 0
    while i < len(ids):
        if ids[i:i + 2] == [2, 3]:
            out.append(11)
            i += 2
        else:
            out.append(ids[i])
            i += 1
    return out


_EDITS = {'merge across parts': (_join_the_dog, ['the', 'dog ran', 'home']),
          'end token after each text': (lambda ids: ids + [1], ['the', 'dog ran', 'home'])}


@pytest.mark.parametrize('name', list(_EDITS))
def test_a_tokenizer_whose_earlier_tokens_move_is_refused_before_a_forward_pass(tiny, name):
    from brainscore.model_helpers.replies import segment_log_probabilities
    network, tokenizer = tiny
    edit, parts = _EDITS[name]
    calls = []
    network.register_forward_hook(lambda *args: calls.append(1))
    with pytest.raises(ExecutionNotFeasibleError, match=r'part 2\b') as caught:
        segment_log_probabilities(network, _EditedTokens(tokenizer, edit), parts)
    assert 'first tokens' in str(caught.value) and 'add_special_tokens' in str(caught.value)
    assert calls == []


def test_a_stable_tokenizer_keeps_its_values_and_parameters(tiny):
    from brainscore.model_helpers.replies import segment_log_probabilities
    network, tokenizer = tiny
    parts = ['the dog', 'ran home .', 'the', 'cat']
    values, parameters = segment_log_probabilities(network, _EditedTokens(tokenizer, list), parts)
    reference, reference_parameters = segment_log_probabilities(network, tokenizer, parts)
    assert values == reference and parameters == reference_parameters
