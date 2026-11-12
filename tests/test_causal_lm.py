"""build_causal_lm_subject: record, intervene, reply and score, through participate() and process()."""
import json

import numpy as np
import pytest

pytestmark = pytest.mark.unit

torch = pytest.importorskip('torch')

from brainscore_core.model_interface import Perturbation, Selection, StateChange
from brainscore_core.supported_data_standards.brainio.stimuli import StimulusSet
from brainscore.model_helpers.causal_lm import build_causal_lm_subject

WORDS = ['[PAD]', '[UNK]', 'user', 'assistant', ':', 'the', 'cat', 'dog', 'sat', 'ran', 'is', 'kind', 'mean']
NEURON_IN = 'model.layers.1.mlp.down_proj:input'


@pytest.fixture
def subject(monkeypatch):
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast, Qwen2Config, Qwen2ForCausalLM
    vocab = {w: i for i, w in enumerate(WORDS)}
    raw = Tokenizer(models.WordLevel(vocab=vocab, unk_token='[UNK]'))
    raw.pre_tokenizer = pre_tokenizers.Whitespace()
    tok = PreTrainedTokenizerFast(tokenizer_object=raw, unk_token='[UNK]', pad_token='[PAD]',
                                  eos_token='[PAD]', model_max_length=64)
    tok.chat_template = ("{% for m in messages %}{{ m['role'] }} : {{ m['content'] }} {% endfor %}"
                         "{% if add_generation_prompt %}assistant : {% endif %}")
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(3)
        net = Qwen2ForCausalLM(Qwen2Config(vocab_size=len(WORDS), hidden_size=16, intermediate_size=24,
                                           num_hidden_layers=2, num_attention_heads=2,
                                           num_key_value_heads=1, max_position_embeddings=64)).eval()
    monkeypatch.setattr('brainscore.model_helpers._device.select_device', lambda device=None: 'cpu')
    model = build_causal_lm_subject('tiny-qwen', net, tok, provenance={'revision': 'test'},
                                    region_layer_map={'mlp_in': NEURON_IN, 'block1': 'model.layers.1'})
    yield model, net, tok
    model.reset()


def _stimuli():
    s = StimulusSet({'stimulus_id': ['a', 'b'], 'sentence': ['the cat sat', 'the dog ran is kind']})
    s.identifier = 'causal-lm-test'
    return s


def _scores(model, contexts, targets, raw_text=True):
    """Each target's raw score after its context, through participate.

    A raw-score choice needs two distinct options per row; the second is a
    filler word, and only the first column is read.
    """
    from brainscore_core.contract import TaskContext
    from brainscore_core.responses import Choice
    from brainscore_core.streaming_helpers import stimulus_session
    fillers = ['mean' if target != 'mean' else 'kind' for target in targets]
    stimuli = StimulusSet({'stimulus_id': [f's{i}' for i in range(len(contexts))], 'text': list(contexts),
                           'options': [(t, f) for t, f in zip(targets, fillers)]})
    stimuli.identifier = 'causal-lm-scores'
    task = TaskContext(response=Choice(labels='options', raw_scores=True, raw_text=raw_text))
    session = stimulus_session(stimuli, ['behavior:probabilities'], task)
    model.participate(session)
    scores = session.collect('behavior:probabilities')
    assert scores.attrs['measure'] == {'kind': 'implied', 'source': 'likelihood', 'scale': 'log'}
    return [float(v) for v in scores.values[:, 0]]


def _reply(model, text, **settings):
    from brainscore_core.contract import TaskContext
    from brainscore_core.responses import Text
    from brainscore_core.streaming_helpers import stimulus_session
    stimuli = StimulusSet({'stimulus_id': ['q'], 'text': [text]})
    stimuli.identifier = 'causal-lm-written'
    session = stimulus_session(stimuli, ['behavior:text'], TaskContext(response=Text(**settings)))
    model.participate(session)
    return str(session.collect('behavior:text').values[0])


def _manual_logprob(net, tok, context, target):
    c, t = tok(context)['input_ids'], tok(target)['input_ids']
    with torch.no_grad():
        logp = net(torch.tensor([c + t])).logits[0, :-1].log_softmax(-1)
    return [logp[len(c) - 1 + i, t[i]].item() for i in range(len(t))]


def test_logprobs_match_a_manual_forward_pass(subject):
    """Plain text: log p(target | context), as the removed request channel computed it."""
    model, net, tok = subject
    contexts, targets = ['the cat', 'the dog ran is'], ['sat', 'kind mean']
    for context, target, value in zip(contexts, targets, _scores(model, contexts, targets)):
        assert value == pytest.approx(sum(_manual_logprob(net, tok, context, target)), abs=1e-5)


def test_chat_scoring_puts_the_text_in_a_user_turn(subject):
    """Without raw_text the trial (text, then the choice's directive) is a chat user turn."""
    model, net, tok = subject
    value = _scores(model, ['the cat'], ['sat'], raw_text=False)[0]
    shown = 'the cat\n\nAnswer with exactly one of: sat, mean.'
    prompt = tok.apply_chat_template([{'role': 'user', 'content': shown}], tokenize=False,
                                     add_generation_prompt=True)
    assert value == pytest.approx(sum(_manual_logprob(net, tok, prompt, 'sat')), abs=1e-5)


def test_reply_obeys_the_task_settings(subject, monkeypatch):
    model, net, tok = subject
    calls = []
    original = type(net).generate
    monkeypatch.setattr(type(net), 'generate',
                        lambda self, **kwargs: calls.append(kwargs) or original(self, **kwargs))
    first = _reply(model, 'the cat', max_tokens=4)
    assert isinstance(first, str)
    assert calls[-1]['max_new_tokens'] == 4 and calls[-1]['do_sample'] is False
    _reply(model, 'the cat')
    assert calls[-1]['max_new_tokens'] == 256 and calls[-1]['do_sample'] is False
    _reply(model, 'the cat', max_tokens=3, temperature=0.7, top_p=0.9, seed=1)
    assert (calls[-1]['do_sample'], calls[-1]['temperature'], calls[-1]['top_p']) == (True, 0.7, 0.9)
    assert _reply(model, 'the cat', max_tokens=3, temperature=0.7, seed=1) == \
        _reply(model, 'the cat', max_tokens=3, temperature=0.7, seed=1)
    assert tok.padding_side == 'right'  # generation does not change recording's padding


def test_recording_input_site_sees_the_steered_neuron(subject):
    model, net, _ = subject
    model.start_recording('mlp_in')
    base = model.process(_stimuli())
    assert base.sizes['neuroid'] == 24  # the MLP's intermediate units, not the hidden size
    model.process(StateChange(kind='steer', perturbation=Perturbation(kind='drive', amount=5.0),
                              target=Selection(layer=NEURON_IN, indices=[4])))
    steered = model.process(_stimuli())
    diff = np.nan_to_num(steered.values - base.values)
    assert np.allclose(diff[..., 4][~np.isnan(base.values[..., 4])], 5.0, atol=1e-5)
    assert np.abs(np.delete(diff, 4, axis=-1)).max() == 0
    model.reset()  # also clears the recording setup
    model.start_recording('mlp_in')
    np.testing.assert_array_equal(model.process(_stimuli()).values, base.values)


def test_intervention_changes_scores_and_reset_restores_them(subject):
    model, _, _ = subject
    contexts, targets = ['the cat', 'the dog'], ['sat', 'ran']
    before = _scores(model, contexts, targets, raw_text=False)
    model.process(StateChange(kind='steer', perturbation=Perturbation(kind='drive', amount=10.0),
                              target=Selection(layer=NEURON_IN, indices=[4])))
    during = _scores(model, contexts, targets, raw_text=False)
    model.reset()
    after = _scores(model, contexts, targets, raw_text=False)
    assert during != before and after == before


def test_functional_localizer_finds_and_perturbs_mlp_neurons(subject):
    """Record-then-perturb (FunctionalSelection) works on ':input' sites end to end."""
    from brainscore.selection import FunctionalSelection
    model, net, _ = subject
    localizer = StimulusSet({'stimulus_id': list('abcd'), 'sentence': ['the cat', 'the dog', 'cat sat', 'dog ran'],
                             'label': ['x', 'x', 'y', 'y']})
    localizer.identifier = 'causal-lm-localizer'
    target = FunctionalSelection(recording_target='mlp_in', localizer_stimuli=localizer,
                                 contrast=(['x'], ['y']), n_units=3, sign='abs')
    assert target.resolve(model).layer == NEURON_IN
    before = _scores(model, ['the cat'], ['sat'], raw_text=False)
    model.process(StateChange(kind='lesion', target=target, perturbation=Perturbation(kind='drive', amount=50.0)))
    assert _scores(model, ['the cat'], ['sat'], raw_text=False) != before


def test_declares_what_it_can_do(subject):
    model, _, _ = subject
    assert {'text', 'instruction'} <= model.in_channels
    assert {'neural:mlp_in', 'neural:block1', 'behavior:text', 'behavior:choice',
            'behavior:probabilities'} <= model.out_channels


def test_record_and_push_an_input_site_through_participate(subject):
    from brainscore.instrumentation import intervene
    from brainscore_core.streaming_helpers import neural_response
    model, _, _ = subject
    before = neural_response(model, _stimuli(), record='mlp_in')
    push = StateChange(kind='steer', target=Selection(layer=NEURON_IN, indices=[5]),
                       perturbation=Perturbation(kind='drive', amount=2.5))
    with intervene(model, push):
        after = neural_response(model, _stimuli(), record='mlp_in')
    change = np.nan_to_num(after.values - before.values)
    np.testing.assert_allclose(change[..., 5][~np.isnan(before.values[..., 5])], 2.5, atol=1e-5)
    assert np.abs(np.delete(change, 5, axis=-1)).max() == 0
    restored = neural_response(model, _stimuli(), record='mlp_in')
    np.testing.assert_array_equal(np.nan_to_num(restored.values), np.nan_to_num(before.values))


@pytest.mark.parametrize('contexts, targets', [
    (['the cat', 'the dog ran is', 'cat', 'the dog', 'dog ran is kind', 'the', 'sat', 'the cat sat',
      'mean dog', 'is'],
     ['sat', 'kind mean', 'sat is', 'ran', 'mean', 'cat sat ran', 'is', 'kind', 'ran', 'the dog']),
])
def test_many_rows_match_unbatched_scores(subject, contexts, targets):
    model, net, tok = subject
    assert len(contexts) > 8  # more than the fixture's batch_size
    for context, target, value in zip(contexts, targets, _scores(model, contexts, targets)):
        assert value == pytest.approx(sum(_manual_logprob(net, tok, context, target)), abs=1e-5)


@pytest.mark.parametrize('contexts, targets, message', [
    ([''], ['sat'], 'no tokens to condition on'),
    (['the cat'], [''], 'option strings|no tokens to score'),
])
def test_unscorable_text_is_refused_before_its_forward_pass(subject, monkeypatch, contexts, targets, message):
    model, net, _ = subject
    calls = []
    original = type(net).forward
    monkeypatch.setattr(type(net), 'forward', lambda self, *a, **k: calls.append(1) or original(self, *a, **k))
    with pytest.raises(ValueError, match=message):
        _scores(model, contexts, targets)
    assert calls == []


def test_an_empty_chat_text_is_conditioned_on_the_template(subject):
    model, net, tok = subject
    value = _scores(model, [''], ['sat'], raw_text=False)[0]
    shown = 'Answer with exactly one of: sat, mean.'
    prompt = tok.apply_chat_template([{'role': 'user', 'content': shown}], tokenize=False,
                                     add_generation_prompt=True)
    assert value == pytest.approx(sum(_manual_logprob(net, tok, prompt, 'sat')), abs=1e-5)
    assert value < 0


# Provenance travels with neural responses.

def test_neural_responses_carry_a_copy_of_the_provenance(subject):
    from brainscore_core.streaming_helpers import stimulus_session
    from brainscore_core.streaming_helpers import neural_response
    model, _, _ = subject
    expected = {'identifier': 'tiny-qwen', 'revision': 'test'}
    session = stimulus_session(_stimuli(), ['neural:mlp_in'])
    model.participate(session)
    (event,) = session.emitted
    assert event.meta == {'driver': 'interact', 'record': 'mlp_in'}  # record format unchanged
    assert event.payload.attrs['provenance'] == expected
    event.payload.attrs['provenance']['revision'] = 'edited'  # a copy per response
    assert neural_response(model, _stimuli(), record='mlp_in').attrs['provenance'] == expected
    model.start_recording('mlp_in')
    assert model.process(_stimuli()).attrs['provenance'] == expected


def test_caller_provenance_is_copied_at_registration(subject):
    from brainscore_core.streaming_helpers import neural_response
    _, net, tok = subject
    supplied = {'revision': 'abc', 'settings': {'dtype': 'float32'}}
    model = build_causal_lm_subject('tiny-qwen-copy', net, tok, provenance=supplied,
                                    region_layer_map={'mlp_in': NEURON_IN})
    supplied['revision'] = 'changed'
    supplied['settings']['dtype'] = 'changed'
    expected = {'identifier': 'tiny-qwen-copy', 'revision': 'abc', 'settings': {'dtype': 'float32'}}
    assert neural_response(model, _stimuli(), record='mlp_in').attrs['provenance'] == expected


def test_text_wrapper_adds_no_provenance_by_default(subject):
    from brainscore.model_helpers.text_wrapper import TextWrapper
    _, net, tok = subject
    assembly = TextWrapper(net, tok, identifier='plain', layer_aggregation='per_token')(_stimuli(), [NEURON_IN])
    assert 'provenance' not in assembly.attrs


@pytest.mark.parametrize('layer', [':input', 'model.layers.1.mlp.down_proj:middle:input',
                                   'model.layers.1.mlp.down_proj:input:input'])
def test_recording_rejects_malformed_sites(subject, layer):
    from brainscore.model_helpers.text_wrapper import TextWrapper
    _, net, tok = subject
    with pytest.raises(ValueError, match='site|empty module path'):
        TextWrapper(net, tok, identifier='strict', layer_aggregation='per_token')(_stimuli(), [layer])
    assert not any(m._forward_hooks or m._forward_pre_hooks for m in net.modules())


def test_the_instruction_is_the_system_turn():
    from jinja2.exceptions import TemplateError
    from brainscore_core.responses import Prompt
    from brainscore.model_helpers.causal_lm import _chat_text, _instruction_and_text

    class Tokenizer:
        def __init__(self, takes_system):
            self.takes_system = takes_system

        def apply_chat_template(self, messages, tokenize, add_generation_prompt):
            if messages[0]['role'] == 'system' and not self.takes_system:
                raise TemplateError('no system turn')
            return messages

    prompt = Prompt((('text', 'Follow the rules.'), ('text', 'Is it a word?')), names=('instruction', 'text'))
    instruction, text = _instruction_and_text(prompt)
    assert (instruction, text) == ('Follow the rules.', 'Is it a word?')
    turns = [{'role': 'user', 'content': text}]
    assert _chat_text(Tokenizer(True), instruction, turns, {}) == [
        {'role': 'system', 'content': 'Follow the rules.'}, {'role': 'user', 'content': 'Is it a word?'}]
    # A template with no system turn gets the instruction first in the user turn.
    assert _chat_text(Tokenizer(False), instruction, turns, {}) == [
        {'role': 'user', 'content': 'Follow the rules.\n\nIs it a word?'}]
