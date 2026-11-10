"""build_causal_lm_subject: record, intervene, generate and score, through participate() and process()."""
import json

import numpy as np
import pytest

pytestmark = pytest.mark.unit

torch = pytest.importorskip('torch')

from brainscore_core.model_interface import Perturbation, Selection, StateChange
from brainscore_core.streaming import StreamEvent
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


def _ask(model, channel, payload):
    response = model.process(StreamEvent(channel, payload, None, {}))
    assert response.channel == 'response_trace' and response.payload['valid'] is True
    return response.payload


def _stimuli():
    s = StimulusSet({'stimulus_id': ['a', 'b'], 'sentence': ['the cat sat', 'the dog ran is kind']})
    s.identifier = 'causal-lm-test'
    return s


def test_logprobs_match_a_manual_forward_pass(subject):
    model, net, tok = subject
    payload = _ask(model, 'conditional_logprob_request',
                   {'contexts': ['the cat', 'the dog ran is'], 'targets': ['sat', 'kind mean']})
    assert payload['provenance']['revision'] == 'test'
    for context, target, value in zip(['the cat', 'the dog ran is'], ['sat', 'kind mean'], payload['answer']):
        c, t = tok(context)['input_ids'], tok(target)['input_ids']
        logp = net(torch.tensor([c + t])).logits[0, :-1].log_softmax(-1)
        expected = sum(logp[len(c) - 1 + i, t[i]].item() for i in range(len(t)))
        assert value == pytest.approx(expected, abs=1e-5)  # batched + padded == unbatched


def test_mean_reduction_and_chat_contexts(subject):
    model, _, _ = subject
    req = {'contexts': ['the cat'], 'targets': ['sat is kind'], 'context_format': 'chat'}
    total = _ask(model, 'conditional_logprob_request', req)['answer'][0]
    mean = _ask(model, 'conditional_logprob_request', {**req, 'reduction': 'mean'})['answer'][0]
    assert mean == pytest.approx(total / 3, abs=1e-6)


def test_generation_single_and_batched(subject):
    model, _, tok = subject
    single = _ask(model, 'generation_request', {'messages': [{'role': 'user', 'content': 'the cat'}], 'max_tokens': 4})
    batch = _ask(model, 'generation_request', {'conversations': [[{'role': 'user', 'content': 'the cat'}],
                                                                  [{'role': 'user', 'content': 'the dog ran'}]],
                                               'max_tokens': 4})
    assert isinstance(single['answer'], str)
    assert batch['raw']['texts'][0] == single['answer']
    assert tok.padding_side == 'right'  # generation's left padding does not leak into recording


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
    req = {'contexts': ['the cat', 'the dog'], 'targets': ['sat', 'ran'], 'context_format': 'chat'}
    before = _ask(model, 'conditional_logprob_request', req)['answer']
    model.process(StateChange(kind='steer', perturbation=Perturbation(kind='drive', amount=10.0),
                              target=Selection(layer=NEURON_IN, indices=[4])))
    during = _ask(model, 'conditional_logprob_request', req)['answer']
    model.reset()
    after = _ask(model, 'conditional_logprob_request', req)['answer']
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
    req = {'contexts': ['the cat'], 'targets': ['sat'], 'context_format': 'chat'}
    before = _ask(model, 'conditional_logprob_request', req)['answer']
    model.process(StateChange(kind='lesion', target=target, perturbation=Perturbation(kind='drive', amount=50.0)))
    assert _ask(model, 'conditional_logprob_request', req)['answer'] != before


# Through participate(): the contract's only evaluation method.

def _session_ask(model, channel, payload):
    from brainscore_core.streaming import BufferedSession
    session = BufferedSession([StreamEvent(channel, payload, None, {})],
                              requested_output_channels=['response_trace'])
    model.participate(session)
    (response,) = session.emitted
    assert response.channel == 'response_trace' and response.payload['valid'] is True
    return response.payload


def test_declares_what_it_can_do(subject):
    model, _, _ = subject
    assert {'text', 'generation_request', 'conditional_logprob_request'} <= model.in_channels
    assert {'neural:mlp_in', 'neural:block1', 'response_trace'} <= model.out_channels


def test_generate_and_score_through_participate_match_process(subject):
    model, _, _ = subject
    request = {'contexts': ['the cat', 'the dog ran is'], 'targets': ['sat', 'kind mean']}
    assert _session_ask(model, 'conditional_logprob_request', request)['answer'] == \
        _ask(model, 'conditional_logprob_request', request)['answer']
    request = {'conversations': [[{'role': 'user', 'content': 'the cat'}]], 'max_tokens': 3}
    assert _session_ask(model, 'generation_request', request)['raw']['texts'] == \
        _ask(model, 'generation_request', request)['raw']['texts']


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


# Conditional log-probability request policy.

def _manual_logprob(net, tok, context, target):
    c, t = tok(context)['input_ids'], tok(target)['input_ids']
    with torch.no_grad():
        logp = net(torch.tensor([c + t])).logits[0, :-1].log_softmax(-1)
    return [logp[len(c) - 1 + i, t[i]].item() for i in range(len(t))]


@pytest.mark.parametrize('reduction', ['sum', 'mean'])
def test_batches_larger_than_batch_size_match_unbatched_scores(subject, reduction):
    model, net, tok = subject
    contexts = ['the cat', 'the dog ran is', 'cat', 'the dog', 'dog ran is kind', 'the',
                'sat', 'the cat sat', 'mean dog', 'is']
    targets = ['sat', 'kind mean', 'sat is', 'ran', 'mean', 'cat sat ran', 'is', 'kind', 'ran', 'the dog']
    assert len(contexts) > 8  # the fixture's batch_size
    answer = _ask(model, 'conditional_logprob_request',
                  {'contexts': contexts, 'targets': targets, 'reduction': reduction})['answer']
    for context, target, value in zip(contexts, targets, answer):
        tokens = _manual_logprob(net, tok, context, target)
        expected = sum(tokens) / (len(tokens) if reduction == 'mean' else 1)
        assert value == pytest.approx(expected, abs=1e-5)


@pytest.mark.parametrize('request_payload, message', [
    ({'contexts': [''], 'targets': ['sat']}, 'no tokens to condition on'),
    ({'contexts': ['the cat', ''], 'targets': ['sat', 'ran']}, 'item 1: the context'),
    ({'contexts': ['the cat'], 'targets': ['']}, 'no tokens to score'),
    ({'contexts': ['the cat'], 'targets': ['sat'], 'context_format': 'Chat'}, 'context_format'),
    ({'contexts': ['the cat'], 'targets': ['sat'], 'context_format': 'chatml'}, 'context_format'),
    ({'contexts': ['the cat'], 'targets': ['sat'], 'reduction': 'avg'}, 'reduction'),
    ({'contexts': ['the cat'], 'targets': ['sat', 'ran']}, 'one context per target'),
])
def test_invalid_requests_are_rejected_before_any_forward_pass(subject, monkeypatch, request_payload, message):
    model, net, _ = subject
    calls = []
    original = type(net).forward
    monkeypatch.setattr(type(net), 'forward', lambda self, *a, **k: calls.append(1) or original(self, *a, **k))
    with pytest.raises(ValueError, match=message):
        _ask(model, 'conditional_logprob_request', request_payload)
    with pytest.raises(ValueError, match=message):
        _session_ask(model, 'conditional_logprob_request', request_payload)
    assert calls == []


def test_non_string_items_are_rejected(subject):
    model, _, _ = subject
    with pytest.raises(TypeError, match='strings'):
        _ask(model, 'conditional_logprob_request', {'contexts': [None], 'targets': ['sat']})


def test_an_empty_chat_context_is_conditioned_on_the_template(subject):
    model, net, tok = subject
    value = _ask(model, 'conditional_logprob_request',
                 {'contexts': [''], 'targets': ['sat'], 'context_format': 'chat'})['answer'][0]
    prompt = tok.apply_chat_template([{'role': 'user', 'content': ''}], tokenize=False,
                                     add_generation_prompt=True)
    assert value == pytest.approx(sum(_manual_logprob(net, tok, prompt, 'sat')), abs=1e-5)
    assert value < 0


# Provenance travels with neural responses as well as response traces.

def test_neural_responses_carry_a_copy_of_the_provenance(subject):
    from brainscore_core.streaming import BufferedSession
    from brainscore_core.streaming_helpers import neural_response
    model, _, _ = subject
    expected = {'identifier': 'tiny-qwen', 'revision': 'test'}
    session = BufferedSession(stimulus_set=_stimuli(), requested_output_channels=['neural:mlp_in'])
    model.participate(session)
    (event,) = session.emitted
    assert event.meta == {'driver': 'interact', 'record': 'mlp_in'}  # record format unchanged
    assert event.payload.attrs['provenance'] == expected
    event.payload.attrs['provenance']['revision'] = 'edited'  # a copy per response
    assert neural_response(model, _stimuli(), record='mlp_in').attrs['provenance'] == expected
    model.start_recording('mlp_in')
    assert model.process(_stimuli()).attrs['provenance'] == expected
    trace = _session_ask(model, 'conditional_logprob_request', {'contexts': ['the cat'], 'targets': ['sat']})
    assert trace['provenance'] == expected


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
    assert _ask(model, 'conditional_logprob_request',
                {'contexts': ['the cat'], 'targets': ['sat']})['provenance'] == expected


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


@pytest.mark.parametrize('contexts, targets', [
    ('the cat', 'sat'), (b'the cat', b'sat'), ({'the cat': 1}, {'sat': 1}),
    (['the cat'], 'sat'), ('the cat', ['sat']), (np.array([['the cat']]), np.array([['sat']])),
])
def test_strings_and_mappings_are_not_read_as_batches(subject, monkeypatch, contexts, targets):
    model, net, _ = subject
    calls = []
    original = type(net).forward
    monkeypatch.setattr(type(net), 'forward', lambda self, *a, **k: calls.append(1) or original(self, *a, **k))
    with pytest.raises(TypeError, match='list, tuple or 1-d array'):
        _session_ask(model, 'conditional_logprob_request', {'contexts': contexts, 'targets': targets})
    assert calls == []


def test_tuple_and_array_batches_match_list_batches(subject):
    model, _, _ = subject
    contexts, targets = ['the cat', 'the dog ran is'], ['sat', 'kind mean']
    expected = _ask(model, 'conditional_logprob_request', {'contexts': contexts, 'targets': targets})['answer']
    for kind in (tuple, np.array):
        assert _ask(model, 'conditional_logprob_request',
                    {'contexts': kind(contexts), 'targets': kind(targets)})['answer'] == expected
