"""build_causal_lm_subject: record, intervene, generate and score, all through process()."""
import json

import numpy as np
import pytest

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
    monkeypatch.setattr('brainscore.model_helpers._device.select_device', lambda: 'cpu')
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
    model.process(StateChange(kind='steer', perturbation=Perturbation(kind='drive_norm', amount=10.0),
                              target=Selection(layer=NEURON_IN, indices=[4])))
    during = _ask(model, 'conditional_logprob_request', req)['answer']
    model.reset()
    after = _ask(model, 'conditional_logprob_request', req)['answer']
    assert during != before and after == before
