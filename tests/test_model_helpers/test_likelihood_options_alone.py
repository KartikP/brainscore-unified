"""The log probability of an option does not depend on the other options of its trial.

A likelihood part scores each option in its own forward pass. The stand-in language model
here gives logits that change with the shape of the batch (its number of rows and its padded
width), as a half-precision model does; an option scored in a batch with other options would
then change with them.
"""
import types

import numpy as np
import pytest
import torch

from brainscore_core.contract import TaskContext
from brainscore_core.responses import Choice
from brainscore_core.streaming_helpers import stimulus_session
from brainscore_core.supported_data_standards.brainio.stimuli import StimulusSet
from brainscore.benchmarks.ewok.test import likelihood_subject, tiny_lm  # noqa: F401  (tiny_lm is a fixture)

pytestmark = pytest.mark.unit

TEXTS = ['the cat', 'a dog is']
OPTION_SETS = [('sat', 'ran'), ('ran', 'sat'), ('sat', 'is kind'), ('sat', 'ran', 'is mean')]


class _BatchShaped(torch.nn.Module):
    """The tiny network, with logits that gain a value growing with the batch size and padded width."""

    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, input_ids, attention_mask=None, **kwargs):
        logits = self.net(input_ids=input_ids, attention_mask=attention_mask).logits
        rows, width = input_ids.shape
        return types.SimpleNamespace(logits=logits + 1e-2 * (rows + width) * torch.arange(logits.shape[-1]))


def _subject(kind, net, tok):
    from brainscore.model_helpers.causal_lm import build_causal_lm_subject
    from brainscore.model_helpers.replies import chat_parts
    if kind == 'chat_parts':
        return likelihood_subject(chat_parts(net, tok)[1])
    return build_causal_lm_subject('batch-shaped', net, tok)


def _collected(model, options, output, verbatim):
    stimuli = StimulusSet({'stimulus_id': [f's{i}' for i in range(len(TEXTS))], 'text': TEXTS,
                           'options': [options] * len(TEXTS)})
    stimuli.identifier = f'options-alone-{"-".join(options)}'
    session = stimulus_session(stimuli, [output], TaskContext(response=Choice(labels='options', verbatim=verbatim)))
    model.participate(session)
    values = session.collect(output).transpose('presentation', 'choice').values
    return {option: values[:, index] for index, option in enumerate(options)}


@pytest.mark.parametrize('kind', ['causal_lm', 'chat_parts'])
def test_verbatim_log_probabilities_of_an_option_are_equal_for_any_other_options(tiny_lm, kind):
    net, tok = tiny_lm
    model = _subject(kind, _BatchShaped(net), tok)
    found = [_collected(model, options, 'behavior:log_probabilities', verbatim=True) for options in OPTION_SETS]
    for option in ('sat', 'ran'):
        values = [scores[option] for scores in found if option in scores]
        assert all(np.array_equal(value, values[0]) for value in values), (option, values)


def test_implied_probabilities_keep_the_ratio_of_two_options_for_any_other_options(tiny_lm):
    """The causal-model helper outside verbatim (the chat-template path): the probabilities of
    two options keep their ratio, whatever other options the trial has."""
    net, tok = tiny_lm
    # A stand-in template that shows only the stimulus text, not the directive that names the
    # options: the context is then the same for every option set.
    tok.chat_template = "{% for message in messages %}{{ message['content'].split('\\n\\n')[0] }} {% endfor %}"
    model = _subject('causal_lm', _BatchShaped(net), tok)
    ratios = []
    for options in OPTION_SETS:
        probabilities = _collected(model, options, 'behavior:probabilities', verbatim=False)
        ratios.append(np.log(probabilities['sat']) - np.log(probabilities['ran'])
                      if 'ran' in probabilities else None)
    ratios = [ratio for ratio in ratios if ratio is not None]
    # The probabilities are float32 values of a softmax: rounding only (a batch of options moves
    # the ratio by about 0.02 here).
    np.testing.assert_allclose(ratios, [ratios[0]] * len(ratios), rtol=0, atol=1e-6)


class _Characters:
    """One token per character (its code), no start token; a chat template that shows the stimulus text."""
    model_max_length = 64   # a real tokenizer states its limit
    pad_token_id = eos_token_id = 0
    pad_token = eos_token = '\x00'

    def __call__(self, text, add_special_tokens=False, **kwargs):
        return {'input_ids': [ord(c) for c in text], 'attention_mask': [1] * len(text)}

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True, **kwargs):
        return ''.join(message['content'].split('\n\n')[0] + ' ' for message in messages)


class _BatchIndependent(torch.nn.Module):
    """Logits that depend only on the preceding token, never on the batch."""

    def __init__(self, dtype):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(1, dtype=dtype))

    def forward(self, input_ids, attention_mask=None, **kwargs):
        table = torch.linspace(-3, 3, 256, dtype=self.weight.dtype)
        return types.SimpleNamespace(logits=torch.stack([torch.stack([torch.roll(table, int(token)) for token in row])
                                                         for row in input_ids]))


def _batched_layout(net, sequences, start):
    """The values of the earlier batched layout: all options of a trial in one right-padded batch."""
    width = max(len(sequence) for sequence in sequences)
    ids = torch.tensor([sequence + [0] * (width - len(sequence)) for sequence in sequences])
    mask = torch.tensor([[1] * len(sequence) + [0] * (width - len(sequence)) for sequence in sequences])
    scored = torch.tensor([[0] * start + [1] * (len(sequence) - start) + [0] * (width - len(sequence))
                           for sequence in sequences])[:, 1:].float()
    with torch.no_grad():
        logp = net(input_ids=ids, attention_mask=mask).logits[:, :-1].float().log_softmax(-1)
    return (logp.gather(-1, ids[:, 1:, None])[..., 0] * scored).sum(1).tolist()


CONTROL_TEXTS, CONTROL_OPTIONS = ['dogs', 'a cat'], ('a', 'long')


@pytest.fixture
def cpu(monkeypatch):
    monkeypatch.setattr('brainscore.model_helpers.device.select_device', lambda device=None: 'cpu')


def _control_stimuli():
    stimuli = StimulusSet({'stimulus_id': ['c0', 'c1'], 'text': CONTROL_TEXTS,
                           'options': [CONTROL_OPTIONS] * len(CONTROL_TEXTS)})
    stimuli.identifier = 'options-alone-control'
    return stimuli


@pytest.mark.parametrize('dtype', [torch.float32, torch.float16])
@pytest.mark.parametrize('kind', ['causal_lm', 'chat_parts'])
def test_a_model_with_no_batch_effect_gives_the_values_of_the_batched_layout(cpu, kind, dtype):
    """Scoring each option alone changes no value of a model whose output does not depend on the batch."""
    net, tok = _BatchIndependent(dtype), _Characters()
    session = stimulus_session(_control_stimuli(), ['behavior:log_probabilities'],
                               TaskContext(response=Choice(labels='options', verbatim=True)))
    _subject(kind, net, tok).participate(session)
    found = session.collect('behavior:log_probabilities').transpose('presentation', 'choice').values
    base = [_batched_layout(net, [tok(f'{text} {option}')['input_ids'] for option in CONTROL_OPTIONS],
                            len(text)) for text in CONTROL_TEXTS]
    assert found.tolist() == base


@pytest.mark.parametrize('dtype', [torch.float32, torch.float16])
def test_the_causal_helper_outside_verbatim_gives_the_values_of_the_batched_layout(cpu, dtype):
    """The chat-template path of the causal-model helper: its implied probabilities equal those of a
    likelihood that gives the batched layout's values."""
    from brainscore import BrainScoreModel
    net, tok = _BatchIndependent(dtype), _Characters()

    def batched(prompt, candidates):
        context = tok(tok.apply_chat_template([{'content': prompt.text()}]))['input_ids']
        values = _batched_layout(net, [context + tok(c)['input_ids'] for c in candidates], len(context))
        return dict(zip(candidates, values))
    base = BrainScoreModel('batched-layout', in_channels={'text'}, likelihood=batched, capability_config={
        'behavior-sessions': {'behavior:probabilities': ('likelihood', 'softmax')}})
    tables = []
    for model in (_subject('causal_lm', net, tok), base):
        session = stimulus_session(_control_stimuli(), ['behavior:probabilities'],
                                   TaskContext(response=Choice(labels='options')))
        model.participate(session)
        tables.append(session.collect('behavior:probabilities').values.tolist())
    assert tables[0] == tables[1]
