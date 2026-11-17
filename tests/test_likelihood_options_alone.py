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
from tests.test_ewok import likelihood_subject, tiny_lm  # noqa: F401  (tiny_lm is a fixture)

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
