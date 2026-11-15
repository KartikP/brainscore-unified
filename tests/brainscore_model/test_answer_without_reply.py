"""An answer part with a likelihood that owns the probabilities and no reply (stage 2, C3 review Y02).

The likelihood scores each prompt before the answer part answers it, as it does
next to a reply, so the probabilities and the reaction time come from the
likelihood.
"""
import pytest

from brainscore_core.contract import TaskContext
from brainscore_core.responses import Choice
from tests.brainscore_model.test_answer_ownership import _model, _run
from tests.brainscore_model.test_behavior_paths import FITTING, STIMULI

pytestmark = pytest.mark.unit


# --- Y02: answer and likelihood without a reply ---------------------------------------

Y02 = {
    'answer+likelihood': (('answer', 'likelihood'),
                          {'behavior:probabilities': ('likelihood', 'softmax')}),
    'answer+likelihood+readout': (('answer', 'likelihood', 'behavioral_readout_layer'),
                                  {'behavior': 'generation',
                                   'behavior:probabilities': ('likelihood', 'softmax')}),
}


@pytest.mark.parametrize('settings, channel', [({}, 'behavior:probabilities'),
                                               ({'reaction_time': 'surprisal'}, 'behavior:rt')],
                         ids=['probabilities', 'reaction time'])
@pytest.mark.parametrize('case', sorted(Y02))
def test_answer_and_likelihood_without_a_reply_give_the_likelihood_probabilities(case, settings, channel):
    parts, owners = Y02[case]
    calls = []
    model = _model(*parts, owners=owners, calls=calls)
    assert channel in model.out_channels
    task = TaskContext(instruction='Which?', response=Choice(['cat', 'dog'], **settings),
                       fitting_stimuli=FITTING)
    session = _run(model, ['behavior:choice', channel], task)
    assert session.collect('behavior:choice').attrs['response']['method'] == 'generation'
    record = session.collect(channel).attrs['response']
    assert (record['parameters']['source_method'] if channel == 'behavior:rt'
            else record['method']) == 'likelihood'
    assert calls.count('likelihood') == calls.count('answer') == len(STIMULI)
    assert 'extract' not in calls
