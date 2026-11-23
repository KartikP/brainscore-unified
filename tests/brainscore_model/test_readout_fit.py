"""The readout is fitted by the first session that requests a format it gives (rule 4.5).

A fit is seen as a feature extraction of the fitting stimuli.
"""
import pytest

from brainscore import BrainScoreModel
from brainscore_core.contract import TaskContext
from brainscore_core.responses import Choice
from tests.brainscore_model.test_answer_ownership import _run
from tests.brainscore_model.test_behavior_paths import FITTING, _features

pytestmark = pytest.mark.unit

PLAIN = TaskContext(instruction='Which?', response=Choice(['cat', 'dog']), fitting_stimuli=FITTING)
FITTING_IDS = list(FITTING['stimulus_id'])


def _model(calls, answer=True):
    """An answer part and a readout (the answer owns the choice), or a readout alone that
    states the choice by argmax. Each feature extraction records the stimuli it reads."""
    def extract(stimuli, layers, **kwargs):
        calls.append(('extract', list(stimuli['stimulus_id'])))
        return _features(stimuli, layers, **kwargs)
    parts = {'answer': lambda stimulus_row, instruction, label_set: calls.append(('answer',)) or label_set[0]}
    owners = {} if answer else {'behavior:choice': ('readout', 'argmax')}
    return BrainScoreModel('fitted', extractor={'text': extract}, in_channels={'text'},
                           behavioral_readout_layer='late', answer_owners=owners,
                           **(parts if answer else {}))


def _fits(calls):
    return [call for call in calls if call == ('extract', FITTING_IDS)]


def test_the_readout_is_fitted_by_the_first_session_that_requests_its_format():
    """The requested formats decide the fit, as they decide the passes: one fit per
    configured task, at the start of the first session that needs the readout."""
    calls = []
    model = _model(calls)
    choice = _run(model, ['behavior:choice'], PLAIN).collect('behavior:choice')
    assert _fits(calls) == [] and ('answer',) in calls               # no fit
    assert choice.values.tolist() == [[1.0, 0.0]] * 3
    probabilities = _run(model, ['behavior:probabilities'], PLAIN).collect('behavior:probabilities')
    assert len(_fits(calls)) == 1
    assert probabilities.attrs['response']['method'] == 'readout'
    _run(model, ['behavior:choice', 'behavior:probabilities'], PLAIN)
    assert len(_fits(calls)) == 1                                       # still one fit


def test_a_readout_that_derives_the_choice_is_fitted_for_a_choice_session():
    calls = []
    choice = _run(_model(calls, answer=False), ['behavior:choice'], PLAIN).collect('behavior:choice')
    assert len(_fits(calls)) == 1 and choice.attrs['response']['method'] == 'readout'
