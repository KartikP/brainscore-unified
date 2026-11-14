"""Conformance test T2 (Response Format Standard, section 11.1), for a BrainScoreModel.

Fake parts give an unparseable reply, no reply, and a non-finite likelihood
row. Through ``participate`` every trial is kept with its state, its raw
response and a reason; a conversion of an invalid row stays invalid; nothing
is filled. The core version of this test, with a custom subject, is core
``tests/test_conformance_t2_invalid_missing.py``.
"""
import numpy as np
import pytest

from brainscore_core.contract import TaskContext
from brainscore_core.responses import Choice, Scale, Text
from tests.brainscore_model.test_task_responses import _ask, _model

pytestmark = pytest.mark.unit

REPLIES = {'a': 'cat', 'b': 'a bird', 'c': None, 'd': 'dog'}


def test_a_reply_choice_keeps_valid_invalid_and_missing_trials():
    model, _ = _model(replies=REPLIES)
    choice = _ask(model, TaskContext(instruction='Which?', response=Choice(['cat', 'dog'])),
                  ['behavior:choice']).collect('behavior:choice')
    assert list(choice['response_state'].values) == ['valid', 'invalid', 'missing', 'valid']
    reasons = list(choice['response_reason'].values)
    assert reasons[0] == reasons[3] == '' and 'no label' in reasons[1] and reasons[2] == 'no answer'
    raw = list(choice['raw_answer'].values)
    assert raw[0] == 'cat' and raw[1] == 'a bird' and raw[2] != raw[2]   # None kept as a null
    assert np.isnan(choice.values[1:3]).all()                           # nothing filled


@pytest.mark.parametrize('response, channel', [(Scale(1, 7), 'behavior:rating'),
                                               (Text(), 'behavior:text')])
def test_a_missing_reply_is_missing_for_every_reply_format(response, channel):
    model, _ = _model(replies={'a': '3', 'b': 'x', 'c': None, 'd': '9'})
    answer = _ask(model, TaskContext(instruction='Say.', response=response), [channel]).collect(channel)
    states = list(answer['response_state'].values)
    assert states[2] == 'missing'
    if channel == 'behavior:rating':
        assert states == ['valid', 'invalid', 'missing', 'invalid']    # 9 is outside 1 to 7
    else:
        assert states == ['valid', 'valid', 'missing', 'valid']


def test_a_conversion_of_a_non_finite_likelihood_row_stays_invalid():
    scores = {'a': {'cat': -1.0, 'dog': -2.0}, 'b': {'cat': np.nan, 'dog': -1.0},
              'c': {'cat': -np.inf, 'dog': -np.inf}, 'd': {'cat': -3.0, 'dog': -1.0}}
    model, _ = _model(scores=scores)
    session = _ask(model, TaskContext(instruction='Which?',
                                      response=Choice(['cat', 'dog'], probabilities=True)),
                   ['behavior:choice', 'behavior:probabilities'])
    choice = session.collect('behavior:choice')
    probabilities = session.collect('behavior:probabilities')
    assert list(choice['response_state'].values) == ['valid', 'invalid', 'invalid', 'valid']
    assert list(probabilities['response_state'].values) == ['valid', 'invalid', 'invalid', 'valid']
    assert np.isnan(choice.values[1:3]).all()
    np.testing.assert_array_equal(choice.values[[0, 3]], [[1, 0], [0, 1]])
