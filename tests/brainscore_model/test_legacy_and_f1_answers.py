"""The legacy answer on bare ``behavior`` and the answer on ``behavior:choice``, as they are today.

ROAR and induced dyslexia send a legacy task on bare ``behavior`` now and will send a plain
Choice on ``behavior:choice``. These tests pin, for the stand-in parts of each registered part
set, what each form returns and which part answers it, so that the change of request moves no
value without failing here.
"""
import json
import struct

import numpy as np
import pytest

from brainscore_core.contract import TaskContext
from brainscore_core.responses import Choice
from tests.brainscore_model.test_answer_ownership import BLIP2, QWEN, READOUT, REGISTERED_STATEMENTS, _model, _run
from tests.brainscore_model.test_behavior_paths import FITTING

pytestmark = pytest.mark.unit

LABELS = ['cat', 'dog']
LEGACY = TaskContext(task_type='probabilities', label_set=LABELS, instruction='Which?', fitting_stimuli=FITTING)
PROBABILITIES = TaskContext(instruction='Which?', response=Choice(LABELS),
                           fitting_stimuli=FITTING)
PLAIN = TaskContext(instruction='Which?', response=Choice(LABELS), fitting_stimuli=FITTING)


def _answers(parts, outputs, task):
    """The collected answer of one output and the calls the stand-in parts received."""
    calls = []
    model = _model(*parts, owners=REGISTERED_STATEMENTS[parts], calls=calls)
    return _run(model, outputs, task).collect(outputs[0]), calls


def _bits(rows):
    return [struct.pack('<d', float(v)).hex() for v in np.asarray(rows, dtype='<f8').ravel()]


def test_readout_legacy_rows_are_the_source_rows_of_the_choice_and_the_choice_is_their_argmax():
    legacy, legacy_calls = _answers(READOUT, ['behavior'], LEGACY)
    choice, choice_calls = _answers(READOUT, ['behavior:choice'], PLAIN)
    probabilities, _ = _answers(READOUT, ['behavior:probabilities'], PROBABILITIES)
    assert legacy.coords['choice'].values.tolist() == choice.coords['choice'].values.tolist() == LABELS
    assert legacy_calls == choice_calls == ['extract', 'extract']       # one fit and one read-out each
    source = [json.loads(text) for text in choice.indexes['presentation'].get_level_values('raw_answer')]
    assert _bits(legacy.values) == _bits(source)                          # float64, bit for bit
    assert _bits(probabilities.values) == _bits(legacy.values)               # the stored rows: float64 too
    argmax = np.eye(len(LABELS))[np.asarray(legacy.values).argmax(axis=1)]    # readout order, first wins a tie
    assert _bits(argmax) == _bits(choice.values)
    record = choice.attrs['response']
    assert (record['origin'], record['method'], record['conversion'], record['parameters']['option_order']) == (
        'derived', 'readout', 'argmax', 'readout')


@pytest.mark.parametrize('parts', [BLIP2, QWEN], ids=['answer+readout', 'answer+reply+likelihood+readout'])
def test_the_legacy_one_hot_of_an_answer_part_equals_the_choice_one_hot(parts):
    legacy, _ = _answers(parts, ['behavior'], LEGACY)
    choice, _ = _answers(parts, ['behavior:choice'], PLAIN)
    assert legacy.values.tolist() == choice.values.tolist() == [[1.0, 0.0]] * 3
    assert _bits(legacy.values) == _bits(choice.values)


def test_answer_and_readout_set_answers_both_forms_through_the_answer_part():
    for outputs, task in ((['behavior'], LEGACY), (['behavior:choice'], PLAIN)):
        answer, calls = _answers(BLIP2, outputs, task)
        assert calls == ['answer'] * 3 and 'extract' not in calls


def test_the_full_part_set_answers_legacy_and_the_plain_choice_through_answer():
    """Within generation a plain Choice goes to answer= when it is wired."""
    _, legacy_calls = _answers(QWEN, ['behavior'], LEGACY)
    choice, choice_calls = _answers(QWEN, ['behavior:choice'], PLAIN)
    assert legacy_calls == ['answer'] * 3
    assert choice_calls == ['answer'] * 3
    assert choice.attrs['response']['method'] == 'generation' and choice.attrs['response']['origin'] == 'native'


def _counting_fits(model):
    fits, fit = [], model._behavioral.fit_behavioral_readout
    model._behavioral.fit_behavioral_readout = lambda fitting: fits.append(len(fitting)) or fit(fitting)
    return fits


def test_the_readout_is_fitted_by_the_first_session_that_requests_its_format():
    """The requested formats decide the fit, as they decide the passes: one fit per
    configured task, at the start of the first session that needs the readout."""
    calls = []
    model = _model(*BLIP2, owners=REGISTERED_STATEMENTS[BLIP2], calls=calls)
    fits = _counting_fits(model)
    choice = _run(model, ['behavior:choice'], PLAIN).collect('behavior:choice')
    assert fits == [] and calls == ['answer'] * 3                  # no fit, no extraction
    assert choice.values.tolist() == [[1.0, 0.0]] * 3
    probabilities = _run(model, ['behavior:probabilities'], PLAIN).collect('behavior:probabilities')
    assert fits == [len(FITTING)]
    assert probabilities.attrs['response']['method'] == 'readout'
    _run(model, ['behavior:choice', 'behavior:probabilities'], PLAIN)
    assert fits == [len(FITTING)]                                   # still one fit


def test_a_readout_that_derives_the_choice_is_fitted_for_a_choice_session():
    model = _model(*READOUT, owners=REGISTERED_STATEMENTS[READOUT])
    fits = _counting_fits(model)
    choice = _run(model, ['behavior:choice'], PLAIN).collect('behavior:choice')
    assert fits == [len(FITTING)] and choice.attrs['response']['method'] == 'readout'


@pytest.mark.parametrize('parts, fitted', [(BLIP2, []), (QWEN, []), (READOUT, [len(FITTING)])],
                         ids=['answer+readout', 'answer+reply+likelihood+readout', 'readout'])
def test_a_legacy_task_is_fitted_as_before(parts, fitted):
    model = _model(*parts, owners=REGISTERED_STATEMENTS[parts])
    fits = _counting_fits(model)
    _run(model, ['behavior'], LEGACY)
    assert fits == fitted
