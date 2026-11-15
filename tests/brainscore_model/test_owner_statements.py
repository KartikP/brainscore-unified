"""Owner statements: a part for a native format, ``(part, conversion)`` for a derived one (rule 4.11).

Stage 2, chunk C2: the statement form and its construction checks. Each test
builds through ``BrainScoreModel(...)``; the run-time tests participate in a
stimulus session.
"""
import re

import pytest

from brainscore import BrainScoreModel
from brainscore_core.contract import TaskContext
from brainscore_core.responses import Choice
from tests.brainscore_model.test_answer_ownership import _model, _run
from tests.brainscore_model.test_behavior_paths import FITTING

pytestmark = pytest.mark.unit

QWEN = ('answer', 'reply', 'likelihood', 'behavioral_readout_layer')
QWEN_STATEMENT = {'behavior': 'generation', 'behavior:rating': 'generation',
                  'behavior:probabilities': ('likelihood', 'softmax')}
CHOICE = TaskContext(instruction='Which?', response=Choice(['cat', 'dog']),
                     fitting_stimuli=FITTING)
FORMATS = ("['behavior', 'behavior:choice', 'behavior:probabilities', "
           "'behavior:rating', 'behavior:text']")


def _readout(owners=None):
    return _model('behavioral_readout_layer', owners=owners)


def test_a_readout_only_model_builds_without_a_statement():
    assert 'behavior:choice' not in _readout().out_channels   # derived, not stated (C3)


@pytest.mark.parametrize('owners, message', [
    ({'behavior:choice': 'readout'},
     r"'readout' gives 'behavior:choice' only by a conversion\. "
     r"Write 'behavior:choice': \('readout', 'argmax'\)"),
    ({'behavior:probabilities': ('readout', 'argmax')},
     r"natively\. Write 'behavior:probabilities': 'readout'"),
    ({'behavior:choice': ('readout', 'softmax_argmax')}, r"Write \('readout', 'argmax'\)"),
    ({'behavior:choice': ('readout',)}, r"a statement is a part \('readout'\) or a tuple"),
    ({'behavior:choice': ['readout', 'argmax']}, r"a statement is a part"),
    ({'behavior:rt': 'readout'}, r"follows the owner of 'behavior:probabilities'"),
    ({'behavior': ('readout', 'argmax')}, r"'behavior' takes a part, not a conversion"),
    ({'behavior:probabilities': 'likelihood'}, r"names 'likelihood' for 'behavior:probabilities', "
                                              r"but no likelihood= part is wired"),
    ({'behavior:choise': 'readout'},
     r"names 'behavior:choise', which is not a response format\. Formats: " + re.escape(FORMATS)),
    ({'behavior:choice': 'readuot'}, r"parts are 'generation', 'likelihood', 'readout'"),
    ({'behavior:choice': ('readout', 'argmx')}, r"conversions are 'argmax', 'softmax', 'softmax_argmax'"),
], ids=['native string for a derived format', 'conversion on a native format', 'wrong conversion',
        'one-tuple', 'list', 'rt', 'conversion on bare behavior', 'part not wired', 'misspelled key',
        'misspelled part', 'misspelled conversion'])
def test_a_wrong_statement_is_refused_with_what_to_write(owners, message):
    with pytest.raises(ValueError, match=message):
        _readout(owners)


def test_confidence_is_no_response_format():
    """``behavior:confidence`` is deleted: a statement naming it lists the formats, and no
    reply model declares it."""
    with pytest.raises(ValueError, match=r"names 'behavior:confidence', which is not a response "
                                         r"format\. Formats: " + re.escape(FORMATS)):
        _model('reply', owners={'behavior:confidence': 'generation'})
    assert 'behavior:confidence' not in _model(*QWEN, owners=QWEN_STATEMENT).out_channels


def test_the_qwen_statement_answers_choice_by_generation_and_probabilities_by_softmax():
    model = _model(*QWEN, owners=QWEN_STATEMENT)
    session = _run(model, ['behavior:choice', 'behavior:probabilities'],
                   TaskContext(instruction='Which?', response=Choice(['cat', 'dog'])))
    assert session.collect('behavior:choice').attrs['response']['method'] == 'generation'
    probabilities = session.collect('behavior:probabilities').attrs['response']
    assert (probabilities['method'], probabilities['conversion']) == ('likelihood', 'softmax')


def test_the_qwen_set_without_a_rating_owner_is_refused():
    owners = {k: v for k, v in QWEN_STATEMENT.items() if k != 'behavior:rating'}
    with pytest.raises(ValueError, match=r"the parts \['generation', 'readout'\] give 'behavior:rating' "
                                         r"natively\. State the owner: capability_config="
                                         r"\{'behavior-sessions': \{'behavior:rating': 'generation'\}\} "
                                         r"\(or 'readout'\)"):
        _model(*QWEN, owners=owners)


def test_the_qwen_set_without_a_probabilities_owner_takes_them_from_the_readout():
    owners = {k: v for k, v in QWEN_STATEMENT.items() if k != 'behavior:probabilities'}
    session = _run(_model(*QWEN, owners=owners), ['behavior:choice', 'behavior:probabilities'], CHOICE)
    record = session.collect('behavior:probabilities').attrs['response']
    assert (record['method'], record['origin']) == ('readout', 'native')


def test_raw_scores_from_a_stated_likelihood_are_log_scores():
    model = _model('reply', 'likelihood', owners={'behavior:probabilities': ('likelihood', 'softmax')})
    session = _run(model, ['behavior:probabilities'],
                   TaskContext(response=Choice(['cat', 'dog'], raw_scores=True)))
    scores = session.collect('behavior:probabilities')
    assert scores.attrs['response']['category'] == 'model-derived'
    assert scores.values[0].tolist() == [0.0, -1.0]      # the fake likelihood's log scores


def test_a_redundant_native_statement_is_accepted():
    model = BrainScoreModel('m', extractor={'text': lambda *a, **k: None}, in_channels={'text'},
                            behavioral_readout_layer='late',
                            capability_config={'behavior-sessions': {'behavior:rating': 'readout'}})
    assert 'behavior:rating' in model.out_channels


def test_no_message_names_a_word_a_registrant_cannot_write():
    """Each 'Write' and 'State' in a refusal names only formats, parts and conversions."""
    for owners in ({'behavior:choice': 'readout'}, {'behavior:probabilities': ('readout', 'argmax')},
                   {'behavior:choice': ('readout', 'softmax_argmax')}):
        with pytest.raises(ValueError) as refused:
            _readout(owners)
        assert 'log_probabilities' not in str(refused.value)
