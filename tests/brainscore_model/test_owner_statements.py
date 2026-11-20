"""Owner statements: a part for a native format, ``(part, conversion)`` for a derived one (rule 4.11).

The statement form and its construction checks. Each test
builds through ``BrainScoreModel(...)``; the run-time tests participate in a
stimulus session.
"""
import re

import pytest

from brainscore import BrainScoreModel
from brainscore_core.compatibility import ExecutionNotFeasibleError
from brainscore_core.contract import TaskContext
from brainscore_core.responses import Choice
from tests.brainscore_model.test_answer_ownership import _model, _run
from tests.brainscore_model.test_behavior_paths import FITTING

pytestmark = pytest.mark.unit

QWEN = ('answer', 'reply', 'likelihood', 'behavioral_readout_layer')
QWEN_STATEMENT = {'behavior:rating': 'generation',
                  'behavior:probabilities': ('likelihood', 'softmax')}
CHOICE = TaskContext(instruction='Which?', response=Choice(['cat', 'dog']),
                     fitting_stimuli=FITTING)
FORMATS = ("['behavior:choice', 'behavior:log_probabilities', 'behavior:probabilities', "
           "'behavior:rating', 'behavior:text']")


def _readout(owners=None):
    return _model('behavioral_readout_layer', owners=owners)


def test_a_readout_only_model_builds_without_a_statement():
    assert 'behavior:choice' not in _readout().out_channels   # derived, not stated


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
    ({'behavior:probabilities': 'likelihood'}, r"names 'likelihood' for 'behavior:probabilities', "
                                              r"but no likelihood= part is wired"),
    ({'behavior:choise': 'readout'},
     r"names 'behavior:choise', which is not a response format\. Formats: " + re.escape(FORMATS)),
    ({'behavior:choice': 'readuot'}, r"parts are 'generation', 'likelihood', 'readout'"),
    ({'behavior:choice': ('readout', 'argmx')}, r"conversions are 'argmax', 'softmax', 'softmax_argmax'"),
], ids=['native string for a derived format', 'conversion on a native format', 'wrong conversion',
        'one-tuple', 'list', 'rt', 'part not wired', 'misspelled key',
        'misspelled part', 'misspelled conversion'])
def test_a_wrong_statement_is_refused_with_what_to_write(owners, message):
    with pytest.raises(ValueError, match=message):
        _readout(owners)


@pytest.mark.parametrize('statement', ['generation', 'readout', ('readout', 'argmax')])
def test_a_statement_for_bare_behavior_is_refused_with_the_line_to_delete(statement):
    """Bare behavior takes no statement: it carries the answer under the response's own format."""
    with pytest.raises(ValueError, match=r"names 'behavior'\. Delete this line"):
        _model('answer', 'behavioral_readout_layer', owners={'behavior': statement})


@pytest.mark.parametrize('parts', [('answer',), ('behavioral_readout_layer',),
                                   ('answer', 'behavioral_readout_layer'), QWEN])
def test_bare_behavior_is_declared_when_an_answer_or_a_readout_is_wired(parts):
    owners = {'behavior:rating': 'generation'} if 'reply' in parts else {}
    assert 'behavior' in _model(*parts, owners=owners).out_channels


@pytest.mark.parametrize('parts', [('reply',), ('likelihood',), ('reply', 'likelihood')])
def test_bare_behavior_is_not_declared_without_an_answer_or_a_readout(parts):
    assert 'behavior' not in _model(*parts).out_channels


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


def test_log_probabilities_come_natively_from_the_likelihood():
    model = _model('reply', 'likelihood', owners={'behavior:probabilities': ('likelihood', 'softmax')})
    assert 'behavior:log_probabilities' in model.out_channels    # no statement needed: one native part
    session = _run(model, ['behavior:log_probabilities'],
                   TaskContext(response=Choice(['cat', 'dog'], verbatim=True)))
    scores = session.collect('behavior:log_probabilities')
    record = scores.attrs['response']
    assert (record['format'], record['method'], record['origin'], record['category']) == (
        'behavior:log_probabilities', 'likelihood', 'native', 'model-derived')
    assert (record['parameters']['units'], record['parameters']['layout']) == ('nats', 'continuation')
    assert scores.values[0].tolist() == [0.0, -1.0]      # the fake likelihood's log scores
    assert scores.values.dtype == 'float64'


def test_log_probabilities_take_no_conversion_statement():
    with pytest.raises(ValueError, match="gives 'behavior:log_probabilities' natively"):
        _model('reply', 'likelihood', owners={'behavior:log_probabilities': ('likelihood', 'softmax')})


def test_a_readout_does_not_declare_log_probabilities():
    assert 'behavior:log_probabilities' not in _readout().out_channels


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


# --- a refusal names the statement to write -------------------------------------------

@pytest.mark.parametrize('parts, channel, line', [
    (('behavioral_readout_layer',), 'behavior:choice',
     "capability_config={'behavior-sessions': {'behavior:choice': ('readout', 'argmax')}}"),
    (('likelihood',), 'behavior:choice',
     "capability_config={'behavior-sessions': {'behavior:choice': ('likelihood', 'softmax_argmax')}}"),
    (('likelihood',), 'behavior:probabilities',
     "capability_config={'behavior-sessions': {'behavior:probabilities': ('likelihood', 'softmax')}}"),
], ids=['readout choice', 'likelihood choice', 'likelihood probabilities'])
def test_a_derived_format_without_its_statement_is_refused_with_the_line_to_write(parts, channel, line):
    calls = []
    model = _model(*parts, calls=calls)
    assert channel not in model.out_channels
    task = TaskContext(instruction='Which?', response=Choice(['cat', 'dog']), fitting_stimuli=FITTING)
    with pytest.raises(ExecutionNotFeasibleError) as raised:
        _run(model, [channel], task)
    assert f"{channel!r} is derived by a wired part; state it: {line}." in str(raised.value)
    assert calls == []


def test_two_parts_that_derive_a_format_are_both_named():
    model = _model('likelihood', 'behavioral_readout_layer')
    task = TaskContext(instruction='Which?', response=Choice(['cat', 'dog']), fitting_stimuli=FITTING)
    with pytest.raises(ExecutionNotFeasibleError) as raised:
        _run(model, ['behavior:choice'], task)
    assert ("'behavior:choice' is derived by 2 wired parts; state one: "
            "capability_config={'behavior-sessions': {'behavior:choice': ('likelihood', 'softmax_argmax')}} or "
            "capability_config={'behavior-sessions': {'behavior:choice': ('readout', 'argmax')}}.") in str(raised.value)


# --- the method 'similarity' (ADR-0081 decision 2, rule 4.11) -------------------------

def test_an_answer_part_can_state_the_method_similarity():
    """The answer part states similarity in place of generation; its choice records it."""
    model = _model('answer', owners={'behavior:choice': 'similarity'})
    assert 'behavior:choice' in model.out_channels
    choice = _run(model, ['behavior:choice'], CHOICE).collect('behavior:choice')
    assert choice.attrs['response']['method'] == 'similarity'


@pytest.mark.parametrize('parts, owners', [
    (('behavioral_readout_layer',), {'behavior:choice': 'similarity'}),
    (('answer', 'reply'), {'behavior:rating': 'similarity'}),
], ids=['no answer part', 'a format the answer part does not give'])
def test_a_similarity_statement_needs_an_answer_part_that_gives_the_format(parts, owners):
    with pytest.raises(ValueError, match='similarity'):
        _model(*parts, owners=owners)
