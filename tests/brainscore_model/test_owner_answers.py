"""The owner of the requested answer format gives the answer (rule 4.11; stage 2, C4).

For a task with a response, the part that answers is the owner of the task's
answer format: the raw scores (``behavior:probabilities``) of a raw-scores
Choice, else the response's first format. A task form its owner cannot serve is
refused before any compute; only bare ``behavior`` (a legacy task) keeps the
fallback to the one other part that can serve it.
"""
from itertools import combinations

import pytest

from brainscore_core.compatibility import ExecutionNotFeasibleError
from brainscore_core.contract import TaskContext
from brainscore_core.responses import Choice
from tests.brainscore_model.test_answer_ownership import (
    ARGUMENTS, BLIP2, CAUSAL_LM, LEGACY_NO_INSTRUCTION, QWEN, RATINGS, REGISTERED_STATEMENTS, SESSIONS,
    _model, _run)
from tests.brainscore_model.fixed_order_baseline import task_from
from tests.brainscore_model.test_behavior_paths import FITTING, STIMULI
from tests.fixed_order import OUTPUTS, fixed_order_owners

pytestmark = pytest.mark.unit

RAW_SCORES = TaskContext(response=Choice(['cat', 'dog'], raw_scores=True))


@pytest.mark.parametrize('parts', [QWEN, CAUSAL_LM], ids=['qwen', 'causal lm'])
def test_raw_scores_come_from_the_owner_of_the_probabilities(parts):
    calls = []
    model = _model(*parts, owners=REGISTERED_STATEMENTS[parts], calls=calls)
    session = _run(model, ['behavior:probabilities'], RAW_SCORES)
    record = session.collect('behavior:probabilities').attrs['response']
    assert (record['method'], record['category']) == ('likelihood', 'model-derived')
    assert set(calls) == {'likelihood'}                  # neither reply nor answer
    assert calls.count('likelihood') == len(STIMULI)


def test_blip2_raw_text_choice_is_refused_before_compute():
    calls = []
    model = _model(*BLIP2, owners=REGISTERED_STATEMENTS[BLIP2], calls=calls)
    task = TaskContext(response=Choice(['cat', 'dog'], raw_text=True), fitting_stimuli=FITTING)
    with pytest.raises(ExecutionNotFeasibleError,
                       match=r"'behavior:choice' is owned by 'generation'.*task settings: raw_text"):
        _run(model, ['behavior:choice'], task)
    assert calls.count('extract') == 0 and calls == []


def test_a_readout_alone_gives_probabilities_natively():
    model = _model('behavioral_readout_layer',
                   owners=REGISTERED_STATEMENTS[('behavioral_readout_layer',)])
    task = TaskContext(instruction='Which?', response=Choice(['cat', 'dog'], probabilities=True),
                       fitting_stimuli=FITTING)
    probabilities = _run(model, ['behavior:probabilities'], task).collect('behavior:probabilities')
    assert probabilities.values.shape == (len(STIMULI), 2)
    record = probabilities.attrs['response']
    assert (record['method'], record['origin']) == ('readout', 'native')


@pytest.mark.parametrize('parts', [QWEN, BLIP2], ids=['qwen', 'blip2'])
def test_a_legacy_task_without_an_instruction_still_reaches_the_readout(parts):
    """The facade sends a legacy task with no instruction; bare behavior keeps the fallback."""
    calls = []
    model = _model(*parts, owners=REGISTERED_STATEMENTS[parts], calls=calls)
    behavior = _run(model, ['behavior'], LEGACY_NO_INSTRUCTION).collect('behavior')
    assert behavior.attrs['response']['method'] == 'readout'
    assert 'answer' not in calls and 'reply' not in calls


def test_two_owners_for_choice_and_probabilities_are_refused_before_compute():
    calls = []
    owners = {'behavior': 'generation', 'behavior:rating': 'generation',
              'behavior:choice': ('likelihood', 'softmax_argmax'), 'behavior:probabilities': 'readout'}
    model = _model(*QWEN, owners=owners, calls=calls)
    task = TaskContext(instruction='Which?', response=Choice(['cat', 'dog'], probabilities=True),
                       fitting_stimuli=FITTING)
    with pytest.raises(ExecutionNotFeasibleError,
                       match=r"choices from its likelihood part and probabilities from its readout part"):
        _run(model, ['behavior:choice', 'behavior:probabilities'], task)
    assert calls == []


# --- no part answers a format channel it does not own ---------------------------------

PART_SETS = [parts for n in range(1, 5) for parts in combinations(ARGUMENTS, n)]


def _owners_or_none(parts):
    try:
        return _model(*parts, owners=fixed_order_owners(*parts))
    except ValueError:
        return None


@pytest.mark.parametrize('parts', PART_SETS, ids='+'.join)
def test_each_answered_format_comes_from_its_owner(parts):
    """Every session of SESSIONS on every part set: each answered format channel was given
    by the part that owns it (the probabilities' owner also gives the choice that comes
    with raw scores; a reaction time names the probabilities' method)."""
    model = _owners_or_none(parts)
    if model is None:
        pytest.skip('the fixed order states no owner for a format with two native parts')
    owners = model._behavioral.owners

    def owner_of(fmt):     # the stated owner, or the one part that gives the format (oracle table)
        givers = {OUTPUTS[name][0] for name in parts if fmt in OUTPUTS[name][1]}
        return owners.get(fmt) or (givers.pop() if len(givers) == 1 else None)
    for name, (description, outputs) in SESSIONS.items():
        task = task_from(description, RATINGS if description['kind'] == 'scale' else FITTING)
        requested = [c for c in outputs if c in model.out_channels]
        if not requested:
            continue
        try:
            session = _run(model, requested, task)
        except ExecutionNotFeasibleError:
            continue
        for channel in requested:
            if channel == 'behavior':
                continue                  # bare behavior: the legacy fallback stays
            record = session.collect(channel).attrs['response']
            raw_scores = getattr(task.response, 'raw_scores', False)
            owner = owner_of('behavior:probabilities' if raw_scores else channel)
            assert record['method'] == owner, (parts, name, channel)


# --- Z01: one format of a crossed likelihood/readout pair -----------------------------

CROSSED = {
    'likelihood choice': {'behavior:choice': ('likelihood', 'softmax_argmax'),
                          'behavior:probabilities': 'readout'},
    'readout choice': {'behavior:choice': ('readout', 'argmax'),
                       'behavior:probabilities': ('likelihood', 'softmax')},
}
CHOICE_AND_PROBABILITIES = TaskContext(instruction='Which?',
                                       response=Choice(['cat', 'dog'], probabilities=True),
                                       fitting_stimuli=FITTING)


@pytest.mark.parametrize('arrangement', sorted(CROSSED))
@pytest.mark.parametrize('channel', ['behavior:choice', 'behavior:probabilities'])
def test_one_format_of_a_crossed_pair_is_served_by_its_owner(arrangement, channel):
    owners = CROSSED[arrangement]
    calls = []
    model = _model('likelihood', 'behavioral_readout_layer', owners=owners, calls=calls)
    answer = _run(model, [channel], CHOICE_AND_PROBABILITIES).collect(channel)
    owner = owners[channel] if isinstance(owners[channel], str) else owners[channel][0]
    assert answer.attrs['response']['method'] == owner
    assert answer.sizes['presentation'] == len(STIMULI)
    if owner == 'likelihood':
        assert calls.count('likelihood') == len(STIMULI)
    else:
        assert 'likelihood' not in calls


@pytest.mark.parametrize('arrangement', sorted(CROSSED))
def test_both_formats_of_a_crossed_pair_are_refused_before_compute(arrangement):
    calls = []
    model = _model('likelihood', 'behavioral_readout_layer', owners=CROSSED[arrangement], calls=calls)
    with pytest.raises(ExecutionNotFeasibleError,
                       match=r"from its (likelihood|readout) part and probabilities from its "
                             r"(readout|likelihood) part in one session; request one format in the session"):
        _run(model, ['behavior:choice', 'behavior:probabilities'], CHOICE_AND_PROBABILITIES)
    assert calls == []
