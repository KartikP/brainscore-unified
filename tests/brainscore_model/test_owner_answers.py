"""The owner of the requested answer format gives the answer (rule 4.11; stage 2, C4).

For a task with a response, the part that answers is the owner of the task's
answer format: the raw scores (``behavior:probabilities``) of a raw-scores
Choice, else the response kind's own format. A task form its owner cannot serve is
refused before any compute; only bare ``behavior`` (a legacy task) keeps the
fallback to the one other part that can serve it.
"""
import pytest

from brainscore_core.compatibility import ExecutionNotFeasibleError
from brainscore_core.contract import TaskContext
from brainscore_core.responses import Choice
from tests.brainscore_model.test_answer_ownership import (
    BLIP2, CAUSAL_LM, LEGACY_NO_INSTRUCTION, QWEN, REGISTERED_STATEMENTS, _model, _run)
from tests.brainscore_model.stage2_answers_baseline import RATINGS, SESSIONS, task_from
from tests.brainscore_model.test_behavior_paths import FITTING, STIMULI

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

G, L = 'generation', ('likelihood', 'softmax')
# Every part set with the statement a registrant writes for it.
STATEMENTS = {
    ('answer',): {}, ('reply',): {}, ('likelihood',): {}, ('behavioral_readout_layer',): {},
    ('answer', 'reply'): {}, ('answer', 'likelihood'): {'behavior:probabilities': L},
    ('reply', 'likelihood'): {'behavior:probabilities': L},
    ('answer', 'reply', 'likelihood'): {'behavior:probabilities': L},
    ('answer', 'behavioral_readout_layer'): {'behavior': G},
    ('reply', 'behavioral_readout_layer'): {'behavior:rating': G},
    ('likelihood', 'behavioral_readout_layer'): {'behavior:choice': ('likelihood', 'softmax_argmax'),
                                                 'behavior:probabilities': L},
    ('answer', 'reply', 'behavioral_readout_layer'): {'behavior': G, 'behavior:rating': G},
    ('answer', 'likelihood', 'behavioral_readout_layer'): {'behavior': G, 'behavior:probabilities': L},
    ('reply', 'likelihood', 'behavioral_readout_layer'): {'behavior:rating': G, 'behavior:probabilities': L},
    ('answer', 'reply', 'likelihood', 'behavioral_readout_layer'): {
        'behavior': G, 'behavior:rating': G, 'behavior:probabilities': L},
}


@pytest.mark.parametrize('parts', sorted(STATEMENTS), ids='+'.join)
def test_each_answered_format_comes_from_its_owner(parts):
    """Every session of SESSIONS on every part set: each answered format channel was given
    by the part that owns it (the probabilities' owner also gives the choice that comes
    with raw scores; a reaction time names the probabilities' method)."""
    model = _model(*parts, owners=STATEMENTS[parts])
    owners = model._behavioral.owners
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
            owner = owners['behavior:probabilities' if raw_scores else channel]
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


# --- stage 3, S2: the formats the session requests choose the parts -------------------

def _count_fits(model):
    """Wrap the readout fit so each call is counted; the fit itself still runs."""
    fits, fit = [], model._behavioral.fit_behavioral_readout
    model._behavioral.fit_behavioral_readout = lambda stimuli: fits.append(1) or fit(stimuli)
    return fits


def test_a_readout_alone_answers_probabilities_with_one_fit():
    model = _model('behavioral_readout_layer',
                   owners=REGISTERED_STATEMENTS[('behavioral_readout_layer',)])
    fits = _count_fits(model)
    answer = _run(model, ['behavior:probabilities'], CHOICE_AND_PROBABILITIES).collect(
        'behavior:probabilities')
    assert answer.attrs['response']['method'] == 'readout'
    assert fits == [1]


@pytest.mark.parametrize('owners', [CROSSED['likelihood choice'], CROSSED['readout choice'],
                                    {'behavior:choice': ('readout', 'argmax')}],
                         ids=['likelihood choice', 'readout choice', 'readout owns both'])
def test_one_configured_task_serves_two_sessions_of_different_formats(owners):
    """Choice only, then probabilities only, on the same task: each format's owner answers,
    and the readout is fitted once in total (the setup is keyed by the task alone)."""
    calls = []
    model = _model('likelihood', 'behavioral_readout_layer', owners=owners, calls=calls)
    fits = _count_fits(model)
    for channel in ('behavior:choice', 'behavior:probabilities'):
        del calls[:]
        answer = _run(model, [channel], CHOICE_AND_PROBABILITIES).collect(channel)
        owner = owners.get(channel, 'readout')
        owner = owner if isinstance(owner, str) else owner[0]
        assert answer.attrs['response']['method'] == owner
        assert calls.count('likelihood') == (len(STIMULI) if owner == 'likelihood' else 0)
    assert fits == [1]


def test_a_choice_only_session_does_not_call_the_probability_owner():
    """Qwen's reply gives the choice; its likelihood owns the probabilities and is called
    only when a session requests them."""
    calls = []
    model = _model(*QWEN, owners=REGISTERED_STATEMENTS[QWEN], calls=calls)
    choice = _run(model, ['behavior:choice'], CHOICE_AND_PROBABILITIES).collect('behavior:choice')
    assert choice.attrs['response']['method'] == 'generation'
    assert calls.count('reply') == len(STIMULI) and 'likelihood' not in calls
    del calls[:]
    _run(model, ['behavior:choice', 'behavior:probabilities'], CHOICE_AND_PROBABILITIES)
    assert calls.count('reply') == calls.count('likelihood') == len(STIMULI)
