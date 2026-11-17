"""The session plan: built once when a session starts, read by the session parts.

``participate`` builds one plan (``_model/plan.py``) before any setup or compute;
a combination that cannot run is refused there. The calls a session makes to
the parts and the extractor are the calls of the code before the plan
(``tests/fixtures/session_call_log.json``, written at unified ``1808a13``): a
session that ran makes the same calls in the same order; a refused session is
refused again, after the same calls or fewer (a refusal may only move earlier).
"""
import json

import pytest

from brainscore import BrainScoreModel
from brainscore_core.compatibility import ExecutionNotFeasibleError
from brainscore_core.contract import TaskContext
from brainscore_core.responses import Choice
from brainscore_core.streaming_helpers import stimulus_session
from tests.brainscore_model.session_call_log import FIXTURE, capture
from tests.brainscore_model.test_behavior_paths import FITTING, STIMULI, _features

pytestmark = pytest.mark.unit


def test_every_session_makes_the_calls_of_the_code_before_the_plan():
    expected = json.loads(FIXTURE.read_text(encoding='utf-8'))
    found = capture()
    assert sorted(found) == sorted(expected)
    for name, before in expected.items():
        after = found[name]
        assert after['refused'] == before['refused'], name
        if before['refused']:
            assert after['calls'] == before['calls'][:len(after['calls'])], name
        else:
            assert after['calls'] == before['calls'], name


def _counting(calls):
    def extract(stimuli, layers, **kwargs):
        calls.append(('extract', tuple(layers)))
        return _features(stimuli, layers, **kwargs)
    return BrainScoreModel('planned', extractor={'text': extract}, in_channels={'text'},
                           region_layer_map={'IT': 'late'}, behavioral_readout_layer='late',
                           capability_config={'behavior-sessions': {'behavior:choice': ('readout', 'argmax')}})


def test_one_plan_per_session_and_none_after_it(monkeypatch):
    from brainscore._model import plan as plan_module
    built = []
    original = plan_module.plan_session
    monkeypatch.setattr(plan_module, 'plan_session',
                        lambda *a, **k: built.append(1) or original(*a, **k))
    model = _counting([])
    session = stimulus_session(STIMULI, ['neural:IT'], streaming=True)
    model.participate(session)
    assert built == [1] and model._session_plan is None


def test_trial_to_trial_variance_is_refused_on_a_stream_too_before_any_pass():
    calls = []
    for streaming in (False, True):
        session = stimulus_session(STIMULI, ['neural:IT'], settings={'require_variance': True},
                                   streaming=streaming)
        with pytest.raises(ExecutionNotFeasibleError, match='trial-to-trial variance'):
            _counting(calls).participate(session)
    assert calls == []


@pytest.mark.parametrize('outputs, task, settings, match', [
    (['behavior:choice'], None, {}, 'needs a task'),
    (['neural:IT'], TaskContext(instruction='Which?', response=Choice(['cat', 'dog']), fitting_stimuli=FITTING),
     {'time_bins': [(0, 50), (50, 100)]}, 'one time bin'),
    (['behavior:probabilities'], TaskContext(task_type='probabilities', fitting_stimuli=FITTING), {},
     r'need TaskContext\(response=...\)'),
])
def test_a_task_session_that_cannot_run_is_refused_by_the_plan_before_any_call(outputs, task, settings, match):
    calls = []
    with pytest.raises(ExecutionNotFeasibleError, match=match):
        _counting(calls).participate(stimulus_session(STIMULI, outputs, task, **settings))
    assert calls == []


def _reply_model(calls, memory=None):
    def reply(prompt):
        calls.append(('reply', prompt.stimulus_id))
        return 'cat'

    def extract(stimuli, layers, **kwargs):
        calls.append(('extract', tuple(layers)))
        return _features(stimuli, layers, **kwargs)
    return BrainScoreModel('chooses-once', extractor={'text': extract}, in_channels={'text'},
                           behavioral_readout_layer='late', reply=reply, memory=memory,
                           capability_config={'behavior-sessions': {
                               'behavior:rating': 'generation', 'behavior:probabilities': 'readout'}})


@pytest.mark.parametrize('delivery', ['table', 'per trial', 'stream'])
def test_every_selection_is_made_once_per_session(delivery, monkeypatch):
    """The answering part and, within generation, the answer or the reply part: once each.

    Before the plan the answering part was chosen once in the driver, once per format
    of the Choice in the set-up (three), once in the fit and once per answered batch
    (6 for a table, 5 + n for n trials one at a time or streamed: 8 here), and the
    answer-or-reply test ran once per batch (1, or n).
    """
    from brainscore._model.behavioral import BehavioralReadout
    chosen, original = [], BehavioralReadout._choose_path
    monkeypatch.setattr(BehavioralReadout, '_choose_path',
                        lambda self, *a, **k: chosen.append(1) or original(self, *a, **k))
    tested, by_answer = [], BehavioralReadout.by_answer
    monkeypatch.setattr(BehavioralReadout, 'by_answer',
                        lambda self, *a, **k: tested.append(1) or by_answer(self, *a, **k))
    calls = []
    model = _reply_model(calls, memory='trial_history' if delivery == 'per trial' else None)
    task = TaskContext(instruction='Which?', response=Choice(['cat', 'dog']), fitting_stimuli=FITTING)
    session = stimulus_session(STIMULI, ['behavior:choice', 'behavior:probabilities'], task,
                               streaming=delivery == 'stream')
    model.participate(session)
    assert session.collect('behavior:choice').sizes['presentation'] == len(STIMULI)
    assert [c for c in calls if c[0] == 'reply'] == [('reply', s) for s in STIMULI['stimulus_id']]
    assert (len(chosen), len(tested)) == (1, 1)


@pytest.mark.parametrize('task, outputs, match', [
    (TaskContext(instruction='Which?', response=Choice(['cat', 'dog']), fitting_stimuli=FITTING,
                 present=('text:no_such_column',)), ['behavior:choice', 'behavior:probabilities'], 'no_such_column'),
    ('audio region', ['behavior:choice', 'neural:A1'], "reads 'audio' input, but the readout presents 'text'"),
])
def test_a_known_table_of_a_stream_is_checked_before_any_set_up(task, outputs, match):
    """Core keeps the table of a streamed delivery, so its prompts and regions are checked first too."""
    from brainscore.selection import LayerSelector
    calls = []
    if task == 'audio region':
        def extract(stimuli, layers, **kwargs):
            calls.append(('extract', tuple(layers)))
            return _features(stimuli, layers, **kwargs)
        model = BrainScoreModel('two-inputs', extractor={'text': extract, 'audio': extract},
                                region_layer_map={'A1': LayerSelector('late', modality='audio')},
                                behavioral_readout_layer='late', capability_config={
                                    'behavior-sessions': {'behavior:choice': ('readout', 'argmax')}})
        task = TaskContext(instruction='Which?', response=Choice(['cat', 'dog']), fitting_stimuli=FITTING)
    else:
        model = _reply_model(calls)
    with pytest.raises((ExecutionNotFeasibleError, ValueError), match=match):
        model.participate(stimulus_session(STIMULI, outputs, task, streaming=True))
    assert calls == []


def test_a_direct_answer_call_with_history_and_only_an_answer_part_is_refused():
    """With no session plan, respond keeps the refusal it had before the plan."""
    model = BrainScoreModel('answers', in_channels={'text'}, memory='trial_history',
                            answer=lambda stimulus_row, instruction, label_set: label_set[0])
    model.start_task(TaskContext(instruction='Which?', response=Choice(['cat', 'dog'])))
    with pytest.raises(ValueError, match="memory='trial_history' needs reply="):
        model._behavioral.respond(STIMULI, ['behavior:choice'])


def test_trial_history_without_a_part_that_sees_it_is_refused_before_the_readout_fit():
    calls = []

    def answer(stimulus_row, instruction, label_set):
        calls.append(('answer',))
        return label_set[0]

    def extract(stimuli, layers, **kwargs):
        calls.append(('extract', tuple(layers)))
        return _features(stimuli, layers, **kwargs)
    model = BrainScoreModel('answers', extractor={'text': extract}, in_channels={'text'},
                            behavioral_readout_layer='late', answer=answer, memory='trial_history',
                            capability_config={'behavior-sessions': {
                                'behavior': 'generation', 'behavior:probabilities': 'readout'}})
    task = TaskContext(instruction='Which?', response=Choice(['cat', 'dog']), fitting_stimuli=FITTING)
    with pytest.raises(ExecutionNotFeasibleError, match="memory='trial_history' needs reply="):
        model.participate(stimulus_session(STIMULI, ['behavior:choice', 'behavior:probabilities'], task))
    assert calls == []          # before the plan: the readout was fitted first


def test_a_region_the_readout_cannot_record_is_refused_before_the_readout_fit():
    from brainscore.selection import LayerSelector
    calls = []

    def extract(stimuli, layers, **kwargs):
        calls.append(('extract', tuple(layers)))
        return _features(stimuli, layers, **kwargs)
    model = BrainScoreModel('two-inputs', extractor={'text': extract, 'audio': extract},
                            region_layer_map={'A1': LayerSelector('late', modality='audio')},
                            behavioral_readout_layer='late',
                            capability_config={'behavior-sessions': {'behavior:choice': ('readout', 'argmax')}})
    task = TaskContext(instruction='Which?', response=Choice(['cat', 'dog']), fitting_stimuli=FITTING)
    with pytest.raises(ExecutionNotFeasibleError, match="reads 'audio' input, but the readout presents 'text'"):
        model.participate(stimulus_session(STIMULI, ['behavior:choice', 'neural:A1'], task))
    assert calls == []          # before the plan: the readout was fitted first


def test_a_bad_prompt_layout_is_refused_before_the_readout_fit():
    calls = []
    model = _reply_model(calls)
    task = TaskContext(instruction='Which?', response=Choice(['cat', 'dog']), fitting_stimuli=FITTING,
                       present=('text:no_such_column',))
    with pytest.raises(ValueError, match='no_such_column'):
        model.participate(stimulus_session(STIMULI, ['behavior:choice', 'behavior:probabilities'], task))
    assert calls == []
