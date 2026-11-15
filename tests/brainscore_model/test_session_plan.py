"""The session plan: built once when a session starts, read by the session parts (step 10).

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
