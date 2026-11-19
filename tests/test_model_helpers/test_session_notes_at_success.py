"""A session's notes reach the score only when the session succeeded, through the public score path.

A benchmark that catches a failed attempt and then runs a good session gets a score whose
public fields (``response``, ``delivery``, ``regions_per_pass``, ``provenance``) describe only
the good session. The registration of a wrapper (``ProvenanceSubject``) and the participants
of a conversation follow the same rule.
"""
import numpy as np
import pytest

import brainscore
from brainscore import BrainScoreModel
from brainscore.benchmarks import benchmark_score, not_estimated
from brainscore.harnesses.multi_agent import DialogueMediator, EchoAgent, multi_agent_rollout
from brainscore.model_helpers.provenance import ProvenanceSubject
from brainscore_core.benchmarks import BenchmarkBase
from brainscore_core.compatibility import ExecutionNotFeasibleError
from brainscore_core.contract import Subject, TaskContext
from brainscore_core.responses import Choice
from brainscore_core.streaming import StreamEvent
from brainscore_core.streaming_helpers import stimulus_session
from tests.brainscore_model.test_behavioral_readout import _make_image_stimulus_set

pytestmark = pytest.mark.integration

TASK = TaskContext(response=Choice(['cat', 'dog']), instruction='What animal is this?')


def _stimuli():
    return _make_image_stimulus_set(['cat', 'dog', 'cat'], identifier='notes-at-success')


class _Answers:
    """An answer part; ``fail_on`` is the call that raises (1 is the second trial)."""

    def __init__(self, fail_on=None):
        self.calls, self.fail_on = 0, fail_on

    def __call__(self, stimulus_row, instruction, label_set):
        self.calls += 1
        if self.calls - 1 == self.fail_on:
            raise RuntimeError('the answer part failed on this trial')
        return 'cat'


def _model(answer):
    return BrainScoreModel(identifier='notes-model', model=None, region_layer_map={},
                           extractor={'vision': lambda *a, **kw: None}, answer=answer)


class _Retry(BenchmarkBase):
    """Runs ``attempt`` (expected to raise ``refusal``), if any, then one good listed session."""

    def __init__(self, attempt=None, refusal=None):
        self.attempt, self.refusal = attempt, refusal
        self.required_input_channels = frozenset({'vision:image'})
        self.requested_output_channels = frozenset({'behavior:choice'})
        super().__init__(identifier='notes-retry', version=1, parent='behavior', ceiling=not_estimated())

    def __call__(self, subject):
        if self.attempt is not None:
            with pytest.raises(self.refusal):       # a benchmark that recovers from a failed attempt
                self.attempt(subject)
        session = stimulus_session(_stimuli(), ['behavior:choice'], TASK)
        ProvenanceSubject(subject, {'attempt': 'good'}).participate(session)
        return benchmark_score(self, raw=float(len(session.emitted)), metric='count')


def _score(attempt, refusal, answer=None):
    return brainscore.score(_model(answer or _Answers()), _Retry(attempt, refusal), check_mem=False)


def _registrations(score):
    return [entry['registration'] for entry in score.attrs['provenance'] if 'registration' in entry]


def _only_the_good_session(score):
    assert score.attrs['delivery'] == 'listed'
    assert _registrations(score) == [{'attempt': 'good'}]
    assert [record['method'] for record in score.attrs['response']] == ['generation']


def test_a_session_the_plan_refuses_before_any_response_leaves_nothing():
    def refused(subject):           # the model maps no region: its plan refuses neural:IT
        session = stimulus_session(_stimuli(), ['neural:IT'], TaskContext(task_type='passive'),
                                   streaming=True)
        ProvenanceSubject(subject, {'attempt': 'refused'}).participate(session)
    _only_the_good_session(_score(refused, ExecutionNotFeasibleError))


def test_a_session_that_fails_after_a_response_leaves_nothing():
    answer = _Answers(fail_on=1)

    def late(subject):
        session = stimulus_session(_stimuli(), ['behavior:choice'], TASK, streaming=True)
        try:
            ProvenanceSubject(subject, {'attempt': 'late'}).participate(session)
        finally:
            assert len(session.emitted) == 1     # the first trial was answered and accepted
            answer.fail_on = None
    _only_the_good_session(_score(late, RuntimeError, answer))


def test_a_wrapper_around_a_wrapper_that_fails_after_its_inner_session_leaves_nothing():
    class _FailsAfter(ProvenanceSubject):
        def participate(self, session):
            super().participate(session)
            raise RuntimeError('the outer wrapper failed after the inner session returned')

    def late(subject):
        session = stimulus_session(_stimuli(), ['behavior:choice'], TASK, streaming=True)
        inner = ProvenanceSubject(subject, {'attempt': 'inner'})
        _FailsAfter(ProvenanceSubject(inner, {'attempt': 'middle'}), {'attempt': 'outer'}).participate(session)
    _only_the_good_session(_score(late, RuntimeError))


def test_a_good_session_through_a_wrapper_states_the_registration_and_the_model_once():
    score = brainscore.score(ProvenanceSubject(_model(_Answers()), {'dtype': 'bf16'}),
                             _Retry(), check_mem=False)
    assert score.attrs['delivery'] == 'listed'
    assert score.attrs['regions_per_pass'] == 'all'
    assert sorted(map(str, _registrations(score))) == sorted(map(str, [{'dtype': 'bf16'}, {'attempt': 'good'}]))
    stated = [entry for entry in score.attrs['provenance'] if 'parts' in entry]
    assert [entry['subject'] for entry in stated] == ['notes-model']
    assert stated[0]['parts'] != 'not stated'


class _Silent(Subject):
    """Answers each turn and states no parts; ``fails`` refuses its first turn."""
    in_channels = {'text', 'control'}
    out_channels = {'behavior:text'}

    def __init__(self, identifier, fails=False):
        self._identifier, self.fails = identifier, fails

    @property
    def identifier(self):
        return self._identifier

    def participate(self, session):
        while (event := session.next_input()) is not None:
            if event.channel == 'control' and event.payload == 'respond':
                if self.fails:
                    raise RuntimeError('this participant cannot answer')
                session.respond(StreamEvent('behavior:text', 'hello', event.t_ms, {}))


class _Conversation(BenchmarkBase):
    """A failed conversation (caught), then one with the scored subject alone."""

    def __init__(self):
        self.required_input_channels = frozenset({'text'})
        self.requested_output_channels = frozenset({'behavior:text'})
        super().__init__(identifier='notes-conversation', version=1, parent='behavior',
                         ceiling=not_estimated())

    def __call__(self, subject):
        with pytest.raises(RuntimeError, match='cannot answer'):
            multi_agent_rollout([EchoAgent('partner-of-a-failed-run'), _Silent('refuser', fails=True)],
                                DialogueMediator(), n_turns=1)
        multi_agent_rollout([subject, EchoAgent('partner')], DialogueMediator(), n_turns=1)
        return benchmark_score(self, raw=1.0, metric='count')


def test_a_failed_conversation_names_none_of_its_participants():
    score = brainscore.score(_Silent('scored-silent'), _Conversation(), check_mem=False)
    participants = sorted(entry['subject'] for entry in score.attrs['provenance'])
    assert participants == ['partner', 'scored-silent']


class _NoTurns(BenchmarkBase):
    """One conversation of the scored subject alone, with no turn."""

    def __init__(self):
        self.required_input_channels = frozenset({'text'})
        self.requested_output_channels = frozenset({'behavior:text'})
        super().__init__(identifier='notes-no-turns', version=1, parent='behavior', ceiling=not_estimated())

    def __call__(self, subject):
        multi_agent_rollout([subject], DialogueMediator(), n_turns=0)
        return benchmark_score(self, raw=1.0, metric='count')


def test_a_conversation_with_no_turn_states_how_its_sessions_ran(monkeypatch):
    """A conversation that succeeds states its delivery and the store state of its
    sessions also when no turn was answered (as on the release branch)."""
    monkeypatch.setenv('RESULTCACHING_DISABLE', '1')
    score = brainscore.score(EchoAgent('seat'), _NoTurns(), check_mem=False)
    assert score.attrs['delivery'] == 'stream'
    seat = [entry for entry in score.attrs['provenance'] if entry['subject'] == 'seat']
    assert seat and all(entry['activation_store'] == 'off' for entry in seat)


class _Inner(BenchmarkBase):
    """Scores the subject it was given; its sessions name that subject."""

    def __init__(self):
        self.required_input_channels = frozenset({'text'})
        self.requested_output_channels = frozenset({'behavior:text'})
        super().__init__(identifier='notes-inner', version=1, parent='behavior', ceiling=not_estimated())

    def __call__(self, subject):
        multi_agent_rollout([subject], DialogueMediator(), n_turns=1)
        return benchmark_score(self, raw=1.0, metric='count')


class _Outer(_Inner):
    """Runs a nested score() of another subject; no session of its own."""

    def __call__(self, subject):
        brainscore.score(_Silent('loaded-inner'), _Inner(), check_mem=False)
        return benchmark_score(self, raw=1.0, metric='count')


def test_a_nested_score_names_the_inner_subject_it_ran():
    score = brainscore.score(_Silent('outer-subject'), _Outer(), check_mem=False)
    assert [entry['subject'] for entry in score.attrs['provenance']] == ['loaded-inner']
