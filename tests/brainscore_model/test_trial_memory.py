"""Memory and trial independence: ``begin`` empties history, nothing else; protocols decide when it falls."""
import numpy as np
import pytest

from brainscore import BrainScoreModel, score
from brainscore.harnesses.human_replay import replay_actions
from brainscore.model_helpers.policy_wrapper import PolicyWrapper
from brainscore_core._evaluation.measures import record_measures
from brainscore_core.compatibility import CompatibilityError, check_channel_compatibility
from brainscore_core.events import Perturbation, PerturbationApplied, Selection, StateChange
from brainscore_core.metrics import Score
from brainscore_core.model_interface import TaskContext
from brainscore_core.responses import Text
from brainscore_core.streaming import Session, begin_event, _with_trial_memory
from brainscore_core.streaming_helpers import apply_state_change, stimulus_session
from brainscore_core.supported_data_standards.brainio.stimuli import StimulusSet

pytestmark = pytest.mark.unit

N = 5
STIMULI = StimulusSet({'stimulus_id': [f's{i}' for i in range(N)], 'sentence': list('abcde')})
STIMULI.identifier = 'trial-memory'


def _task(protocol):
    return TaskContext(instruction='Say anything.', response=Text(), trial_memory=protocol)


def _history_lengths(model, protocol, stimuli=STIMULI, session=None):
    session = session or stimulus_session(stimuli, ['behavior:text'], task=_task(protocol))
    model.participate(session)
    return list(session.collect('behavior:text').values)


def _model(memory):
    return BrainScoreModel('counts', supported_modalities={'text'}, memory=memory,
                           reply=lambda prompt: str(len(prompt.history)))


def test_trials_without_memory_start_empty_in_any_order():
    model = _model('trial_history')
    assert _history_lengths(model, 'none') == ['0'] * N
    shuffled = STIMULI.iloc[[3, 0, 4, 1, 2]]
    shuffled.identifier = 'shuffled'
    assert _history_lengths(model, 'none', shuffled) == ['0'] * N


def test_a_conversation_accumulates_and_the_next_episode_restarts():
    model = _model('trial_history')
    assert _history_lengths(model, 'history') == ['0', '1', '2', '3', '4']
    assert _history_lengths(model, 'history') == ['0', '1', '2', '3', '4']   # new session
    assert model._behavioral.history == []


def test_a_subject_without_the_trait_always_has_empty_history():
    model = _model(None)
    assert _history_lengths(model, 'none') == ['0'] * N
    assert _history_lengths(model, 'history') == ['0'] * N


class _ConversationBenchmark:
    identifier = 'conversation'
    required_input_channels = {'text'}
    requested_output_channels = {'behavior:text'}
    trial_memory = 'history'


def test_a_conversation_memory_requires_the_trait_and_nothing_else_is_declared():
    with pytest.raises(CompatibilityError, match='memory:trial_history'):
        check_channel_compatibility(_model(None), _ConversationBenchmark())
    check_channel_compatibility(_model('trial_history'), _ConversationBenchmark())
    with pytest.raises(ValueError, match='trial_memory'):
        TaskContext(task_type='passive', trial_memory='chat')


def test_the_events_carry_the_trial_memory():
    def marks(protocol):
        events = list(_with_trial_memory(_rows(), protocol))
        return [e.payload if e.channel == 'control' else 'row' for e in events]
    assert marks('none') == ['begin', 'row', 'end'] * 3
    assert marks('history') == ['begin'] + ['row'] * 3 + ['end']
    session = stimulus_session(STIMULI, task=_task('none')).declare_inputs({'control', 'text'})
    controls = [e.payload for e in iter(session.next_input, None) if e.channel == 'control']
    assert controls == ['begin', 'end'] * N


def _rows():
    from brainscore_core.streaming import StreamEvent
    return [StreamEvent('text', str(i), 0.0, {'stimulus_index': i, 'stimulus_id': f's{i}',
                                              'column': 'sentence'})
            for i in range(3)]


def test_begin_keeps_interventions_and_the_task_and_clears_episode_state():
    log = {'policy': 0, 'extractor': 0}

    class Policy:
        def __call__(self, row):
            return {'motor': np.zeros(1)}

        def reset(self):
            log['policy'] += 1

    class Extractor:
        identifier = 'stateful'

        def __call__(self, stimuli, layers, **kwargs):
            raise AssertionError('not called')

        def reset(self):
            log['extractor'] += 1

    state = {'lesion': False}

    def intervention(change):
        state['lesion'] = True

        def cleanup():
            state['lesion'] = False
        return PerturbationApplied(handle_id='h', target=change.target,
                                   perturbation=change.perturbation, applied_at=0), cleanup

    model = BrainScoreModel('s', extractor={'text': Extractor()}, policy=Policy(),
                            intervention=intervention, memory='trial_history')
    apply_state_change(model, StateChange(kind='ablation', target=Selection(layer='x'),
                                          perturbation=Perturbation(kind='zero')))
    task = TaskContext(task_type='passive')
    model.start_task(task)
    model.participate(Session([begin_event()], ('motor',)))             # what a begin does:
    assert log == {'policy': 1, 'extractor': 1}                        # parts clear their episode state
    model._behavioral.history = [('earlier', 'reply')]
    model._begin_episode()
    assert model._behavioral.history == []                             # h empty
    assert log == {'policy': 2, 'extractor': 2}
    assert state['lesion'] and 'h' in model._interventions.active_interventions      # w unchanged
    assert model._behavioral.task is task                      # s unchanged


def test_replay_keeps_a_lesion_and_clears_the_policy_between_episodes():
    class Policy:
        def __init__(self):
            self.steps_seen = []

        def __call__(self, observation, history):
            self.steps_seen.append(len(history))
            return 0

    inner = Policy()
    wrapper = PolicyWrapper(inner, max_history=8)
    state = {'lesion': False}

    def intervention(change):
        state['lesion'] = True

        def cleanup():
            state['lesion'] = False
        return PerturbationApplied(handle_id='h', target=change.target,
                                   perturbation=change.perturbation, applied_at=0), cleanup

    model = BrainScoreModel('s', policy=wrapper, intervention=intervention)
    apply_state_change(model, StateChange(kind='ablation', target=Selection(layer='x'),
                                          perturbation=Perturbation(kind='zero')))
    steps = [{'observation': {'state': i}} for i in range(3)]
    replay_actions(model, steps, n_actions=2)
    replay_actions(model, steps, n_actions=2)
    assert inner.steps_seen == [0, 1, 2, 0, 1, 2]      # the episode state restarted at each begin
    assert state['lesion'] and 'h' in model._interventions.active_interventions   # the lesion survived both


class _Benchmark:
    identifier = 'applies-a-lesion'
    version = 1
    required_input_channels = {'text'}
    requested_output_channels = {'perturbation'}

    def __init__(self, undo):
        self.undo = undo

    def __call__(self, subject):
        applied = apply_state_change(subject, StateChange(
            kind='ablation', target=Selection(layer='x'), perturbation=Perturbation(kind='zero')))
        if self.undo:
            apply_state_change(subject, StateChange(kind='reset', handle_id=applied.handle_id))
        return Score(1.0)


def _lesionable(handles):
    def intervention(change):
        handle = f'h{len(handles)}'
        handles.append(handle)
        return PerturbationApplied(handle_id=handle, target=change.target,
                                   perturbation=change.perturbation, applied_at=0), lambda: None
    return BrainScoreModel('lesionable', supported_modalities={'text'}, intervention=intervention)


def test_a_benchmark_that_leaves_its_lesion_fails_and_one_that_undoes_it_passes():
    result = score(_lesionable([]), _Benchmark(undo=True), check_mem=False)
    assert result.attrs['subject_traits'] == () and result.attrs['trial_memory'] == 'none'
    with pytest.raises(RuntimeError, match='left intervention handle'):
        score(_lesionable([]), _Benchmark(undo=False), check_mem=False)


def test_a_lesion_the_caller_applied_is_not_counted_against_the_benchmark():
    model = _lesionable([])
    apply_state_change(model, StateChange(kind='ablation', target=Selection(layer='x'),
                                          perturbation=Perturbation(kind='zero')))
    score(model, _Benchmark(undo=True), check_mem=False)
    assert 'h0' in model._interventions.active_interventions         # the caller's lesion stays


def test_the_record_tracks_handles_applied_and_undone_across_sessions():
    model = _lesionable([])
    with record_measures() as record:
        first = apply_state_change(model, StateChange(
            kind='ablation', target=Selection(layer='x'), perturbation=Perturbation(kind='zero')))
        apply_state_change(model, StateChange(kind='reset', handle_id=first.handle_id))
        apply_state_change(model, StateChange(
            kind='ablation', target=Selection(layer='x'), perturbation=Perturbation(kind='zero')))
    assert record.applied == {'h0', 'h1'} and record.unreleased == {'h1'}


def test_the_experiment_plan_records_traits_and_trial_memory(tmp_path):
    from brainscore.experiments import Experiment, RecordBehavior, TaskProtocol
    protocol = TaskProtocol(STIMULI, _task('history'))
    plan = Experiment(subject=_model('trial_history'), protocol=protocol,
                      tools=[RecordBehavior()], output_dir=tmp_path / 'run').validate()
    assert plan['subject_traits'] == ['memory:trial_history']
    assert plan['protocol']['config']['trial_memory'] == 'history'
    assert 'memory:trial_history' in plan['protocol']['config']['required_traits']
    from brainscore_core.compatibility import CompatibilityError
    # The protocol is checked with the rule score() uses.
    with pytest.raises(CompatibilityError, match='memory:trial_history'):
        Experiment(subject=_model(None), protocol=protocol, tools=[RecordBehavior()],
                   output_dir=tmp_path / 'run2').validate()
