"""Capability registry tests for BrainScoreModel."""

import importlib
import sys
import textwrap

from brainscore.capabilities import capability_registry
from brainscore import BrainScoreModel
from brainscore_core.model_interface import Perturbation, PerturbationApplied, Selection, StateChange, TaskContext


class StubActivationsModel:
    def __init__(self, return_value='activations_output'):
        self._return_value = return_value
        self.call_args = None

    def __call__(self, stimuli, layers=None, **kwargs):
        del kwargs
        self.call_args = {
            'stimuli': stimuli,
            'layers': layers,
        }
        return self._return_value


class StubStimulusSet:
    def __init__(self, columns):
        self.columns = columns


def test_neural_encoding_capability_is_registered_dispatch_path():
    act_model = StubActivationsModel(return_value='vision_assembly')
    model = BrainScoreModel(
        identifier='resnet',
        model='torch_model',
        region_layer_map={'IT': 'layer4'},
        extractor={'vision': act_model},
    )
    model.start_recording('IT')
    stimuli = StubStimulusSet(columns=['image_file_name', 'stimulus_id'])

    result = model.process(stimuli)

    assert 'neural-encoding' in capability_registry
    assert result == 'vision_assembly'
    assert act_model.call_args['stimuli'] is stimuli
    assert act_model.call_args['layers'] == ['layer4']
    assert 'neural-encoding' in model._capability_state


def test_no_event_capability_answers_behavior():
    """A behavioral answer comes only from a session (BehaviorSessions); process()
    after a configured task still reaches the neural extraction."""
    act_model = StubActivationsModel(return_value='vision_assembly')
    model = BrainScoreModel(
        identifier='behavioral',
        model='torch_model',
        region_layer_map={'IT': 'layer4'},
        extractor={'vision': act_model},
    )
    model._behavioral.task = TaskContext(task_type='probabilities')
    model._behavioral.response_path = 'generation'
    model._behavioral.readout_classifier = object()
    model.start_recording('IT')
    stimuli = StubStimulusSet(columns=['image_file_name', 'stimulus_id'])

    assert model.process(stimuli) == 'vision_assembly'
    assert not {'behavioral-generation', 'behavioral-readout'} & set(capability_registry)


def test_closed_loop_capability_dispatches_a_row():
    calls = []

    def policy(row):
        calls.append(row)
        return {'motor': [0.0, 1.0]}

    model = BrainScoreModel(
        identifier='embodied',
        policy=policy,
    )
    row = {'observation': {'n': 7}}

    result = model.process(row)

    assert 'closed-loop' in capability_registry
    assert result == {'motor': [0.0, 1.0]}
    assert calls == [row]
    assert 'closed-loop' in model._capability_state


def test_closed_loop_capability_dispatches_a_text_message():
    calls = []

    def policy(row):
        calls.append(row)
        return {'text': 'pong'}

    model = BrainScoreModel(
        identifier='agent',
        policy=policy,
    )

    result = model.process({'text': 'ping'})

    assert result == {'text': 'pong'}
    assert calls == [{'text': 'ping'}]


def test_state_change_capability_dispatches_and_tracks_cleanup():
    cleanup_calls = []

    def intervention(state_change):
        applied = PerturbationApplied(
            handle_id='capability-handle',
            target=state_change.target,
            perturbation=state_change.perturbation,
        )

        def cleanup():
            cleanup_calls.append(applied.handle_id)

        return applied, cleanup

    model = BrainScoreModel(
        identifier='perturbed',
        intervention=intervention,
    )
    apply_event = StateChange(
        kind='ablation',
        target=Selection(layer='layer4'),
        perturbation=Perturbation(kind='zero'),
    )

    result = model.process(apply_event)

    assert 'intervention' in capability_registry
    assert isinstance(result, PerturbationApplied)
    assert result.handle_id == 'capability-handle'
    assert 'capability-handle' in model._interventions.active_interventions
    assert 'intervention' in model._capability_state

    reset_result = model.process(
        StateChange(kind='reset', handle_id='capability-handle')
    )

    assert reset_result is None
    assert cleanup_calls == ['capability-handle']
    assert model._interventions.active_interventions == {}


def test_self_registering_capability_file_dispatches_without_host_edits(tmp_path):
    module_path = tmp_path / 'demo_capability_plugin.py'
    module_path.write_text(textwrap.dedent("""
        from brainscore.capabilities import Capability, register_capability


        class DemoCapability(Capability):
            identifier = 'test-demo-capability'
            order = 10

            def setup(self, model):
                return {'model_identifier': model.identifier}

            def handles(self, model, event, **kwargs):
                return type(event).__name__ == 'DemoCapabilityEvent'

            def process(self, model, event, **kwargs):
                return model.capability_state(self.identifier)['model_identifier']


        register_capability(DemoCapability())
    """), encoding='utf-8')

    sys.path.insert(0, str(tmp_path))
    try:
        importlib.import_module('demo_capability_plugin')

        class DemoCapabilityEvent:
            pass

        model = BrainScoreModel(
            identifier='demo-model',
        )

        assert model.process(DemoCapabilityEvent()) == 'demo-model'
    finally:
        capability_registry.pop('test-demo-capability', None)
        sys.path.remove(str(tmp_path))
        sys.modules.pop('demo_capability_plugin', None)


# --- one registry, one order: an added capability is tried before a built-in one ---

def _added(capability):
    from brainscore.capabilities import register_capability
    register_capability(capability)
    return capability


def test_an_added_session_capability_runs_a_session_a_built_in_one_also_claims():
    """Added capabilities default to order 1000; the built-in session ones are 10000."""
    from brainscore.capabilities import Capability
    from brainscore_core.streaming import Session, StreamEvent

    class Recorder(Capability):
        identifier = 'test-added-neural-session'

        def enabled_for(self, subject):
            return subject.identifier == 'added-session'

        def supports_session(self, subject, channels):
            return set(channels) == {'neural:IT'}

        def participate(self, subject, session):
            while session.next_input() is not None:
                pass
            session.respond(StreamEvent('neural:IT', 'from-added', 0.0))

    _added(Recorder())
    try:
        model = BrainScoreModel(identifier='added-session', region_layer_map={'IT': 'layer4'},
                                extractor={'vision': StubActivationsModel()})
        assert model.check_session_support({'neural:IT'}).identifier == 'test-added-neural-session'
        other = BrainScoreModel(identifier='other', region_layer_map={'IT': 'layer4'},
                                extractor={'vision': StubActivationsModel()})
        assert other.check_session_support({'neural:IT'}).identifier == 'neural-sessions'
    finally:
        capability_registry.pop('test-added-neural-session', None)


def test_an_added_event_capability_before_a_built_in_one_answers_first():
    """An added event capability with a lower order answers an event a built-in one handles."""
    from brainscore.capabilities import Capability
    calls = []

    class Early(Capability):
        identifier = 'test-added-early-state-change'
        order = 10

        def handles(self, subject, event, **kwargs):
            return isinstance(event, StateChange)

        def process(self, subject, event, **kwargs):
            calls.append(event)
            return 'added'

    _added(Early())
    try:
        model = BrainScoreModel(identifier='early',
                                intervention=lambda change: (_ for _ in ()).throw(AssertionError))
        change = StateChange(kind='ablation', target=Selection(layer='layer4'),
                             perturbation=Perturbation(kind='zero'))
        assert model.process(change) == 'added'
        assert calls == [change]
    finally:
        capability_registry.pop('test-added-early-state-change', None)


def test_built_in_capabilities_each_take_one_role():
    """An event capability claims no session; a session capability handles no event."""
    model = BrainScoreModel(identifier='roles', region_layer_map={'IT': 'layer4'},
                            extractor={'vision': StubActivationsModel()},
                            policy=lambda row: {'motor': 0},
                            intervention=lambda change: (None, lambda: None))
    events = [StubStimulusSet(['image_path']), {'observation': 1},
              StateChange(kind='reset', handle_id='h')]
    for capability in capability_registry.values():
        if not capability.enabled_for(model):
            continue
        claims_session = any(capability.supports_session(model, frozenset(channels))
                             for channels in ({'neural:IT'}, {'motor'}, {'perturbation'}))
        handles_event = any(capability.handles(model, event) for event in events)
        assert not (claims_session and handles_event), capability.identifier
        assert claims_session or handles_event, capability.identifier
