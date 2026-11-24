"""Capability tests for BrainScoreModel: one selection per session, and
``process`` as the one-row call for a loop that a simulator owns."""

import pytest

from brainscore import BrainScoreModel
from brainscore_core.contract import TaskContext
from brainscore_core.responses import Choice
from brainscore_core.events import Perturbation, PerturbationApplied, Selection, StateChange
from brainscore_core.streaming_helpers import apply_state_change


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


def test_process_gives_a_row_to_the_policy():
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

    assert result == {'motor': [0.0, 1.0]}
    assert calls == [row]


@pytest.mark.parametrize('given', ['stimuli', 'change'])
def test_process_takes_only_a_row(given):
    """The one-row call has one job; stimuli and changes go through a session."""
    import pandas as pd
    model = BrainScoreModel(identifier='embodied', policy=lambda row: {'motor': [0.0]})
    value = (pd.DataFrame({'stimulus_id': ['s0']}) if given == 'stimuli'
             else StateChange(kind='ablation', target=Selection(layer='x'), perturbation=Perturbation(kind='zero')))
    with pytest.raises(TypeError, match='one row'):
        model.process(value)


def test_process_gives_a_text_message_to_the_policy():
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


def test_a_state_change_is_applied_and_undone_through_sessions():
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

    result = apply_state_change(model, apply_event)

    assert isinstance(result, PerturbationApplied)
    assert result.handle_id == 'capability-handle'

    reset_result = apply_state_change(model, StateChange(kind='reset', handle_id='capability-handle'))

    assert reset_result == 'capability-handle'   # the session acknowledges the undone handle
    assert cleanup_calls == ['capability-handle']
    with pytest.raises(KeyError, match='No active intervention'):   # nothing left to undo
        apply_state_change(model, StateChange(kind='reset', handle_id='capability-handle'))


# --- one selection: an added capability is tried before a built-in one ---

def test_an_added_session_capability_runs_a_session_a_built_in_one_also_claims():
    """A capability given to the model runs a session its built-in part also claims."""
    from brainscore.capabilities import Capability
    from brainscore_core.streaming import Session, StreamEvent

    class Recorder(Capability):
        identifier = 'test-added-neural-session'

        def supports_session(self, subject, channels):
            return set(channels) == {'neural:IT'}

        def participate(self, subject, session):
            while session.next_input() is not None:
                pass
            session.respond(StreamEvent('neural:IT', 'from-added', 0.0))

    model = BrainScoreModel(identifier='added-session', region_layer_map={'IT': 'layer4'},
                            extractor={'vision': StubActivationsModel()}, capabilities=[Recorder()])
    assert model.check_session_support({'neural:IT'}).identifier == 'test-added-neural-session'
    other = BrainScoreModel(identifier='other', region_layer_map={'IT': 'layer4'},
                            extractor={'vision': StubActivationsModel()})
    assert other.check_session_support({'neural:IT'}).identifier == 'neural-sessions'


def test_participate_selects_the_part_once_and_never_calls_process():
    """Each session kind runs on the part its session capability selected; the
    model's process() convenience is not on the path."""
    import numpy as np
    from brainscore_core.responses import Choice
    from brainscore_core.streaming import Session, StreamEvent
    from brainscore_core.streaming_helpers import EnvironmentSource, apply_state_change, stimulus_session
    from brainscore_core.supported_data_standards.brainio.assemblies import NeuroidAssembly
    from brainscore_core.supported_data_standards.brainio.stimuli import StimulusSet

    def extractor(stimuli, layers=None, **kwargs):
        ids = list(stimuli['stimulus_id'])
        return NeuroidAssembly(np.ones((len(ids), 1)), dims=['presentation', 'neuroid'], coords={
            'stimulus_id': ('presentation', ids), 'neuroid_id': ('neuroid', ['n']),
            'layer': ('neuroid', [layers[0]])})

    class Steps:
        def reset(self):
            return {'observation': 0}

        def step(self, outputs):
            return {'observation': 1}, 'terminal'

    class NoProcess(BrainScoreModel):
        def process(self, input_event, multi_modality=False):
            raise AssertionError('participate called process()')

    model = NoProcess('once', extractor={'vision': extractor}, region_layer_map={'IT': 'layer4'},
                      behavioral_readout_layer='layer4', policy=lambda row: {'motor': 0},
                      # The readout derives the choice; the statement declares it (rule 4.11).
                      answer_owners={'behavior:choice': ('readout', 'argmax')},
                      intervention=lambda change: (PerturbationApplied(
                          handle_id='h', target=change.target, perturbation=change.perturbation),
                          lambda: None))
    stimuli = StimulusSet({'stimulus_id': ['a', 'b'], 'image_path': ['a.png', 'b.png'],
                           'image_label': ['x', 'y']})
    stimuli.identifier = 'once'
    model.participate(stimulus_session(stimuli, ['neural:IT']))
    task = TaskContext(response=Choice(['x', 'y']), fitting_stimuli=stimuli)
    model.participate(stimulus_session(stimuli, ['behavior:choice', 'neural:IT'], task=task))
    model.participate(Session(EnvironmentSource(Steps()), ('motor',)))
    change = StateChange(kind='ablation', target=Selection(layer='layer4'),
                         perturbation=Perturbation(kind='zero'))
    assert apply_state_change(model, change).handle_id == 'h'
    assert model.check_session_support({'neural:IT'}).identifier == 'neural-sessions'
