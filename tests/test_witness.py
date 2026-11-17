"""Tests for the Witness: it watches a subject's sessions through ``observed``.

Covers: (1) it records every step of a closed-loop episode (the embodied grid
game), (2) it restores ``participate`` on exit, (3) it works for a subject with
``participate`` only, (4) the trace is JSON-serializable (numpy frames
stripped), and (5) it sees an intervention session.
"""
import json
import numpy as np
import pandas as pd
import pytest

from brainscore import BrainScoreModel
from brainscore_core import Subject
from brainscore_core.contract import TaskContext
from brainscore_core.streaming import StreamEvent
from brainscore_core.streaming_helpers import (behavioral_response, drain_session_inputs,
                                               neural_response, session_stimuli)
from brainscore_core.supported_data_standards.brainio.assemblies import BehavioralAssembly, NeuroidAssembly
from brainscore_core.supported_data_standards.brainio.stimuli import StimulusSet
from brainscore.harnesses.grid_game import GridGameEnv, play_game, greedy_oracle_policy
from brainscore.witness import Witness


def _oracle_model():
    return BrainScoreModel('grid-oracle', policy=greedy_oracle_policy)


def test_witness_records_environment_steps():
    model = _oracle_model()
    env = GridGameEnv(size=5, seed=504, max_steps=20)
    with Witness(model, label='oracle') as w:
        res = play_game(model, env, max_steps=20)
    assert res['solved'] and res['steps'] == res['optimal_steps']
    assert len(w.events) == res['steps']
    for ev in w.events:
        assert ev.event_type == 'environment_step'
        assert ev.mode == 'action'
        assert ev.did['kind'] == 'action'
        assert isinstance(ev.did['action'], list)
        assert ev.saw and 'frame_shape' in ev.saw[0]


def test_witness_restores_participate_on_exit():
    model = _oracle_model()
    with Witness(model):
        assert 'participate' in vars(model)            # watched
    assert 'participate' not in vars(model)            # the class method again
    env = GridGameEnv(size=5, seed=1, max_steps=10)
    res = play_game(model, env, max_steps=10)
    assert 'solved' in res


def test_witness_trace_is_json_serializable():
    model = _oracle_model()
    env = GridGameEnv(size=5, seed=504, max_steps=20)
    with Witness(model) as w:
        play_game(model, env, max_steps=20)
    blob = json.dumps({'summary': w.summary(), 'events': w.trace()}, default=str)
    reloaded = json.loads(blob)
    assert reloaded['summary']['n_calls'] == len(w.events)
    # numpy frame must have been stripped from the serialized 'saw'
    for ev in reloaded['events']:
        for s in ev['saw']:
            assert '_frame' not in s


class _ParticipateOnly(Subject):
    """A subject with participate only: no process, no BrainScoreModel internals."""
    identifier = 'participate-only'
    in_channels = frozenset({'text'})
    out_channels = frozenset({'behavior'})

    def participate(self, session):
        stimuli = session_stimuli(session, drain_session_inputs(session))
        session.respond(StreamEvent('behavior', BehavioralAssembly(
            ['yes'] * len(stimuli), coords={'stimulus_id': ('presentation', list(stimuli['stimulus_id']))},
            dims=['presentation']), t_ms=None))

    def reset(self):
        pass


def _text_stimuli(*sentences):
    stimuli = StimulusSet({'stimulus_id': [f's{i}' for i in range(len(sentences))],
                           'sentence': list(sentences)})
    stimuli.identifier = 'witness-text'
    return stimuli


@pytest.mark.unit
def test_witness_sees_a_participate_only_subject():
    model = _ParticipateOnly()
    task = TaskContext(task_type='probabilities', label_set=['yes', 'no'], instruction='Is it a word?')
    with Witness(model, label='bare') as w:
        behavioral_response(model, _text_stimuli('hello world', 'lexical test'), task)
    assert len(w.events) == 1
    ev = w.events[0]
    assert (ev.event_type, ev.mode, ev.modalities, ev.n_inputs) == ('stimuli', 'behavior', ['text'], 2)
    assert ev.instruction == 'Is it a word?'
    assert [view['text']['text'] for view in ev.saw] == ['hello world', 'lexical test']
    assert ev.did['kind'] == 'behavior' and ev.did['prediction_sample'] == ['yes', 'yes']


def test_witness_records_error_and_reraises():
    class _Boom(_ParticipateOnly):
        def participate(self, session):
            drain_session_inputs(session)
            raise ValueError('boom')
    model = _Boom()
    with Witness(model) as w:
        with pytest.raises(ValueError):
            behavioral_response(model, _text_stimuli('x'), TaskContext(task_type='probabilities',
                                                                       label_set=['a', 'b']))
    assert len(w.events) == 1 and w.events[0].error and 'boom' in w.events[0].error


@pytest.mark.unit
def test_witness_on_real_brainscoremodel_preserves_output():
    class _TextExtractor:
        identifier = 'witness-text-extractor'

        def __init__(self):
            self.calls = 0

        def __call__(self, stimuli, layers):
            self.calls += 1
            return NeuroidAssembly(np.ones((len(stimuli), 2)), dims=['presentation', 'neuroid'], coords={
                'stimulus_id': ('presentation', list(stimuli['stimulus_id'])),
                'layer': ('neuroid', [layers[0]] * 2), 'neuroid_id': ('neuroid', [0, 1])})

    extractor = _TextExtractor()
    model = BrainScoreModel('witness-text', extractor={'text': extractor},
                            region_layer_map={'language': 'L'})
    with Witness(model) as w:
        out = neural_response(model, _text_stimuli('one', 'two'), ['neural:language'])
    assert extractor.calls == 1             # Witness adds no extraction call
    ev = w.events[0]
    assert (ev.mode, ev.recording_target, ev.did['kind'], ev.did['shape']) == \
        ('neural', ['language'], 'neural', list(out.shape))
    assert [s['text']['text'] for s in ev.saw] == ['one', 'two']


@pytest.mark.unit
def test_witness_sees_an_intervention():
    from brainscore_core.events import Perturbation, PerturbationApplied, Selection, StateChange
    from brainscore_core.streaming_helpers import apply_state_change
    model = BrainScoreModel('witness-intervention', intervention=lambda change: (
        PerturbationApplied('h', change.target, change.perturbation), lambda: None))
    with Witness(model) as w:
        apply_state_change(model, StateChange('ablation', Selection('L', [0]), Perturbation('zero')))
    (ev,) = w.events
    assert (ev.event_type, ev.mode) == ('intervention', 'intervention')
    assert ev.saw[0]['kind'] == 'ablation'
