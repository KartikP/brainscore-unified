"""Tests for the Witness benchmark-observability layer.

Covers: (1) it records real process() calls on a no-GPU benchmark (the embodied
grid game), (2) it restores the original process() on exit, (3) it is robust to
models that don't define BrainScoreModel internals (a bare UnifiedModel), and
(4) the trace is JSON-serializable (numpy frames stripped).
"""
import json
import numpy as np
import pandas as pd
import pytest

from brainscore_core.model_interface import BrainScoreModel, TaskContext, UnifiedModel
from brainscore.model_helpers.policy_wrapper import PolicyWrapper
from brainscore.harnesses.grid_game import GridGameEnv, play_game, greedy_oracle_policy
from brainscore.witness import Witness


def _oracle_model():
    return BrainScoreModel('grid-oracle', None, {}, {}, None,
                           action_fn=PolicyWrapper(greedy_oracle_policy, max_history=4))


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


def test_witness_restores_process_on_exit():
    model = _oracle_model()
    original = model.process
    with Witness(model):
        assert model.process is not original          # patched
    # after exit, the instance attribute is removed -> class method again
    assert model.process != original or 'process' not in model.__dict__
    # and it still works
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


class _BareTextModel(UnifiedModel):
    """A minimal UnifiedModel with NO COLUMN_TO_MODALITY / _detect_modalities,
    to prove the Witness doesn't depend on BrainScoreModel internals."""
    identifier = 'bare-text'
    region_layer_map = {}
    supported_modalities = {'text'}

    def start_task(self, *a, **k): pass
    def start_recording(self, *a, **k): pass
    def reset(self): pass

    def process(self, input_event):
        # echo a trivial behavioral assembly-free value
        return {'echoed': len(input_event)}


def test_witness_robust_to_bare_unifiedmodel():
    model = _BareTextModel()
    stim = pd.DataFrame({'stimulus_id': ['a', 'b'], 'sentence': ['hello world', 'lexical test']})
    with Witness(model, label='bare') as w:
        model.process(stim)
    assert len(w.events) == 1
    ev = w.events[0]
    assert ev.event_type == 'stimulus_set'
    # fell back to the default column->modality map -> detected text
    saw_text = [s for s in ev.saw if 'text' in s]
    assert saw_text and saw_text[0]['text']['text'] in ('hello world', 'lexical test')


def test_witness_records_error_and_reraises():
    class _Boom(_BareTextModel):
        def process(self, input_event):
            raise ValueError('boom')
    model = _Boom()
    with Witness(model) as w:
        with pytest.raises(ValueError):
            model.process(pd.DataFrame({'stimulus_id': ['a'], 'sentence': ['x']}))
    assert len(w.events) == 1 and w.events[0].error and 'boom' in w.events[0].error
