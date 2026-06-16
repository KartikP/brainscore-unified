"""Tests are the spec. A capability closure must satisfy its slot's contract and
dispatch through process(). Example shown for action_fn; mirror for generation_fn /
state_change_fn. Adapt freely.
"""
import numpy as np

from brainscore_core.model_interface import (
    BrainScoreModel, EnvironmentStep, EnvironmentResponse)


def _stub_action_fn():
    def act(env_step) -> EnvironmentResponse:
        return EnvironmentResponse(action=0)   # TODO: real policy
    return act


def test_action_fn_dispatches_through_process():
    model = BrainScoreModel(identifier='cap-demo', model=None,
                            region_layer_map={}, preprocessors={},
                            action_fn=_stub_action_fn())
    step = EnvironmentStep(observation={'frame': np.zeros((4, 4, 3), 'uint8'),
                                        'instruction': 'go', 'legal_actions': {0: 'up'}},
                           step_num=0)
    response = model.process(step)
    assert isinstance(response, EnvironmentResponse)
    assert int(np.asarray(response.action).reshape(-1)[0]) == 0


def test_missing_capability_fails_clearly():
    import pytest
    model = BrainScoreModel(identifier='no-cap', model=None,
                            region_layer_map={}, preprocessors={})   # no action_fn
    step = EnvironmentStep(observation={}, step_num=0)
    with pytest.raises(NotImplementedError):
        model.process(step)
