"""A closed loop is one Session over an EnvironmentSource.

The subject reads rows between ``respond`` controls, calls the policy and answers
on the output channels it returns. Neural recording during the loop comes from the
same forward pass that produced the action.
"""
from copy import deepcopy

import numpy as np
import pytest
import torch
from torch import nn

from brainscore import BrainScoreModel
from brainscore_core import io_catalog
from brainscore_core.streaming import FixedSource, Session
from brainscore_core.streaming_helpers import EnvironmentSource, run_environment

pytestmark = pytest.mark.unit


@pytest.fixture
def speech_channel(monkeypatch):
    """An extension output family: a policy that answers in words, not on 'motor'."""
    monkeypatch.setattr(io_catalog, '_CATALOG', dict(io_catalog._CATALOG))
    io_catalog.register_channel(io_catalog.CatalogEntry(
        'speech', io_catalog.OUTPUT, 'a policy answer', 'a str the policy said',
        'the test environment', payload_type=str))


class CountingEnvironment:
    """Deterministic: row n carries n; the episode ends after ``length`` actions."""

    def __init__(self, length=5):
        self.length, self.n, self.actions = length, 0, []

    def reset(self):
        self.n = 0
        return {'observation': {'x': [0.0, 1.0, 2.0]}}

    def step(self, outputs):
        (action,) = outputs.values()
        self.actions.append(action)
        self.n += 1
        return ({'observation': {'x': [float(self.n), 1.0, 2.0]}},
                'terminal' if self.n == self.length else None)


def test_replay_from_a_fixed_source_reproduces_the_live_responses():
    model = BrainScoreModel('counting', policy=lambda row: {'motor': int(sum(row['observation']['x']))})
    live = Session(EnvironmentSource(CountingEnvironment()), ('motor',))
    recorded = []
    with live.observe(on_input=recorded.append):
        model.participate(live)
    live_responses = [event.payload for event in live.emitted]
    assert len(live_responses) == 5

    replay = Session(FixedSource(deepcopy(recorded)), ('motor',))
    model.reset()
    model.participate(replay)
    assert [event.payload for event in replay.emitted] == live_responses


def _network():
    torch.manual_seed(0)
    return nn.Sequential(nn.Linear(3, 4), nn.ReLU(), nn.Linear(4, 2)).eval()


class _Extractor:
    """Declares the 'vision' input; the closed loop never calls it."""
    identifier = 'unused'

    def __call__(self, stimuli, layers):
        raise AssertionError('the closed loop records through hooks')


def _recording_model(net, forward_calls):
    net.register_forward_pre_hook(lambda module, args: forward_calls.append(1))

    def policy(row):
        with torch.no_grad():
            scores = net(torch.tensor(row['observation']['x']))
        return {'motor': int(scores.argmax())}

    return BrainScoreModel('recording-policy', model=net, region_layer_map={'V4': '0'},
                           extractor={'vision': _Extractor()}, policy=policy)


def test_one_session_yields_motor_and_neural_from_one_forward_pass():
    net, forward_calls = _network(), []
    model = _recording_model(net, forward_calls)
    session = Session(EnvironmentSource(CountingEnvironment(4)), ('motor', 'neural:V4'))
    model.participate(session)

    assert len(forward_calls) == 4          # exactly one forward pass per step
    motor = session.collect('motor')
    neural = session.collect('neural:V4')
    assert len(motor) == 4 and neural.sizes['presentation'] == 4
    expected = np.stack([net[0](torch.tensor([float(n), 1.0, 2.0])).detach().numpy() for n in range(4)])
    np.testing.assert_allclose(np.asarray(neural.values).reshape(4, -1), expected, atol=1e-6)


def test_a_policy_without_an_extractor_is_not_applicable_to_neural_outputs():
    model = BrainScoreModel('no-extractor', policy=lambda row: {'motor': 0})
    with pytest.raises(NotImplementedError, match='No session handler'):
        model.participate(Session(EnvironmentSource(CountingEnvironment()), ('motor', 'neural:V4')))


def test_a_policy_may_answer_on_a_non_motor_channel_with_no_new_class(speech_channel):
    model = BrainScoreModel('talker', policy=lambda row: {'speech': f"x={row['observation']['x'][0]}"},
                            policy_out_channels=('speech',))
    env = CountingEnvironment(2)
    session = Session(EnvironmentSource(env, action_channel='speech'), ('speech',))
    model.participate(session)
    assert env.actions == ['x=0.0', 'x=1.0']
    assert [event.payload for event in session.collect('speech')] == env.actions


def test_run_environment_helper_takes_the_outputs(speech_channel):
    model = BrainScoreModel('talker', policy=lambda row: {'speech': 'hi'}, policy_out_channels=('speech',))
    assert run_environment(model, CountingEnvironment(2), action_channel='speech') == ['hi', 'hi']


def test_a_stateful_policy_is_cleared_when_an_episode_begins():
    from brainscore.model_helpers.policy_wrapper import PolicyWrapper
    model = BrainScoreModel('stateful', policy=PolicyWrapper(lambda obs, history: len(history)))
    first = run_environment(model, CountingEnvironment(3))
    second = run_environment(model, CountingEnvironment(3))   # no model.reset() in between
    assert first == second == [0, 1, 2]
