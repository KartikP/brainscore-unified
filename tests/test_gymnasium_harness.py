"""Tests for the Gymnasium harness — drive a standard env through process(EnvironmentStep).

Uses a random policy (no GPU, no model weights) so it runs anywhere gymnasium+minigrid
are installed. Skips cleanly if they're absent.
"""
import numpy as np
import pytest

pytest.importorskip("gymnasium")
pytest.importorskip("minigrid")

from brainscore_core.model_interface import BrainScoreModel
from brainscore.model_helpers.policy_wrapper import PolicyWrapper
from brainscore.harnesses.gymnasium_harness import play_gym_episode, random_gym_policy, MINIGRID_ACTIONS


def _random_model():
    return BrainScoreModel('gym-random', None, {}, {}, None,
                           action_fn=PolicyWrapper(random_gym_policy(0), max_history=0))


def test_play_gym_episode_runs_and_reports():
    res = play_gym_episode(_random_model(), 'MiniGrid-DoorKey-6x6-v0', max_steps=20, seed=3)
    assert res['env_id'] == 'MiniGrid-DoorKey-6x6-v0'
    assert res['mission'] and 'key' in res['mission'].lower()
    assert isinstance(res['solved'], bool)
    assert 1 <= res['steps'] <= 20
    assert all(0 <= a < len(MINIGRID_ACTIONS) for a in res['actions'])


def test_episode_drives_process_environment_step(monkeypatch):
    """Confirm the loop actually calls model.process with EnvironmentStep carrying a frame."""
    seen = {}
    model = _random_model()
    orig = model.process

    def spy(ev):
        seen.setdefault('events', 0)
        seen['events'] += 1
        assert type(ev).__name__ == 'EnvironmentStep'
        assert 'frame' in ev.observation and np.asarray(ev.observation['frame']).ndim == 3
        return orig(ev)

    model.process = spy
    res = play_gym_episode(model, 'MiniGrid-Empty-5x5-v0', max_steps=15, seed=0)
    assert seen['events'] == res['steps']
