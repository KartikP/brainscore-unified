"""Tests for the Gymnasium harness — drive a standard env through a closed-loop session.

Uses a random policy (no GPU, no model weights) so it runs anywhere gymnasium+minigrid
are installed. The required CI profile rejects skips when either is absent.
"""
import numpy as np
import pytest

pytest.importorskip("gymnasium")
pytest.importorskip("minigrid")

from brainscore import BrainScoreModel
from brainscore.model_helpers.policy_wrapper import PolicyWrapper
from brainscore.harnesses.gymnasium_harness import play_gym_episode, random_gym_policy, MINIGRID_ACTIONS


def _random_model():
    return BrainScoreModel('gym-random',
                           policy=PolicyWrapper(random_gym_policy(0), max_history=0))


def test_play_gym_episode_runs_and_reports():
    res = play_gym_episode(_random_model(), 'MiniGrid-DoorKey-6x6-v0', max_steps=20, seed=3)
    assert res['env_id'] == 'MiniGrid-DoorKey-6x6-v0'
    assert res['mission'] and 'key' in res['mission'].lower()
    assert isinstance(res['solved'], bool)
    assert 1 <= res['steps'] <= 20
    assert all(0 <= a < len(MINIGRID_ACTIONS) for a in res['actions'])


def test_each_step_reaches_the_policy_as_a_row_with_a_frame():
    """The loop gives the policy one row per step, with the rendered frame."""
    rows = []
    act = PolicyWrapper(random_gym_policy(0), max_history=0)

    def policy(row):
        rows.append(row)
        return act(row)

    res = play_gym_episode(BrainScoreModel('gym-spy', policy=policy), 'MiniGrid-Empty-5x5-v0',
                           max_steps=15, seed=0)
    assert len(rows) == res['steps']
    assert all(np.asarray(row['observation']['frame']).ndim == 3 for row in rows)
