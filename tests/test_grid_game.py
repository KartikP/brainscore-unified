"""Tests for the closed-loop grid video game harness.

Validates (a) the environment mechanics, (b) that a model drives the loop end
to end through ``process(EnvironmentStep)``, (c) that the greedy oracle solves
wall-free boards optimally, and (d) — the null requirement — that a random
action policy is clearly worse than the oracle. (d) is what proves the task is
non-trivial and the closed loop is actually doing something.
"""
import numpy as np
import pytest

from brainscore_core.model_interface import BrainScoreModel
from brainscore.model_helpers.policy_wrapper import PolicyWrapper
from brainscore.harnesses.grid_game import (
    GridGameEnv, ACTIONS, greedy_oracle_policy, random_action_policy,
    play_game, evaluate_policy,
)


def _embodied_model(policy):
    """A model that is *only* a policy — no preprocessors, no activations."""
    return BrainScoreModel(
        identifier='grid-player', model=None, region_layer_map={},
        preprocessors={}, activations_model=None,
        action_fn=PolicyWrapper(policy, max_history=4),
    )


class TestEnvMechanics:
    def test_reset_places_agent_and_goal_distinct(self):
        env = GridGameEnv(size=5, seed=1)
        obs = env.reset()
        assert env.agent_pos != env.goal_pos
        assert obs['frame'].shape == (5 * 32, 5 * 32, 3)
        assert obs['frame'].dtype == np.uint8

    def test_step_moves_and_clamps_to_bounds(self):
        env = GridGameEnv(size=4, seed=0)
        env.agent_pos = (0, 0)
        # moving up from the top row is a no-op (clamped)
        _, _, _, info = env.step(0)
        assert env.agent_pos == (0, 0)
        assert info['moved'] is False

    def test_reaching_goal_gives_reward_and_done(self):
        env = GridGameEnv(size=4, seed=0)
        env.agent_pos = (1, 0)
        env.goal_pos = (0, 0)
        obs, reward, done, info = env.step(0)  # up -> onto goal
        assert reward == 1.0 and done and info['reached_goal']

    def test_walls_block_movement(self):
        env = GridGameEnv(size=4, seed=0, n_walls=0)
        env.agent_pos = (1, 1)
        env.walls = {(0, 1)}
        env.step(0)  # try to move up into a wall
        assert env.agent_pos == (1, 1)

    def test_ascii_board_has_player_and_goal(self):
        env = GridGameEnv(size=4, seed=0)
        env.agent_pos = (0, 0)
        env.goal_pos = (3, 3)
        board = env.ascii_board()
        assert 'P' in board and 'G' in board
        assert board.count('P') == 1 and board.count('G') == 1
        assert len(board.splitlines()) == 4
        # the observation exposes it for thinking text policies
        assert env._observe()['ascii'] == board


class TestClosedLoop:
    def test_oracle_solves_wall_free_optimally(self):
        env = GridGameEnv(size=6, seed=7, n_walls=0)
        model = _embodied_model(greedy_oracle_policy)
        res = play_game(model, env)
        assert res['solved']
        # wall-free: oracle takes exactly the Manhattan distance
        assert res['steps'] == res['optimal_steps']
        assert res['efficiency'] == 1.0

    def test_process_dispatches_to_action_fn(self):
        # The loop must go through model.process(EnvironmentStep), not a side door.
        env = GridGameEnv(size=5, seed=3)
        calls = {'n': 0}

        def counting_policy(obs, history):
            calls['n'] += 1
            return greedy_oracle_policy(obs, history)

        model = _embodied_model(counting_policy)
        res = play_game(model, env)
        assert calls['n'] == res['steps']     # one action_fn call per tick
        assert res['solved']

    def test_actions_are_legal_indices(self):
        env = GridGameEnv(size=6, seed=2)
        model = _embodied_model(greedy_oracle_policy)
        res = play_game(model, env)
        assert all(a in ACTIONS for a in res['actions'])


class TestNullSeparation:
    def test_oracle_beats_random_null(self):
        def make(policy):
            return _embodied_model(policy)

        oracle = evaluate_policy(make, greedy_oracle_policy, n_episodes=25, size=6)
        rand = evaluate_policy(make, random_action_policy(seed=0),
                               n_episodes=25, size=6)
        # Oracle solves every wall-free board; random rarely does within the cap.
        assert oracle['success_rate'] == 1.0
        assert rand['success_rate'] < oracle['success_rate']
        # And when both solve, the oracle is far more efficient.
        assert oracle['mean_efficiency'] > rand['mean_efficiency']

    def test_random_null_is_above_zero_but_weak(self):
        # Sanity: a random walker occasionally stumbles onto the goal, so the
        # floor is small-positive, not exactly zero — exactly why we measure it.
        def make(policy):
            return _embodied_model(policy)
        rand = evaluate_policy(make, random_action_policy(seed=1),
                               n_episodes=40, size=5)
        assert 0.0 <= rand['success_rate'] < 0.8
