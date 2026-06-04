"""Unit tests for the embodied grid-game benchmark.

Pure CPU, no model weights: the candidates are the oracle / random / a fixed
greedy policy wrapped as action_fn models — the same shape a real VLM policy
takes. Verifies ceiling, floor, the success-rate metric, and registration.
"""
import numpy as np
import pytest

from brainscore.harnesses.grid_game import (
    greedy_oracle_policy, random_action_policy)
from brainscore.benchmarks.grid_game.benchmark import (
    GridGameBenchmark, policy_model)


@pytest.fixture(scope='module')
def bench():
    return GridGameBenchmark(size=5, n_episodes=8, max_steps=20, base_seed=500)


def test_oracle_hits_ceiling(bench):
    score = bench(policy_model(greedy_oracle_policy, 'oracle'))
    assert score.attrs['success_rate'] == 1.0
    assert float(score) == pytest.approx(1.0)
    assert score.attrs['above_floor'] is True


def test_random_at_or_below_ceiling(bench):
    score = bench(policy_model(random_action_policy(7), 'rnd'))
    assert 0.0 <= score.attrs['raw'] <= score.attrs['ceiling']
    # the random floor the benchmark measured is a sane probability
    assert 0.0 <= score.attrs['random_floor'] <= 1.0


def test_score_attrs_present(bench):
    score = bench(policy_model(greedy_oracle_policy, 'oracle'))
    for k in ('raw', 'ceiling', 'success_rate', 'mean_efficiency',
              'random_floor', 'n_episodes', 'above_floor'):
        assert k in score.attrs
    assert score.attrs['n_episodes'] == 8


def test_same_boards_every_candidate(bench):
    # determinism: scoring the same policy twice gives the identical rate
    a = bench(policy_model(random_action_policy(3), 'a')).attrs['success_rate']
    b = bench(policy_model(random_action_policy(3), 'b')).attrs['success_rate']
    assert a == b


def test_registered():
    import brainscore
    assert 'GridGame-reach-5x5' in brainscore.benchmark_registry
    bm = brainscore.load_benchmark('GridGame-reach-5x5')
    assert isinstance(bm, GridGameBenchmark)
