"""Unit tests for the embodied grid-game benchmark.

Pure CPU, no model weights: the subjects are the oracle / random / a fixed
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


def test_a_seeded_null_gives_one_score_each_time():
    """The registered settings: one subject scored twice, and a new subject with
    the same seed, give the first score; the first score is the measured floor."""
    import brainscore
    bench = brainscore.load_benchmark('GridGame-reach-5x5')
    subject = policy_model(random_action_policy(0), 'random')
    scores = [float(bench(subject)) for _ in range(2)]
    scores.append(float(bench(policy_model(random_action_policy(0), 'random-again'))))
    assert scores == [scores[0]] * 3
    assert scores[0] * float(bench.ceiling) == bench._random_floor == 0.2


def test_a_seeded_null_still_draws_new_actions_across_steps_and_episodes():
    from brainscore.harnesses.grid_game import GridGameEnv, play_game
    subject = policy_model(random_action_policy(0), 'random')
    actions = [play_game(subject, GridGameEnv(size=5, seed=500 + i, max_steps=20))['actions']
               for i in range(3)]
    assert all(len(set(episode)) > 1 for episode in actions)
    assert len({tuple(episode[:5]) for episode in actions}) == 3


def test_registered():
    import brainscore
    assert 'GridGame-reach-5x5' in brainscore.benchmark_registry
    bm = brainscore.load_benchmark('GridGame-reach-5x5')
    assert isinstance(bm, GridGameBenchmark)


from brainscore_core.contract import Subject
from brainscore_core.streaming import RESPOND, StreamEvent


class _Mover(Subject):
    """A subject with only ``participate``: it answers every step with action 0."""

    identifier = 'participate-only-mover'
    in_channels = {'observation', 'instruction', 'reward', 'control'}
    out_channels = {'motor'}

    def participate(self, session):
        session.declare_inputs(self.in_channels)
        while (event := session.next_input()) is not None:
            if event.channel == 'control' and event.payload == RESPOND:
                session.respond(StreamEvent('motor', 0, event.t_ms, {}))


def test_declares_the_rows_it_sends_and_the_action_it_needs(bench):
    from tests.benchmark_declarations import assert_declares, assert_preflight
    assert_declares(bench, {'observation', 'instruction', 'reward'}, {'motor'})
    assert_preflight(bench)


def test_a_participate_only_subject_gets_a_score(bench):
    score = bench(_Mover())
    assert 0.0 <= score.attrs['raw'] <= 1.0
