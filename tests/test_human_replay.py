import numpy as np
import pytest

from brainscore import BrainScoreModel
from brainscore_core.events import EnvironmentStep
from brainscore.harnesses.human_replay import (
    HumanAgreementBenchmark, human_agreement, replay_actions)
from brainscore.model_helpers.policy_wrapper import PolicyWrapper

pytestmark = pytest.mark.unit


def subject(policy):
    return BrainScoreModel('s', None, {}, {}, None,
                           policy=PolicyWrapper(policy, max_history=8))


def steps(n):
    return [EnvironmentStep(observation={'state': i}) for i in range(n)]


def test_metric_matches_hand_computation():
    # State 1: 3 of 4 chose action 0. State 2: one answer, so it is not scored.
    result = human_agreement([0, 1], [[3, 1], [0, 1]])
    assert result['raw'] == pytest.approx(0.75)
    assert result['ceiling'] == pytest.approx((3 * 2) / (4 * 3))
    assert result['chance'] == pytest.approx(0.5)
    assert result['n_scored'] == 1


def test_no_scorable_state_is_an_error():
    with pytest.raises(ValueError, match='two or more'):
        human_agreement([0], [[1, 0]])


def test_replay_shows_every_state_in_order_and_ignores_actions():
    seen = []

    def policy(observation, history):
        seen.append((observation['state'], len(history)))
        return 1

    assert replay_actions(subject(policy), steps(3), n_actions=2) == [1, 1, 1]
    assert seen == [(0, 0), (1, 1), (2, 2)]  # Same sequence and history a person had.


def test_benchmark_separates_human_like_from_random_and_contrary():
    # Humans mostly pick action (state % 3); the ceiling comes from their spread.
    rng = np.random.RandomState(0)
    episodes = []
    for _ in range(4):
        episode = []
        for step in steps(10):
            counts = rng.multinomial(19, [0.1, 0.1, 0.1]).tolist()
            counts[step.observation['state'] % 3] += 11
            episode.append((step, counts))
        episodes.append(episode)
    benchmark = HumanAgreementBenchmark('synthetic-human_agreement', episodes, n_actions=3)

    human_like = benchmark(subject(lambda obs, history: obs['state'] % 3))
    contrary = benchmark(subject(lambda obs, history: (obs['state'] + 1) % 3))

    assert human_like.attrs['above_floor'] and not contrary.attrs['above_floor']
    assert float(human_like) > 1.0 > float(contrary)  # Modal choice beats human-human.
    assert human_like.attrs['n_scored_states'] == 40
    assert benchmark.parent == 'closed_loop'


def test_counts_must_match_action_space():
    with pytest.raises(ValueError, match='non-negative integers'):
        HumanAgreementBenchmark('bad', [[(steps(1)[0], [1, 1, 1])]], n_actions=2)
