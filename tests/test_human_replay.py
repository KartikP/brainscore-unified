import numpy as np
import pytest

from brainscore import BrainScoreModel
from brainscore.harnesses.human_replay import (
    HumanAgreementBenchmark, human_agreement, replay_actions)

pytestmark = pytest.mark.unit


def subject(policy):
    return BrainScoreModel('s', policy=policy)


def steps(n):
    return [{'observation': {'state': i}} for i in range(n)]


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

    class Policy:
        """Remembers the states it saw in this episode."""
        def __init__(self):
            self.history = []

        def reset(self):
            self.history.clear()

        def __call__(self, row):
            seen.append((row['observation']['state'], len(self.history)))
            self.history.append(row['observation'])
            return {'motor': 1}

    assert replay_actions(subject(Policy()), steps(3), n_actions=2) == [1, 1, 1]
    assert seen == [(0, 0), (1, 1), (2, 2)]  # Same sequence and history a person had.


def test_benchmark_separates_human_like_from_random_and_contrary():
    # Humans mostly pick action (state % 3); the ceiling comes from their spread.
    rng = np.random.RandomState(0)
    episodes = []
    for _ in range(4):
        episode = []
        for step in steps(10):
            counts = rng.multinomial(19, [0.1, 0.1, 0.1]).tolist()
            counts[step['observation']['state'] % 3] += 11
            episode.append((step, counts))
        episodes.append(episode)
    benchmark = HumanAgreementBenchmark('synthetic-human_agreement', episodes, n_actions=3)

    human_like = benchmark(subject(lambda row: {'motor': row['observation']['state'] % 3}))
    contrary = benchmark(subject(lambda row: {'motor': (row['observation']['state'] + 1) % 3}))

    assert human_like.attrs['above_floor'] and not contrary.attrs['above_floor']
    assert float(human_like) > 1.0 > float(contrary)  # Modal choice beats human-human.
    assert human_like.attrs['n_scored_states'] == 40
    assert benchmark.parent == 'closed_loop'


def test_counts_must_match_action_space():
    with pytest.raises(ValueError, match='non-negative integers'):
        HumanAgreementBenchmark('bad', [[(steps(1)[0], [1, 1, 1])]], n_actions=2)


def test_declares_the_recorded_steps_in_and_the_action_out():
    from tests.benchmark_declarations import assert_declares, assert_preflight
    episode = [(step, [3, 1]) for step in steps(3)]
    benchmark = HumanAgreementBenchmark('declared', [episode], n_actions=2)
    assert_declares(benchmark, {'observation'}, {'motor'})
    assert_preflight(benchmark)


def test_a_participate_only_subject_gets_a_score():
    from brainscore_core.contract import Subject
    from brainscore_core.streaming import RESPOND, StreamEvent

    class Mover(Subject):
        identifier = 'participate-only-mover'
        in_channels = {'observation', 'control'}
        out_channels = {'motor'}

        def participate(self, session):
            session.declare_inputs(self.in_channels)
            while (event := session.next_input()) is not None:
                if event.channel == 'control' and event.payload == RESPOND:
                    session.respond(StreamEvent('motor', 0, event.t_ms, {}))

    episode = [(step, [3, 1]) for step in steps(4)]
    score = HumanAgreementBenchmark('mover', [episode], n_actions=2)(Mover())
    assert score.attrs['n_scored_states'] == 4
