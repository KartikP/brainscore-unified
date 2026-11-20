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
    the same seed, give the first score."""
    import brainscore
    bench = brainscore.load_benchmark('GridGame-reach-5x5')
    subject = policy_model(random_action_policy(0), 'random')
    scores = [float(bench(subject)) for _ in range(2)]
    scores.append(float(bench(policy_model(random_action_policy(0), 'random-again'))))
    assert scores == [scores[0]] * 3


def test_the_registered_floor_is_the_mean_rate_of_the_random_policy_over_seeds_0_to_29():
    """The floor of the owner's decision (2026-11-19): the random policy with seeds 0 to 29,
    each on the 15 registered boards, gives 167 successes in 450 episodes (measured
    2026-11-19). One seed alone gives from 0.2 to 0.6."""
    import brainscore
    score = brainscore.load_benchmark('GridGame-reach-5x5')(policy_model(random_action_policy(0), 'r'))
    assert score.attrs['random_floor'] == pytest.approx(167 / 450)


def test_a_seeded_null_gives_an_episode_the_same_actions_after_any_other_episode():
    from brainscore.harnesses.grid_game import GridGameEnv, play_game

    def episode(seed):
        return GridGameEnv(size=5, seed=seed, max_steps=20)

    def actions(subject, seed):
        return play_game(subject, episode(seed))['actions']

    alone = actions(policy_model(random_action_policy(0), 'b-alone'), 501)
    subject = policy_model(random_action_policy(0), 'a-then-b')
    actions(subject, 500)
    a_then_b = actions(subject, 501)
    subject = policy_model(random_action_policy(0), 'b-then-a')
    b_then_a = actions(subject, 501)
    actions(subject, 500)
    subject.reset()
    after_reset = actions(subject, 501)
    assert alone == a_then_b == b_then_a == after_reset
    assert actions(policy_model(random_action_policy(1), 'other-seed'), 501) != alone


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


def test_the_calibration_runs_at_first_use_and_stays_out_of_the_run_record(monkeypatch):
    """The oracle and the random floor run when the ceiling is first read, not when the
    benchmark is built nor for an incompatible subject; the provenance of a scored subject
    names only that subject."""
    import brainscore
    from brainscore_core.streaming import Session
    sessions = []
    start = Session.__init__

    def counted(self, *args, **kwargs):
        sessions.append(True)
        start(self, *args, **kwargs)
    monkeypatch.setattr(Session, '__init__', counted)
    built = GridGameBenchmark(size=5, n_episodes=2)
    assert not sessions                                       # nothing measured at build
    assert np.isnan(float(brainscore.score(brainscore.load_model('chance-baseline'), built)))
    assert not sessions                                       # N/A before any episode
    score = brainscore.score(policy_model(random_action_policy(3), 'probe'), built)
    assert sessions
    assert [entry['subject'] for entry in score.attrs['provenance']] == ['probe']
    assert float(score.attrs['ceiling']) == 1.0


# --- an invalid move (rule 4.12, decided 2026-11-19) -------------------------------

def _oracle_with_invalid_moves(invalid, seen):
    """The oracle, except that ``invalid(call, step)`` steps give no move (an unreadable reply)."""
    state = {'call': 0, 'step': 0}

    def policy(row):
        seen.append(row['observation']['ascii'])
        call, step = state['call'], state['step']
        state['call'] += 1
        state['step'] += 1
        return {'motor': None} if invalid(call, step) else greedy_oracle_policy(row)
    policy.reset = lambda: state.__setitem__('step', 0)
    return policy


def test_one_invalid_move_keeps_the_agent_in_place_and_the_run_goes_on(bench):
    seen = []
    score = bench(policy_model(_oracle_with_invalid_moves(lambda call, step: call == 0, seen), 'one'))
    assert seen[1] == seen[0]                       # the same board: the agent stayed for that step
    assert score.attrs['success_rate'] == 1.0       # every episode ran and was solved
    assert score.attrs['n_invalid'] == 1 and score.attrs['n_trials'] == len(seen)
    assert 'invalid' in score.attrs['answer_policy']
    assert score.attrs['result_validity'] == 'valid'


def test_above_ten_percent_invalid_steps_the_result_is_invalid_and_returned(bench):
    score = bench(policy_model(_oracle_with_invalid_moves(lambda call, step: step < 3, []), 'many'))
    assert score.attrs['n_invalid'] == 3 * bench.n_episodes
    assert score.attrs['result_validity'] == 'invalid'
    assert np.isfinite(float(score))
