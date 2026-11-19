"""Closed-loop grid game as a registered Brain-Score benchmark.

Wraps the closed-loop ``GridGameEnv`` (``brainscore.harnesses.grid_game``) as a
proper ``Benchmark``: ``__call__(subject)`` runs N reach-the-goal episodes
through ``run_environment(subject, ...)`` (the subject's ``participate``) and
returns the success rate,
ceiled by the greedy oracle, with the random-action floor in ``attrs``.

This is the closed-loop capability as a *registered benchmark*, not just a script —
the same closed loop that drives the scaling-curve demo. The environment IS the
data source (a deterministic simulator seeded per episode), so there is no
dataset to download. The subject must have a ``policy`` part (it emits
the ``motor`` channel); a feature/behavioral model with no ``policy`` part is
correctly incompatible.
"""
import contextvars
from functools import cached_property

import numpy as np

from brainscore_core.benchmarks import BenchmarkBase
from brainscore_core.metrics import Score
from brainscore import BrainScoreModel
from brainscore.harnesses.grid_game import (
    GridGameEnv, play_game, greedy_oracle_policy, random_action_policy,
)
from brainscore.benchmarks._score import benchmark_score


# The random floor is the mean success rate of the random policy over these seeds
# (ADR-0038): one seed varies from 0.2 to 0.6 on the default boards.
FLOOR_SEEDS = range(30)


def _run(subject, n_episodes, size, max_steps, base_seed):
    """Run ``n_episodes`` seeded episodes; return (success_rate, mean_efficiency)."""
    solves, effs = [], []
    for i in range(n_episodes):
        env = GridGameEnv(size=size, seed=base_seed + i, max_steps=max_steps)
        res = play_game(subject, env)
        solves.append(1.0 if res['solved'] else 0.0)
        if res['solved']:
            effs.append(res.get('efficiency', 0.0))
    return float(np.mean(solves)), (float(np.mean(effs)) if effs else 0.0)


def policy_model(policy, identifier):
    """A ``row -> {'motor': action}`` policy as a scorable subject (the same
    shape a real model takes: a ``policy`` part on a ``BrainScoreModel``). Used
    for the oracle ceiling + random floor; both read ``observation``."""
    return BrainScoreModel(identifier=identifier, policy=policy)


class GridGameBenchmark(BenchmarkBase):
    """Closed-loop reach-the-goal grid game, scored through a closed-loop session.

    Metric: success rate over ``n_episodes`` (each a fresh seeded board),
    ceiling-normalized by the greedy oracle. ``attrs`` carry the raw rate, the
    random-action floor (the mean rate of the random policy over ``FLOOR_SEEDS``),
    mean path efficiency, and an ``above_floor`` flag.

    :param size: grid side length. :param n_episodes: episodes to average.
    :param max_steps: per-episode step budget. :param base_seed: episode seeds
        are ``base_seed + i`` so every subject sees the identical boards.
    """

    # Each episode is one trial that starts with an empty history.
    trial_memory = 'none'

    def __init__(self, size: int = 5, n_episodes: int = 15,
                 max_steps: int = 20, base_seed: int = 500):
        self.size = size
        self.n_episodes = n_episodes
        self.max_steps = max_steps
        self.base_seed = base_seed
        # Every row holds the view, the goal and the last reward; the game needs actions,
        # so a subject without a policy is N/A.
        self.required_input_channels = frozenset({'observation', 'instruction', 'reward'})
        self.requested_output_channels = frozenset({'motor'})

        super().__init__(
            identifier=f'GridGame-reach-{size}x{size}',
            version=1, parent='closed_loop', ceiling=None, bibtex='')

    @cached_property
    def _calibration(self):
        """The oracle rate and the random floor, measured at first use, not when the
        benchmark is built: a subject that is N/A waits for no episode. The episodes run
        in an empty context, so they are not noted in the run record of a scored subject."""
        common = dict(n_episodes=self.n_episodes, size=self.size,
                      max_steps=self.max_steps, base_seed=self.base_seed)

        def measure():
            oracle_rate, _ = _run(policy_model(greedy_oracle_policy, 'oracle'), **common)
            floor = float(np.mean([
                _run(policy_model(random_action_policy(seed), f'random-{seed}'), **common)[0]
                for seed in FLOOR_SEEDS]))
            return oracle_rate, floor
        return contextvars.Context().run(measure)

    @cached_property
    def ceiling(self) -> Score:
        return Score(max(self._calibration[0], 1e-6))

    @property
    def _random_floor(self) -> float:
        return self._calibration[1]

    def __call__(self, subject) -> Score:
        rate, eff = _run(subject, n_episodes=self.n_episodes, size=self.size,
                         max_steps=self.max_steps, base_seed=self.base_seed)
        score = benchmark_score(self, raw=rate, metric='success_rate')
        score.attrs['success_rate'] = rate
        score.attrs['mean_efficiency'] = eff
        score.attrs['random_floor'] = self._random_floor
        score.attrs['n_episodes'] = self.n_episodes
        score.attrs['above_floor'] = bool(rate > self._random_floor)
        return score
