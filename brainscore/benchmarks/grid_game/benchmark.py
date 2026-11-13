"""Closed-loop grid game as a registered Brain-Score benchmark.

Wraps the closed-loop ``GridGameEnv`` (``brainscore.harnesses.grid_game``) as a
proper ``Benchmark``: ``__call__(subject)`` runs N reach-the-goal episodes
through ``run_environment(subject, ...)`` (the subject's ``participate``) and
returns the success rate,
ceiled by the greedy oracle, with the random-action floor in ``attrs``.

This is the closed-loop capability as a *registered benchmark*, not just a script —
the same closed loop that drives the scaling-curve demo. The environment IS the
data source (a deterministic simulator seeded per episode), so there is no
dataset to download. The subject must declare an ``action_fn`` (it emits
the ``motor`` channel); a feature/behavioral model with no ``action_fn`` is
correctly incompatible.
"""
import numpy as np

from brainscore_core.benchmarks import BenchmarkBase
from brainscore_core.metrics import Score
from brainscore import BrainScoreModel
from brainscore.harnesses.grid_game import (
    GridGameEnv, play_game, greedy_oracle_policy, random_action_policy,
)
from brainscore.model_helpers.policy_wrapper import PolicyWrapper


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
    """Wrap a bare ``(observation, history) -> action`` policy as a scorable
    subject (the same shape a real model takes: ``action_fn`` on a
    ``BrainScoreModel``). Used for the oracle ceiling + random floor."""
    return BrainScoreModel(
        identifier=identifier,
        policy=PolicyWrapper(policy, max_history=4),
    )


class GridGameBenchmark(BenchmarkBase):
    """Closed-loop reach-the-goal grid game, scored through a closed-loop session.

    Metric: success rate over ``n_episodes`` (each a fresh seeded board),
    ceiling-normalized by the greedy oracle. ``attrs`` carry the raw rate, the
    random-action floor, mean path efficiency, and an ``above_floor`` flag.

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

        common = dict(n_episodes=n_episodes, size=size,
                      max_steps=max_steps, base_seed=base_seed)
        oracle_rate, _ = _run(policy_model(greedy_oracle_policy, 'oracle'), **common)
        self._random_floor, _ = _run(
            policy_model(random_action_policy(0), 'random'), **common)

        super().__init__(
            identifier=f'GridGame-reach-{size}x{size}',
            version=1, parent='closed_loop',
            ceiling=Score(max(oracle_rate, 1e-6)), bibtex='')

    def __call__(self, subject) -> Score:
        rate, eff = _run(subject, n_episodes=self.n_episodes, size=self.size,
                         max_steps=self.max_steps, base_seed=self.base_seed)
        ceiling = float(self.ceiling)
        score = Score(rate / ceiling)
        score.attrs['raw'] = Score(rate)
        score.attrs['ceiling'] = ceiling
        score.attrs['success_rate'] = rate
        score.attrs['mean_efficiency'] = eff
        score.attrs['random_floor'] = self._random_floor
        score.attrs['n_episodes'] = self.n_episodes
        score.attrs['above_floor'] = bool(rate > self._random_floor)
        return score
