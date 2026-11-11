"""Compare a closed-loop subject's choices with human choices in the same states.

Task success says whether a model can play; it does not say whether it plays
like a person. This module replays recorded human states to the subject one at a
time and scores how often its move matches what humans did there, normalized by
how often two humans agree with each other.

Data format. An episode is a list of ``(step, human_counts)`` pairs:

- ``step`` is the row (a mapping of channel -> payload) a participant faced
  (history-relevant order, so a stateful policy sees the same sequence a
  person did).
- ``human_counts[a]`` is how many participants chose action ``a`` in that state.

Steps answered by fewer than two participants are still shown (they keep the
subject's history faithful) but are not scored, because human agreement, the
ceiling, needs at least two answers.

Replay is open-loop: the subject's move never changes the next state, so every
subject sees exactly the states humans saw. That is what makes the comparison
fair; it also means replay cannot measure task success. Pair it with a
closed-loop benchmark (for example the grid game) for that.

Metric (per scored state, then averaged over states):

- model agreement  = counts[model_action] / n
  (probability a random participant made the model's choice)
- human ceiling    = sum_a counts[a] * (counts[a] - 1) / (n * (n - 1))
  (probability two different participants agree)
- chance floor     = 1 / n_actions

The normalized score is model agreement / human ceiling. A model that always
picks the most common human choice can exceed 1; the score is not clipped, so
that case stays visible.
"""
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np

from brainscore_core.benchmarks import BenchmarkBase
from brainscore_core.environment import DiscreteSpec
from brainscore_core.metrics import Score
from brainscore_core.streaming_helpers import run_environment

HumanEpisode = Sequence[Tuple[Mapping[str, Any], Sequence[int]]]


class ReplayEnvironment:
    """Serve recorded states in order, ignoring the subject's actions.

    After the last recorded state is answered, a terminal copy of it ends the
    episode so every recorded state receives exactly one action.
    """

    def __init__(self, steps: Sequence[Mapping[str, Any]], n_actions: int):
        if not steps:
            raise ValueError('A replay episode needs at least one step')
        self._steps = [dict(step) for step in steps]
        self._spec = DiscreteSpec(n_actions)
        self._index = 0

    def action_spec(self):
        return self._spec

    def reset(self):
        self._index = 0
        return self._steps[0]

    def step(self, outputs):
        del outputs  # ponytail: open-loop replay; the recorded next state is fixed.
        self._index += 1
        if self._index < len(self._steps):
            return self._steps[self._index], None
        return self._steps[-1], 'terminal'



def replay_actions(subject, steps: Sequence[Mapping[str, Any]], n_actions: int) -> List[int]:
    """Replay ``steps`` through ``participate``, return its choices.

    The episode's ``begin`` clears the subject's episode state; its interventions
    and task setup stay.
    """
    responses = run_environment(subject, ReplayEnvironment(steps, n_actions))
    actions = [int(np.asarray(action).item()) for action in responses]
    if len(actions) != len(steps):
        raise RuntimeError(f'Expected {len(steps)} actions, got {len(actions)}')
    return actions


def human_agreement(model_actions: Sequence[int],
                    human_counts: Sequence[Sequence[int]]) -> Dict[str, float]:
    """Score model choices against human choice counts (see module docstring)."""
    if len(model_actions) != len(human_counts):
        raise ValueError('One model action is needed per human state')
    model, human, chance = [], [], []
    for action, counts in zip(model_actions, human_counts):
        counts = np.asarray(counts, dtype=float)
        n = counts.sum()
        if n < 2:
            continue
        model.append(counts[action] / n)
        human.append((counts * (counts - 1)).sum() / (n * (n - 1)))
        chance.append(1.0 / len(counts))
    if not model:
        raise ValueError('No state has answers from two or more participants')
    return {'raw': float(np.mean(model)), 'ceiling': float(np.mean(human)),
            'chance': float(np.mean(chance)), 'n_scored': len(model)}


def _check_counts(counts, n_actions):
    counts = np.asarray(counts)
    if counts.shape != (n_actions,) or counts.dtype.kind not in 'iu' or (counts < 0).any():
        raise ValueError(f'human_counts must be {n_actions} non-negative integers')


class HumanAgreementBenchmark(BenchmarkBase):
    """Human-choice agreement on replayed states, ceiled by human-human agreement.

    :param identifier: benchmark name, e.g. ``'MyStudy2026-gridgame-human_agreement'``.
    :param episodes: list of episodes in the module's data format.
    :param n_actions: size of the discrete action space.
    """

    # Asks for one action per state. Each episode is one trial that starts empty.
    requested_output_channels = frozenset({'motor'})
    trial_protocol = 'independent'

    def __init__(self, identifier: str, episodes: Sequence[HumanEpisode],
                 n_actions: int, bibtex: str = None):
        if not episodes:
            raise ValueError('Provide at least one episode')
        for episode in episodes:
            for _, counts in episode:
                _check_counts(counts, n_actions)
        self.episodes = episodes
        self.n_actions = n_actions
        # The recorded states are the inputs: one channel for each key a step holds.
        self.required_input_channels = frozenset(
            channel for episode in episodes for step, _ in episode for channel in step)
        counts = [c for episode in episodes for _, c in episode]
        # The ceiling depends only on the human data; any valid action works here.
        self._summary = human_agreement([0] * len(counts), counts)
        super().__init__(identifier=identifier, version=1, parent='closed_loop',
                         ceiling=Score(self._summary['ceiling']), bibtex=bibtex)

    def __call__(self, candidate) -> Score:
        actions, counts = [], []
        for episode in self.episodes:
            actions += replay_actions(candidate, [s for s, _ in episode], self.n_actions)
            counts += [c for _, c in episode]
        result = human_agreement(actions, counts)
        score = Score(result['raw'] / result['ceiling'])
        score.attrs['raw'] = Score(result['raw'])
        score.attrs['ceiling'] = result['ceiling']
        score.attrs['chance_floor'] = result['chance']
        score.attrs['n_scored_states'] = result['n_scored']
        score.attrs['above_floor'] = bool(result['raw'] > result['chance'])
        score.attrs['evaluation'] = 'open_loop_human_replay'
        return score
