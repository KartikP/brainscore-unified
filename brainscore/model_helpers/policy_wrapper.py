"""PolicyWrapper: per-episode memory for a closed-loop policy.

A benchmark runs a closed-loop session (``run_environment``): each tick the
subject receives a row (a mapping of channel -> payload), returns an action, and
the environment produces the next row, without ``reset()`` in between. A
stateless policy ignores prior steps; a stateful policy needs to carry context
across them.

``PolicyWrapper`` provides that state. It wraps a per-step policy callable,
maintains a rolling history of recent (observation, action) pairs, and exposes a
``policy``-compatible ``__call__`` so it plugs straight into
``BrainScoreModel(policy=...)``. The history is cleared by ``reset()``, which
``BrainScoreModel`` calls when an episode begins. The model contract is
unchanged: this is a harness around the policy, not a new method on the interface.
"""
from collections import deque
from typing import Any, Callable, Deque, Mapping, Tuple

from brainscore_core.streaming import StreamEvent


class PolicyWrapper:
    """Wrap a per-step policy into a stateful ``policy`` callable.

    :param policy: ``callable(observation, history) -> action`` where ``action``
        is either a raw action payload or a ``StreamEvent`` on the output channel
        (to attach meta). ``observation`` is ``row['observation']``; ``history``
        is a tuple of recent ``(observation, action)`` pairs, oldest first, up to
        ``max_history`` entries.
    :param max_history: rolling window length. ``0`` disables history (the policy
        always sees an empty tuple), making the wrapper effectively stateless.
    :param channel: the output channel the action is returned on.
    """

    def __init__(self, policy: Callable, max_history: int = 8, channel: str = 'motor'):
        self._policy = policy
        self._max_history = max_history
        self._channel = channel
        maxlen = max_history if max_history > 0 else 0
        self._history: Deque[Tuple[Any, Any]] = deque(maxlen=maxlen or None)

    @property
    def history(self) -> Tuple[Tuple[Any, Any], ...]:
        """The current rolling history (oldest first)."""
        return tuple(self._history)

    def reset(self) -> None:
        """Clear the rolling history."""
        self._history.clear()

    def __call__(self, row: Mapping[str, Any]) -> Mapping[str, Any]:
        history = () if self._max_history == 0 else tuple(self._history)
        observation = row.get('observation')
        result = self._policy(observation, history)
        if self._max_history != 0:
            action = result.payload if isinstance(result, StreamEvent) else result
            self._history.append((observation, action))
        return {self._channel: result}
