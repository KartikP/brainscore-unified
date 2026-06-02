"""PolicyWrapper: a stateful model harness for closed-loop embodied dispatch.

v1.5 builds the closed-loop embodied session that the design reserved. A
benchmark drives the loop by calling ``process(EnvironmentStep(...))`` once per
tick, applying the returned action, and producing the next step, without calling
``reset()`` in between. A stateless policy ignores prior steps; a stateful policy
needs to carry context across them.

``PolicyWrapper`` provides that state. It wraps a per-step policy callable,
maintains a rolling history of recent (observation, action) pairs, clears the
history when ``EnvironmentStep.is_first`` is set, and exposes an
``action_fn``-compatible ``__call__`` so it plugs straight into
``BrainScoreModel(action_fn=...)``. The model contract is unchanged: this is a
harness around the policy, not a new method on the interface.
"""
from collections import deque
from typing import Any, Callable, Deque, Optional, Tuple

from brainscore_core.model_interface import EnvironmentResponse, EnvironmentStep


class PolicyWrapper:
    """Wrap a per-step policy into a stateful ``action_fn``.

    :param policy: ``callable(observation, history) -> action`` where ``action``
        is either a raw action payload or an :class:`EnvironmentResponse`.
        ``history`` is a tuple of recent ``(observation, action)`` pairs, oldest
        first, up to ``max_history`` entries.
    :param max_history: rolling window length. ``0`` disables history (the policy
        always sees an empty tuple), making the wrapper effectively stateless.
    :param reset_on_first: when True (default), clear the history whenever an
        ``EnvironmentStep`` arrives with ``is_first=True``.
    """

    def __init__(self, policy: Callable, max_history: int = 8,
                 reset_on_first: bool = True):
        self._policy = policy
        self._max_history = max_history
        self._reset_on_first = reset_on_first
        maxlen = max_history if max_history > 0 else 0
        self._history: Deque[Tuple[Any, Any]] = deque(maxlen=maxlen or None)

    @property
    def history(self) -> Tuple[Tuple[Any, Any], ...]:
        """The current rolling history (oldest first)."""
        return tuple(self._history)

    def reset(self) -> None:
        """Clear the rolling history."""
        self._history.clear()

    def __call__(self, env_step: EnvironmentStep) -> EnvironmentResponse:
        if self._reset_on_first and getattr(env_step, "is_first", False):
            self.reset()

        history = () if self._max_history == 0 else tuple(self._history)
        result = self._policy(env_step.observation, history)
        response = result if isinstance(result, EnvironmentResponse) \
            else EnvironmentResponse(action=result)

        if self._max_history != 0:
            self._history.append((env_step.observation, response.action))
        return response
