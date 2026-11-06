"""Completion tracking for a fixed, passive behavioral stimulus sequence."""
from brainscore_core.streaming_helpers import BehavioralSession
from brainscore_core.streaming import BufferedSession


class FiniteSession(BufferedSession):
    """A fixed sequence, with an explicit receipt after input exhaustion."""

    complete = False

    def next_input(self) -> object:
        event = super().next_input()
        if event is None:
            self.complete = True
        return event


class FiniteBehaviorSession(BehavioralSession):
    """Complete only after the subject consumes the finite input sequence.

    The current experiments runner requires a completion receipt that the
    checked-out core's BufferedSession does not yet expose. This adapter applies
    only to passive sequences, not environments or live sources.
    """

    complete = False

    def next_input(self) -> object:
        event = super().next_input()
        if event is None:
            self.complete = True
        return event
