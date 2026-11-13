"""Closed-loop capability: the subject sees a row and returns outputs.

Closed-loop means each response can change what the subject observes next
(games, dialogue, robots). A step is a row: a mapping of channel -> payload. The
policy reads one row and returns a mapping of output channel -> payload (for
example ``{'motor': action}``); it needs no Brain-Score import. "Embodied" is
reserved for the robotics subset; core knows nothing about cameras or joints.
Device payloads are validated by their harness (for example
``brainscore.harnesses.robotics``), not here.
"""

from collections.abc import Mapping

from .base import Capability
from .registry import register_capability


class ClosedLoopCapability(Capability):
    """Route a row (one closed-loop step) to the subject's ``policy``.

    ``policy`` has one signature: it receives a row and returns a mapping of
    output channel -> payload. A message from another agent is a ``text`` entry
    of the row, with the sender in the event meta, so a responder handles games,
    dialogue and robots the same way.
    """

    identifier = 'closed-loop'
    order = 300

    def handles(self, subject, event, **kwargs) -> bool:
        del subject, kwargs
        return isinstance(event, Mapping)

    def process(self, subject, event, **kwargs):
        del kwargs
        if subject._policy is None:
            raise NotImplementedError(
                f"Subject '{subject.identifier}' has no policy. Closed-loop evaluation "
                f"requires BrainScoreModel(policy=...), a callable "
                f"policy(row) -> {{output channel: payload}}."
            )
        outputs = subject._policy(event)
        if not isinstance(outputs, Mapping):
            raise TypeError(
                f"Subject '{subject.identifier}' policy returned {type(outputs).__name__}; "
                f"expected a mapping of output channel -> payload, for example "
                f"{{'motor': action}}."
            )
        return outputs


register_capability(ClosedLoopCapability())
