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
    """Route a row (one closed-loop step) to the model's ``policy``.

    ``policy`` has one signature: it receives a row and returns a mapping of
    output channel -> payload. A message from another agent is a ``text`` entry
    of the row, with the sender in the event meta, so a responder handles games,
    dialogue and robots the same way.
    """

    identifier = 'closed-loop'
    order = 300

    def handles(self, model, event, **kwargs) -> bool:
        del model, kwargs
        return isinstance(event, Mapping)

    def process(self, model, event, **kwargs):
        del kwargs
        if model._action_fn is None:
            raise NotImplementedError(
                f"Subject '{model.identifier}' has no policy. Closed-loop evaluation "
                f"requires BrainScoreModel(policy=...), a callable "
                f"policy(row) -> {{output channel: payload}}."
            )
        outputs = model._action_fn(event)
        if not isinstance(outputs, Mapping):
            raise TypeError(
                f"Subject '{model.identifier}' policy returned {type(outputs).__name__}; "
                f"expected a mapping of output channel -> payload, for example "
                f"{{'motor': action}}."
            )
        return outputs


register_capability(ClosedLoopCapability())
