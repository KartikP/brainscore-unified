"""The process-level step that hands a StateChange to the intervention part."""

from .base import Capability
from .registry import register_capability
from brainscore_core.events import StateChange


class InterventionCapability(Capability):
    """Hand a ``StateChange`` to the subject's ``intervention`` part.

    The intervention session driver reaches this step; users send a change with
    ``apply_state_change`` or the ``Intervene`` tool.
    """

    identifier = 'intervention'
    order = 50

    def handles(self, model, event, **kwargs) -> bool:
        del model, kwargs
        return isinstance(event, StateChange)

    def process(self, model, event, **kwargs):
        del kwargs
        return model._interventions.dispatch_state_change(event)


register_capability(InterventionCapability())
