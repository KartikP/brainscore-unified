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

    def handles(self, subject, event, **kwargs) -> bool:
        del subject, kwargs
        return isinstance(event, StateChange)

    def process(self, subject, event, **kwargs):
        del kwargs
        try:
            return subject._interventions.dispatch_state_change(event)
        finally:
            # The subject changed under the readout: the next start_task fits
            # again even when the task object is the one already configured.
            subject._behavioral.invalidate()


register_capability(InterventionCapability())
