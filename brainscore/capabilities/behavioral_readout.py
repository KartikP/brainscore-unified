"""Behavioral readout capability."""

from .base import Capability
from .registry import register_capability


class BehavioralReadoutCapability(Capability):
    """Dispatch fitted behavioral readout tasks."""

    identifier = 'behavioral-readout'
    order = 200

    def enabled_for(self, subject) -> bool:
        return (
            subject._behavioral.readout_classifier is not None
            and subject._behavioral.task is not None
            and subject._requires_behavioral_readout(subject._behavioral.task.task_type)
        )

    def handles(self, subject, event, **kwargs) -> bool:
        del subject, kwargs
        try:
            event.columns
        except AttributeError:
            return False
        return True

    def process(self, subject, stimuli, **kwargs):
        del kwargs
        return subject._predict_probabilities(stimuli)


register_capability(BehavioralReadoutCapability())
