"""Behavioral generation capability."""

from .base import Capability
from .registry import register_capability


class BehavioralGenerationCapability(Capability):
    """Dispatch instruction-following behavioral tasks."""

    identifier = 'behavioral-generation'
    order = 100

    def enabled_for(self, subject) -> bool:
        return (
            subject._behavioral.use_generation_for_task
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
        return subject._generate_predictions(stimuli)


register_capability(BehavioralGenerationCapability())
