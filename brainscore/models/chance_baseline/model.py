"""
Pure-chance null baseline.

Ignores the stimuli entirely and returns a uniform distribution over
the label set on every call. Used as a floor for behavioral benchmarks:
if this gets > 0.5 for balanced binary tasks, the benchmark's scoring
has a leak somewhere.

Implemented as a minimal Subject subclass rather than a
BrainScoreModel — it has no preprocessors, no activations, no logistic.
Benchmarks that use start_task() and process() still work.
"""

from typing import Any, Dict, Set

import numpy as np
import pandas as pd

from brainscore_core.model_interface import (
    BrainScoreModel, EnvironmentStep, StateChange, Subject, TaskContext,
)
from brainscore_core.supported_data_standards.brainio.assemblies import (
    BehavioralAssembly,
)


class ChanceBaseline(Subject):
    """A model that returns uniform probabilities over the label set.

    Deterministic and reproducible (same output every call). For a
    balanced binary task the expected accuracy via argmax is 0.5.
    """

    def __init__(self, identifier: str = 'chance-baseline'):
        self._identifier = identifier
        self._task_context = None

    @property
    def identifier(self) -> str:
        return self._identifier

    @property
    def region_layer_map(self) -> Dict[str, str]:
        return {}

    @property
    def supported_modalities(self) -> Set[str]:
        # Accept any modality — benchmarks that check modality will
        # see that we support everything by design.
        return {'vision', 'text', 'audio', 'video'}

    def start_task(self, task_context_or_task, fitting_stimuli=None, **kwargs) -> None:
        if isinstance(task_context_or_task, TaskContext):
            self._task_context = task_context_or_task
        else:
            self._task_context = TaskContext(
                task_type=task_context_or_task, fitting_stimuli=fitting_stimuli,
            )

    def start_recording(self, *args, **kwargs) -> None:
        pass

    def reset(self) -> None:
        self._task_context = None

    def process(self, input_event) -> Any:
        if isinstance(input_event, (StateChange, EnvironmentStep)):
            raise NotImplementedError
        stimuli = input_event
        if self._task_context is None or not self._task_context.label_set:
            raise ValueError(
                "ChanceBaseline requires a TaskContext with label_set. "
                "Call start_task(TaskContext(task_type='probabilities', "
                "label_set=[...])) first.")

        label_set = list(self._task_context.label_set)
        n_stimuli = len(stimuli)
        n_labels = len(label_set)
        # Uniform distribution over labels
        probs = np.full((n_stimuli, n_labels), 1.0 / n_labels, dtype=np.float32)

        stimulus_ids = list(stimuli['stimulus_id'].values)
        presentation_coords = {
            'stimulus_id': ('presentation', stimulus_ids),
        }
        for column in stimuli.columns:
            if column == 'stimulus_id':
                continue
            presentation_coords[column] = ('presentation', list(stimuli[column].values))

        return BehavioralAssembly(
            probs,
            coords={**presentation_coords, 'choice': label_set},
            dims=['presentation', 'choice'],
        )

    # Legacy compat (in case benchmarks call these directly)
    def look_at(self, stimuli, **kwargs):
        return self.process(stimuli)

    def visual_degrees(self) -> int:
        return 8


def get_model(identifier: str) -> ChanceBaseline:
    assert identifier == 'chance-baseline'
    return ChanceBaseline(identifier=identifier)
