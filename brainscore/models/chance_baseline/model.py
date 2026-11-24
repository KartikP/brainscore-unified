"""
Pure-chance null baseline.

Ignores the stimuli entirely and returns a uniform distribution over
the label set on every call. Used as a floor for behavioral benchmarks:
if this gets > 0.5 for balanced binary tasks, the benchmark's scoring
has a leak somewhere.

A ``Subject`` with typed methods. It has no extractor,
no activations, and no logistic readout. A ``Choice`` task gets uniform rows on
``behavior:probabilities`` (an old probabilities benchmark asks there through the
legacy facade) and, on ``behavior:choice``, the choice derived from those rows by
core's ``argmax`` (ties go to the first label), with a record that says so. A task
with a ``label_set`` on bare ``behavior`` gets its uniform rows through ``process``.
"""

from typing import Any, Dict, Set

import numpy as np
import pandas as pd

from brainscore import BrainScoreModel
from collections.abc import Mapping

from brainscore_core.contract import Subject, TaskContext
from brainscore_core.events import StateChange
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
        self._task = None

    @property
    def identifier(self) -> str:
        return self._identifier

    @property
    def region_layer_map(self) -> Dict[str, str]:
        return {}

    @property
    def in_channels(self) -> Set[str]:
        # Accept any input, so a benchmark that checks inputs sees that every
        # one is taken by design.
        return {'vision:image', 'vision:video', 'text', 'audio'}

    @property
    def out_channels(self) -> Set[str]:
        # It answers behavioral tasks; declared, not inferred (Contract v3 rule 2):
        # a task with a label_set on bare behavior, and the choice and probabilities of a Choice.
        return {'behavior', 'behavior:choice', 'behavior:probabilities'}

    def participate(self, session) -> None:
        from brainscore_core.streaming_helpers import drain_session_inputs, drive_typed_session, refuse_variance
        outputs = set(session.outputs or ())
        if not outputs or not outputs <= {'behavior:choice', 'behavior:probabilities'}:
            try:        # the task the session sets holds for this call only
                if outputs == {'behavior'}:
                    self._task = session.task
                    self._label_set()       # a task with no chance answer: refused before the inputs are read
                return drive_typed_session(self, session)
            finally:
                self.reset()
        refuse_variance(self, session)   # identical rows are no variance: refused before the inputs are read
        # Each requested format from the same inputs: one request for both gives both.
        events = drain_session_inputs(session)
        if 'behavior:probabilities' in outputs:
            self._answer_probabilities(session, events)
        if 'behavior:choice' in outputs:
            self._answer_choice(session, events)

    def _answer_choice(self, session, events) -> None:
        """One answer for the whole table: the choice derived from uniform rows, one row
        for each trial (a stimulus shown twice is answered twice)."""
        from brainscore_core.response_formats import argmax, response_record
        from brainscore_core.streaming import StreamEvent
        labels = session.task.response.labels
        if isinstance(labels, str):
            raise ValueError("chance-baseline answers a Choice with shared options; options per "
                             f"row (Choice({labels!r})) are not supported.")
        labels = list(labels)
        trials = []
        for event in events:
            meta = event.meta or {}
            if meta.get('stimulus_id') is not None:
                trial = (meta['stimulus_id'], meta.get('table_row', meta.get('stimulus_index')))
                if trial not in trials:      # one trial may send several columns
                    trials.append(trial)
        ids = [stimulus_id for stimulus_id, _ in trials]
        rows = np.full((len(ids), len(labels)), 1.0 / len(labels))
        indices, valid = argmax(rows)
        one_hot = np.where(valid[:, None], np.eye(len(labels))[indices], np.nan).astype(np.float32)
        answer = BehavioralAssembly(
            one_hot, dims=['presentation', 'choice'],
            coords={'stimulus_id': ('presentation', ids), 'choice': labels,
                    'raw_answer': ('presentation', [str([float(v) for v in row]) for row in rows])},
            attrs={'response': response_record('behavior:choice', 'uniform', conversion='argmax',
                                               source_format='behavior:probabilities')})
        session.respond(StreamEvent('behavior:choice', answer, None, {}))

    def _answer_probabilities(self, session, events) -> None:
        """Uniform rows over the Choice's labels, one per presented stimulus, with the
        stimulus columns (float32, as this model gives its rows)."""
        from brainscore_core.response_formats import response_record
        from brainscore_core.streaming import StreamEvent
        from brainscore_core.streaming_helpers import session_stimuli
        labels = session.task.response.labels
        if isinstance(labels, str):
            raise ValueError("chance-baseline answers a Choice with shared options; options per "
                             f"row (Choice({labels!r})) are not supported.")
        stimuli = session_stimuli(session, events)
        rows = np.full((len(stimuli), len(labels)), 1.0 / len(labels), dtype=np.float32)
        coords = {'stimulus_id': ('presentation', list(stimuli['stimulus_id'].values))}
        coords.update({column: ('presentation', list(stimuli[column].values))
                       for column in stimuli.columns if column != 'stimulus_id'})
        answer = BehavioralAssembly(rows, coords={**coords, 'choice': list(labels)},
                                    dims=['presentation', 'choice'],
                                    attrs={'response': response_record('behavior:probabilities', 'uniform')})
        session.respond(StreamEvent('behavior:probabilities', answer, None, {}))

    def _label_set(self) -> list:
        """The labels of the task's ``label_set``. Other tasks (odd one out, reading times)
        have no chance answer here: refused, not invented."""
        from brainscore_core.compatibility import ExecutionNotFeasibleError
        task = self._task
        task_type = getattr(task, 'task_type', None)
        if task is not None and task.label_set:
            return list(task.label_set)
        raise ExecutionNotFeasibleError(
            f"chance-baseline answers a task with a label_set on bare behavior; it has no chance "
            f"answer for task_type={task_type!r}.", channels={'behavior'}, task_type=task_type)

    def start_task(self, task_or_type, fitting_stimuli=None, **kwargs) -> None:
        if isinstance(task_or_type, TaskContext):
            self._task = task_or_type
        else:
            self._task = TaskContext(
                task_type=task_or_type, fitting_stimuli=fitting_stimuli,
            )

    def start_recording(self, *args, **kwargs) -> None:
        pass

    def reset(self) -> None:
        self._task = None

    def process(self, input_event) -> Any:
        if isinstance(input_event, (StateChange, Mapping)):
            from brainscore_core.compatibility import ExecutionNotFeasibleError
            raise ExecutionNotFeasibleError(
                "chance-baseline answers stimuli only, not a state change or a closed-loop step. Ask "
                "for 'behavior', 'behavior:choice' or 'behavior:probabilities' with a stimulus table.")
        stimuli = input_event
        label_set = self._label_set()
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

    def visual_degrees(self) -> int:
        return 8


def get_model(identifier: str) -> ChanceBaseline:
    assert identifier == 'chance-baseline'
    return ChanceBaseline(identifier=identifier)
