"""Run a task on a subject with recording equipment attached.

    experiment = Experiment(
        subject=model,
        protocol=TaskProtocol(stimuli, task_context, conditions=['baseline', 'lesioned']),
        tools=[RecordNeural(['IT']), RecordBehavior(), Intervene(change, conditions=['lesioned'])],
        output_dir='runs/study')
    result = experiment.run()
    result.neural('IT', condition='lesioned'); result.behavior(condition='lesioned')

Each stimulus is presented once per trial. The recorders ask the subject for
their channels in that one session, so the neural recording comes from the
same presentation that produced the behavior.
"""
from contextlib import contextmanager
from dataclasses import replace

from brainscore_core.events import StateChange
from brainscore_core.streaming import BufferedSession
from brainscore_core.streaming_helpers import apply_state_change

from .runner import SessionProtocol, Tool
from .tools import RecordInputsOutputs


class TaskProtocol(SessionProtocol):
    """Present ``stimuli`` under ``task``, once per condition and trial.

    The task says what the subject does; ``TaskContext(task_type='passive')``
    is plain viewing. The attached recorders say what is measured: each
    session requests exactly their channels. With a readout task, the readout
    is fitted on the intact network and tested on the changed one; pass
    ``refit_readout=True`` to fit it on the changed network instead.
    """

    def __init__(self, stimuli, task, *, identifier='task', conditions=None,
                 trials=(0,), refit_readout=False, metadata=None):
        probe = BufferedSession(stimulus_set=stimuli)
        inputs = set()
        while (event := probe.next_input()) is not None:
            inputs.add(event.channel)
        self.stimuli = stimuli
        self.task = replace(task, readout_fit='current' if refit_readout else 'intact')
        super().__init__(identifier, self._session, conditions=conditions, trials=trials,
                         input_channels=inputs, output_channels=(), metadata=metadata)

    def accept_requests(self, channels):
        if not channels:
            raise ValueError('Attach a recorder (RecordNeural, RecordBehavior): '
                             'nothing would be measured.')
        if 'behavior' in channels and self.task.task_type == 'passive':
            raise ValueError('A passive task produces no behavior; remove RecordBehavior '
                             'or give the task a task_type that asks for an answer.')
        self.output_channels = frozenset(channels)

    def describe(self):
        return {**super().describe(), 'task_type': self.task.task_type,
                'readout_fit': self.task.readout_fit}

    @contextmanager
    def _session(self, trial):
        yield BufferedSession(stimulus_set=self.stimuli, task_context=self.task,
                              requested_output_channels=sorted(self.output_channels))


class _ChannelRecorder(RecordInputsOutputs):
    """Save only this recorder's output channels, under its own folder."""

    def validate(self, experiment):
        missing = set(self.output_channels) - set(experiment.subject.out_channels)
        if missing:
            raise ValueError(f'{experiment.subject.identifier!r} does not declare {sorted(missing)}')

    def _keeps(self, event):
        return event['kind'] == 'output' and event['payload'].channel in self.output_channels


class RecordNeural(_ChannelRecorder):
    """Record ``regions`` during the task. Read back with ``result.neural(region)``.

    For a model that answers by generating text, ``during='prompt'`` keeps the
    forward pass that reads the stimulus and instruction; the passes that
    generate the reply are not kept.
    """
    name = 'neural'

    def __init__(self, regions, *, during='prompt', max_bytes=None):
        super().__init__(max_bytes=max_bytes)
        if isinstance(regions, str) or not regions:
            raise ValueError('Pass a nonempty list of region names')
        if during != 'prompt':
            # ponytail: only the prompt pass exists; add 'all_tokens' when a study needs it.
            raise ValueError("during='prompt' is the only option: the forward pass that "
                             'reads the stimulus and instruction, before any generated token.')
        self.regions, self.during = list(regions), during
        self.output_channels = frozenset(f'neural:{region}' for region in self.regions)

    def describe(self):
        return {**super().describe(), 'regions': self.regions, 'during': self.during}


class RecordBehavior(_ChannelRecorder):
    """Record the subject's answers. Read back with ``result.behavior()``."""
    name = 'behavior'
    output_channels = frozenset({'behavior'})


class Intervene(Tool):
    """Apply one undoable change through the subject's own ``intervention``.

    The change is applied at the start of each matching trial and undone at
    its end. Targets use the subject's layer paths, as in ``region_layer_map``.
    """
    name = 'intervention'
    phase = 'intervene'

    def __init__(self, change: StateChange, *, conditions=None, name='intervention'):
        self.change, self.name = change, name
        self.conditions = None if conditions is None else tuple(conditions)

    def describe(self):
        target = self.change.target
        return {'kind': self.change.kind, 'layer': getattr(target, 'layer', None),
                'indices': getattr(target, 'indices', None),
                'perturbation': self.change.perturbation.kind, 'conditions': self.conditions}

    def validate(self, experiment):
        if 'lesion' not in experiment.subject.in_channels and \
                'stimulation' not in experiment.subject.in_channels:
            raise ValueError(f'{experiment.subject.identifier!r} declares no intervention')

    @contextmanager
    def attach(self, context):
        if self.conditions is not None and context.condition not in self.conditions:
            yield
            return
        subject = context.experiment.subject
        applied = apply_state_change(subject, self.change)
        context.publish('intervention', {'handle_id': applied.handle_id}, source=self.name)
        try:
            yield
        finally:
            apply_state_change(subject, StateChange(kind='reset', handle_id=applied.handle_id))
