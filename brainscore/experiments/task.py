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

from brainscore_core.events import StateChange
from brainscore_core.streaming import BufferedSession
from brainscore_core.streaming_helpers import apply_state_change

from .runner import SessionProtocol, Tool
from .tools import RecordInputsOutputs


class TaskProtocol(SessionProtocol):
    """Present ``stimuli`` under ``task_context``, once per condition and trial.

    The session requests ``behavior`` plus every channel the attached tools
    request (``RecordNeural`` asks for ``neural:<region>``).
    """

    def __init__(self, stimuli, task_context, *, identifier='task', conditions=None,
                 trials=(0,), metadata=None):
        probe = BufferedSession(stimulus_set=stimuli)
        inputs = set()
        while (event := probe.next_input()) is not None:
            inputs.add(event.channel)
        self.stimuli, self.task_context = stimuli, task_context
        self._requested = ('behavior',)
        super().__init__(identifier, self._session, conditions=conditions, trials=trials,
                         input_channels=inputs, output_channels={'behavior'},
                         metadata=metadata)

    @contextmanager
    def _session(self, trial):
        yield BufferedSession(stimulus_set=self.stimuli, task_context=self.task_context,
                              requested_output_channels=list(self._requested))

    def run(self, subject, context):
        requested = {'behavior'}.union(*(tool.output_channels for tool in context.experiment.tools))
        self._requested = tuple(sorted(requested))
        return super().run(subject, context)


class _ChannelRecorder(RecordInputsOutputs):
    """Save only this recorder's output channels, under its own folder."""

    def validate(self, experiment):
        missing = set(self.output_channels) - set(experiment.subject.out_channels)
        if missing:
            raise ValueError(f'{experiment.subject.identifier!r} does not declare {sorted(missing)}')

    def _keeps(self, event):
        return event['kind'] == 'output' and event['payload'].channel in self.output_channels


class RecordNeural(_ChannelRecorder):
    """Record ``regions`` during the task. Read back with ``result.neural(region)``."""
    name = 'neural'

    def __init__(self, regions, *, max_bytes=None):
        super().__init__(max_bytes=max_bytes)
        if isinstance(regions, str) or not regions:
            raise ValueError('Pass a nonempty list of region names')
        self.regions = list(regions)
        self.output_channels = frozenset(f'neural:{region}' for region in self.regions)

    def describe(self):
        return {**super().describe(), 'regions': self.regions}


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
