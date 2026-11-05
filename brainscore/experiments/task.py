"""Run a task on a subject with recording equipment attached.

    experiment = Experiment(
        subject=model,
        protocol=TaskProtocol(stimuli, task_context, conditions=['baseline', 'lesioned']),
        tools=[RecordNeural(['IT']), RecordBehavior(), Intervene(change, conditions=['lesioned'])],
        output_dir='runs/study')
    result = experiment.run()
    result.neural('IT', condition='lesioned'); result.behavior(condition='lesioned')

Each stimulus is presented once per repetition. The recorders ask the subject for
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
    """Present ``stimuli`` under ``task``, once per condition and repetition.

    The task says what the subject does; ``TaskContext(task_type='passive')``
    is plain viewing. ``required_traits`` (e.g. ``{'memory:session'}``) makes a
    subject without them fail validation, as a benchmark marks it N/A. The
    attached recorders say what is measured: each
    session requests exactly their channels. With a readout task, the task's
    ``readout_fit`` says which network the readout is fitted on while a change
    is active: ``TaskContext(..., readout_fit='intact')`` fits it before the
    change and tests the changed network, as an animal trained before a lesion.
    """

    def __init__(self, stimuli, task, *, identifier='task', conditions=None,
                 repetitions=(0,), required_traits=(), metadata=None):
        probe = BufferedSession(stimulus_set=stimuli)
        inputs = set()
        while (event := probe.next_input()) is not None:
            inputs.add(event.channel)
        self.stimuli = stimuli
        self.task = task
        # e.g. {'memory:session'} for a task whose trials depend on earlier ones.
        self.required_traits = frozenset(required_traits)
        super().__init__(identifier, self._session, conditions=conditions, repetitions=repetitions,
                         input_channels=inputs, output_channels=(), metadata=metadata)

    def accept_requests(self, channels):
        if not channels:
            raise ValueError('Attach a recorder (RecordNeural, RecordBehavior): '
                             'nothing would be measured.')
        channels = set(channels)
        if 'behavior' in channels:
            if self.task.task_type == 'passive':
                raise ValueError('A passive task produces no behavior; remove RecordBehavior '
                                 'or give the task a response (Choice, Scale, Text).')
            if self.task.response is not None:
                # Every kind of answer the response asks for: e.g. choice and confidence.
                channels = (channels - {'behavior'}) | set(self.task.response.channels)
        self.output_channels = frozenset(channels)

    def validate(self, subject):
        missing = self.required_traits - set(subject.traits)
        if missing:
            raise ValueError(f'This task needs subject traits {sorted(missing)}; '
                             f'{subject.identifier!r} declares {sorted(subject.traits) or "none"}.')
        if self.task.response is not None:
            missing = set(self.task.response.channels) - set(subject.out_channels)
            if missing:
                raise ValueError(f'Task response requires outputs {sorted(missing)} even when '
                                 f'behavior is not recorded; subject {subject.identifier!r} '
                                 f'declares {sorted(subject.out_channels)}.')
        super().validate(subject)

    def describe(self):
        return {**super().describe(), 'task_type': self.task.task_type,
                'required_traits': sorted(self.required_traits),
                'readout_fit': self.task.readout_fit}

    @contextmanager
    def _session(self, repetition):
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
    """Record the subject's answers: every kind the task's response asks for.

    Read back with ``result.behavior()``, or ``result.behavior(measure='confidence')``.
    """
    name = 'behavior'
    output_channels = frozenset({'behavior'})

    def _keeps(self, event):
        channel = event['payload'].channel if event['kind'] == 'output' else ''
        return channel == 'behavior' or channel.startswith('behavior:')


class Intervene(Tool):
    """Apply one undoable change through the subject's own ``intervention``.

    The change is applied at the start of each matching repetition and undone at
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
