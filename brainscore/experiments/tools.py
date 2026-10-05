"""Reusable recording and optional PyTorch instrumentation tools."""
from contextlib import contextmanager, ExitStack
from brainscore.run_record import RunRecorder
from .runner import Tool


class RecordInputsOutputs(Tool):
    """Snapshot events with trial IDs, call IDs, source and elapsed time."""
    name = 'inputs_outputs'

    @contextmanager
    def attach(self, context):
        path = context.directory / self.name
        try:
            with RunRecorder(path, metadata={'experiment': context.manifest['run_id'],
                                            'plan': context.manifest['plan']}) as recorder:
                def record(event):
                    recorder.record('input' if event['kind'] == 'input' else 'output', event)
                try:
                    with context.subscribe(record):
                        yield
                finally:
                    if context.failed:
                        recorder.close(failed=True)
        finally:
            for filename, description in [
                ('manifest.json', 'Record manifest with event-log checksum'),
                ('events.jsonl', 'Events with content-addressed array/file references'),
            ]:
                if (path / filename).is_file():
                    context.artifact(path / filename, producer='brainscore.RunRecorder',
                                     description=description)



class RecordActivity(Tool):
    def __init__(self, targets, *, when='after', name='activity'):
        if when not in ('before', 'after'):
            raise ValueError('when must be before or after')
        self.targets, self.when, self.name = tuple(targets), when, name
        self.phase = 'before_intervention' if when == 'before' else 'after_intervention'

    def describe(self):
        return {'targets': self.targets, 'when': self.when}

    def validate(self, experiment):
        if not any(isinstance(tool, RecordInputsOutputs) for tool in experiment.tools):
            raise ValueError('RecordActivity requires RecordInputsOutputs to persist its events')
        if experiment.instrumentation is None:
            raise ValueError('RecordActivity requires an instrumentation provider')
        experiment.instrumentation.validate('record', self.targets)

    @contextmanager
    def attach(self, context):
        def receive(target, value):
            context.publish('activity', {'target': target, 'value': value,
                                        'when': self.when}, source=self.name)
        with context.experiment.instrumentation.record(self.targets, receive):
            yield


class Ablate(Tool):
    """Zero target outputs for selected trials and remove only owned hooks."""
    name = 'ablation'
    phase = 'intervene'

    def __init__(self, targets, *, trials=None):
        self.targets = tuple(targets)
        self.trials = None if trials is None else tuple(trials)

    def describe(self):
        return {'targets': self.targets, 'trials': self.trials, 'operation': 'zero_output'}

    def validate(self, experiment):
        if experiment.instrumentation is None:
            raise ValueError('Ablate requires an instrumentation provider')
        experiment.instrumentation.validate('ablate', self.targets)
        available = getattr(experiment.protocol, 'trials', ('external',))
        if self.trials is not None and (not self.trials or set(self.trials) - set(available)):
            raise ValueError('Ablation trials must belong to the protocol')

    @contextmanager
    def attach(self, context):
        if self.trials is not None and context.trial_id not in self.trials:
            yield
            return
        with context.experiment.instrumentation.ablate(self.targets):
            context.publish('intervention_start', self.describe(), source=self.name)
            try:
                yield
            finally:
                context.publish('intervention_end', self.describe(), source=self.name)


class TorchInstrumentation:
    """Optional provider for tensor module outputs and whole-output zeroing.

    Explicit targets map names to torch modules, independently of subject type.
    Weights, inputs, random state and pre-existing hooks are not changed.
    """
    def __init__(self, targets):
        self.targets = dict(targets)

    def describe(self):
        return {'provider': 'TorchInstrumentation',
                'targets': {name: type(module).__module__ + '.' + type(module).__qualname__
                            for name, module in self.targets.items()},
                'operations': ['record', 'ablate'], 'scope': 'tensor module outputs'}

    def validate(self, operation, targets):
        import torch
        if operation not in ('record', 'ablate'):
            raise ValueError(f'Unsupported operation: {operation}')
        if not targets or len(set(targets)) != len(targets):
            raise ValueError('Select nonempty distinct targets')
        if len({id(self.targets.get(target)) for target in targets}) != len(targets):
            raise ValueError('Targets must refer to distinct modules')
        for target in targets:
            if target not in self.targets or not isinstance(self.targets[target], torch.nn.Module):
                raise ValueError(f'Unknown or invalid PyTorch target: {target}')

    @contextmanager
    def record(self, targets, receive):
        import torch
        self.validate('record', targets)
        with ExitStack() as stack:
            for target in targets:
                def hook(module, inputs, output, target=target):
                    if not isinstance(output, torch.Tensor):
                        raise TypeError('Activity target must return a tensor')
                    snapshot = output.detach().cpu().clone()
                    if snapshot.dtype == torch.bfloat16:
                        snapshot = snapshot.float()
                    receive(target, {'array': snapshot.numpy(), 'dtype': str(output.dtype),
                                     'device': str(output.device)})
                handle = self.targets[target].register_forward_hook(hook)
                stack.callback(handle.remove)
            yield

    @contextmanager
    def ablate(self, targets):
        import torch
        self.validate('ablate', targets)
        with ExitStack() as stack:
            for target in targets:
                def hook(module, inputs, output):
                    if not isinstance(output, torch.Tensor):
                        raise TypeError('Ablation target must return a tensor')
                    return torch.zeros_like(output)
                handle = self.targets[target].register_forward_hook(hook)
                stack.callback(handle.remove)
            yield
