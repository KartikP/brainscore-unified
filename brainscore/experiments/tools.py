"""Reusable recording and optional PyTorch instrumentation tools."""
from contextlib import contextmanager, ExitStack
import math
from brainscore.run_record import RunRecorder
from brainscore_core.events import Selection, Perturbation, StateChange
from brainscore.instrumentation import observe
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
        return {'targets': [_describe_target(t) for t in self.targets], 'when': self.when}

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
    """Zero selected layer outputs in chosen conditions or repetitions."""
    name = 'ablation'
    phase = 'intervene'
    operation = 'ablate'

    def __init__(self, targets, *, conditions=None, trials=None):
        self.targets = tuple(targets)
        self.conditions = None if conditions is None else tuple(conditions)
        self.trials = None if trials is None else tuple(trials)

    def describe(self):
        return {'targets': [_describe_target(t) for t in self.targets], 'conditions': self.conditions,
                'trials': self.trials, 'operation': 'zero_output'}

    def validate(self, experiment):
        if experiment.instrumentation is None:
            raise ValueError('Ablate requires an instrumentation provider')
        experiment.instrumentation.validate(self.operation, self.targets)
        available_conditions = getattr(experiment.protocol, 'conditions', ('default',))
        if self.conditions is not None and (not self.conditions or set(self.conditions) - set(available_conditions)):
            raise ValueError('Ablation conditions must belong to the protocol')
        available = getattr(experiment.protocol, 'trials', ('external',))
        if self.trials is not None and (not self.trials or set(self.trials) - set(available)):
            raise ValueError('Ablation trials must belong to the protocol')

    @contextmanager
    def attach(self, context):
        if ((self.conditions is not None and context.condition not in self.conditions)
                or (self.trials is not None and context.trial_id not in self.trials)):
            yield
            return
        with self._apply(context.experiment.instrumentation):
            context.publish('intervention_start', self.describe(), source=self.name)
            try:
                yield
            finally:
                context.publish('intervention_end', self.describe(), source=self.name)

    def _apply(self, instrumentation):
        return instrumentation.ablate(self.targets)


class ScaleActivity(Ablate):
    """Multiply selected outputs by a fixed factor without changing model weights."""
    name = 'scale_activity'
    operation = 'scale'

    def __init__(self, targets, *, factor, conditions=None, trials=None):
        super().__init__(targets, conditions=conditions, trials=trials)
        if isinstance(factor, bool) or not isinstance(factor, (int, float)) or not math.isfinite(factor):
            raise ValueError('factor must be a finite number')
        self.factor = float(factor)

    def describe(self):
        return {**super().describe(), 'operation': 'scale_output', 'factor': self.factor}

    def _apply(self, instrumentation):
        return instrumentation.scale(self.targets, factor=self.factor)


def _selection(target):
    return Selection(layer=target) if isinstance(target, str) else target


def _describe_target(target):
    if isinstance(target, str):
        return target
    return {'layer': target.layer, 'indices': target.indices}


class ObserveCalls(Tool):
    """Attach an existing on_start/on_result/on_error observer to method calls.

    Native session tools use on_event instead; a session event is not a method
    call. This adapter preserves existing observers without inventing call data.
    """
    def __init__(self, observer, *, methods=('process',), name='call_observer'):
        self.observer, self.methods, self.name = observer, tuple(methods), name

    def describe(self):
        return {'observer': type(self.observer).__module__ + '.' + type(self.observer).__qualname__,
                'methods': self.methods}

    def validate(self, experiment):
        for method in self.methods:
            if not callable(getattr(experiment.subject, method, None)):
                raise ValueError(f'Subject does not expose {method}')

    @contextmanager
    def attach(self, context):
        with observe(context.experiment.subject, self.observer, methods=self.methods):
            yield


class TorchInstrumentation:
    """Expose model layer paths or explicitly named modules to experiment tools.

    Recording uses ActivationWindow; ablation uses build_pytorch_ablation_fn.
    Selection.indices refers to the last output axis, just as in StateChange.
    It is not automatically the channel axis of a convolutional output.
    """
    def __init__(self, model_or_targets):
        import torch
        self.targets = (dict(model_or_targets.named_modules())
                        if isinstance(model_or_targets, torch.nn.Module) else dict(model_or_targets))

    def describe(self):
        return {'provider': 'TorchInstrumentation',
                'targets': {name: type(module).__module__ + '.' + type(module).__qualname__
                            for name, module in self.targets.items()},
                'operations': ['record', 'ablate', 'scale'],
                'selection': 'layer path; optional last-axis indices'}

    def validate(self, operation, targets):
        import operator
        import torch
        if operation not in ('record', 'ablate', 'scale'):
            raise ValueError(f'Unsupported operation: {operation}')
        selections = [_selection(t) for t in targets]
        if not selections or any(not isinstance(s, Selection) for s in selections):
            raise ValueError('Select layer paths or Selection objects')
        names = [s.layer for s in selections]
        if len(set(names)) != len(names):
            raise ValueError('Select distinct layers; combine unit indices in one Selection')
        for selection in selections:
            if (selection.layer not in self.targets
                    or not isinstance(self.targets[selection.layer], torch.nn.Module)):
                raise ValueError(f'Unknown or invalid PyTorch target: {selection.layer}')
            if selection.indices is not None:
                for index in selection.indices:
                    if operator.index(index) < 0:
                        raise ValueError('Selection indices must be nonnegative')
        if len({id(self.targets[name]) for name in names}) != len(names):
            raise ValueError('Targets must refer to distinct modules')

    @contextmanager
    def record(self, targets, receive):
        from brainscore.activation_window import ActivationWindow, _first_tensor_out
        self.validate('record', targets)
        with ExitStack() as stack:
            for target in targets:
                selection = _selection(target)
                def select(output, selection=selection):
                    tensor = _first_tensor_out(output)
                    if tensor is None:
                        raise TypeError('Activity target must expose a tensor')
                    if selection.indices is not None:
                        tensor = tensor[..., selection.indices]
                    return tensor
                def capture(value, selection=selection):
                    import torch
                    snapshot = value.tensor.clone()
                    if snapshot.dtype == torch.bfloat16:
                        snapshot = snapshot.float()
                    receive(selection.layer, {'array': snapshot.numpy(), 'dtype': value.dtype,
                                              'device': value.device, 'indices': selection.indices})
                stack.enter_context(ActivationWindow(self.targets[selection.layer],
                    select_output=select, on_capture=capture, retain=False, max_captures=None))
            yield

    @contextmanager
    def ablate(self, targets):
        with self._perturb(targets, Perturbation(kind='zero')):
            yield

    @contextmanager
    def scale(self, targets, *, factor):
        if isinstance(factor, bool) or not isinstance(factor, (int, float)) or not math.isfinite(factor):
            raise ValueError('factor must be a finite number')
        with self._perturb(targets, Perturbation(kind='scale', scale=float(factor))):
            yield

    @contextmanager
    def _perturb(self, targets, perturbation):
        import torch
        from brainscore.perturbation import build_pytorch_ablation_fn
        self.validate('ablate', targets)
        with ExitStack() as stack:
            for target in targets:
                selection = _selection(target)
                module = self.targets[selection.layer]
                def validate_output(module, inputs, output):
                    tensor = output[0] if isinstance(output, tuple) and output else output
                    if not isinstance(tensor, torch.Tensor):
                        raise TypeError('Ablation target must return a tensor or a tuple beginning with one')
                handle = module.register_forward_hook(validate_output)
                stack.callback(handle.remove)
                # An auxiliary root provides a stable path without modifying the model.
                root = torch.nn.ModuleDict({'target': module})
                apply = build_pytorch_ablation_fn(root)
                _, cleanup = apply(StateChange(kind='ablation',
                    target=Selection(layer='target', indices=selection.indices),
                    perturbation=perturbation))
                stack.callback(cleanup)
            yield
