"""Synchronous, single-subject experiment lifecycle and artifact provenance."""
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import time
import uuid

from brainscore.instrumentation import ObservedSession, observe
from brainscore.run_record import PayloadCodec, RunRecord


def _identity(value):
    cls = type(value)
    return f'{cls.__module__}.{cls.__qualname__}'


def _write(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


class Tool:
    """Observe events with on_event or manage resources with attach; no registry needed.

    attach(context) returns a context manager. Release resources in finally,
    including if attachment itself fails. Events contain live payloads; copy or
    serialize them before returning. Observers must not mutate these payloads.
    """
    name = 'tool'
    phase = 'observe'

    def describe(self):
        return {}

    def validate(self, experiment):
        pass

    def on_event(self, event):
        """Observe one experiment event. Override for a simple external tool."""
        pass

    @contextmanager
    def attach(self, context):
        with context.subscribe(self.on_event):
            yield


@dataclass(frozen=True)
class Trial:
    """One repetition within an experimental condition."""
    condition: str
    identifier: object


class SessionProtocol:
    """Factory creates a fresh context-managed session for each independent trial.

    The protocol owns trial progression and resets. The factory owns session
    resource cleanup. Channel declarations describe the protocol, not a modality.

    With explicit conditions, the factory receives Trial(condition, identifier).
    Omitting conditions preserves the single-condition factory(trial_id) form.

    """
    def __init__(self, identifier, session_factory, *, conditions=None, trials=(0,),
                 input_channels=(), output_channels=(), metadata=None):
        self.identifier = identifier
        self.session_factory = session_factory
        self.trials = tuple(trials)
        self.conditions = tuple(conditions) if conditions is not None else ('default',)
        self._legacy_factory = conditions is None
        self.input_channels = frozenset(input_channels)
        self.output_channels = frozenset(output_channels)
        self.metadata = dict(metadata or {})

    def describe(self):
        return {'conditions': self.conditions, 'trials': self.trials, 'input_channels': sorted(self.input_channels),
                'output_channels': sorted(self.output_channels), 'metadata': self.metadata,
                'reset_policy': 'before and after each trial; tools attach after initial reset'}

    def validate(self, subject):
        if not callable(getattr(subject, 'interact', None)):
            raise TypeError('SessionProtocol requires subject.interact(session)')
        if not callable(getattr(subject, 'reset', None)):
            raise TypeError('SessionProtocol requires subject.reset()')
        if (not self.conditions or any(not isinstance(c, str) or not c for c in self.conditions)
                or len(set(self.conditions)) != len(self.conditions)):
            raise ValueError('Use nonempty, distinct condition names')
        if not self.trials or len(set(self.trials)) != len(self.trials):
            raise ValueError('Use nonempty, distinct, hashable trial IDs')
        missing = self.input_channels - set(subject.in_channels)
        missing_required = set(getattr(subject, 'required_channels', ())) - self.input_channels
        missing_outputs = self.output_channels - set(subject.out_channels)
        if missing or missing_required or missing_outputs:
            raise ValueError(f'Incompatible channels: unsupported inputs={missing}, '
                             f'missing required inputs={missing_required}, outputs={missing_outputs}')

    def run(self, subject, context):
        results = []
        for condition in self.conditions:
            for trial in self.trials:
                request = trial if self._legacy_factory else Trial(condition, trial)
                try:
                    subject.reset()
                    with context.trial(trial, condition=condition), self.session_factory(request) as session:
                        subject.interact(ObservedSession(session, context))
                        results.append(session.collect() if callable(getattr(session, 'collect', None)) else None)
                finally:
                    subject.reset()
        return results


class CallableProtocol:
    """Preserve an external evaluator's loop and scientific scoring.

    evaluate(subject, context) owns reset/seed/trial semantics. Its selected
    public methods are observed, including errors caught by the evaluator.
    Instrumentation is scoped to the whole evaluator call. Use separate runs
    for baseline and intervention conditions.
    """
    def __init__(self, identifier, evaluate, *, methods=('process',), metadata=None):
        self.identifier, self.evaluate = identifier, evaluate
        self.methods = tuple(methods)
        self.metadata = dict(metadata or {})

    def describe(self):
        return {'methods': self.methods, 'metadata': self.metadata,
                'reset_policy': 'owned by external evaluator'}

    def validate(self, subject):
        if not self.methods or len(set(self.methods)) != len(self.methods):
            raise ValueError('Select distinct methods to observe')
        for method in self.methods:
            if not callable(getattr(subject, method, None)):
                raise ValueError(f'Subject does not expose {method}')

    def run(self, subject, context):
        with context.trial('external'), observe(subject, context, methods=self.methods):
            return self.evaluate(subject, context)


@dataclass(frozen=True)
class ExperimentResult:
    directory: Path
    value: object

    @property
    def record(self):
        """Inspect saved events with the shared RunRecord API."""
        return RunRecord(self.directory)

    @property
    def manifest(self):
        return json.loads((self.directory / 'experiment.json').read_text())


class RunContext:
    def __init__(self, experiment, manifest):
        self.experiment, self.manifest = experiment, manifest
        self.directory = experiment.output_dir
        self.codec = PayloadCodec(self.directory)
        self.listeners = []
        self.trial_id = None
        self.condition = 'default'
        self.event_id = None
        self.sequence = 0
        self.failed = False
        self.started = time.monotonic()

    @contextmanager
    def subscribe(self, callback):
        self.listeners.append(callback)
        try:
            yield
        finally:
            self.listeners.remove(callback)

    def publish(self, kind, payload, *, source='protocol', event_id=None):
        envelope = {'kind': kind, 'payload': payload, 'source': source,
                    'condition': self.condition, 'trial_id': self.trial_id, 'event_id': event_id or self.event_id,
                    'sequence': self.sequence, 'elapsed_s': time.monotonic() - self.started}
        self.sequence += 1
        try:
            for callback in tuple(self.listeners):
                callback(envelope)
        except BaseException:
            self.failed = True
            raise

    def record(self, direction, event):
        if direction == 'input':
            self.event_id = uuid.uuid4().hex
        self.publish(direction, event)

    def on_start(self, call):
        self.event_id = call.event_id
        self.publish('input', {'method': call.method, 'args': call.args, 'kwargs': call.kwargs},
                     event_id=call.event_id)

    def on_result(self, call, result):
        self.publish('output', result, event_id=call.event_id)

    def on_error(self, call, error):
        self.failed = True
        self.publish('error', {'type': type(error).__name__, 'message': str(error)},
                     event_id=call.event_id)

    @contextmanager
    def trial(self, identifier, *, condition='default'):
        self.condition = condition
        self.trial_id, self.event_id = identifier, None
        self.publish('trial_start', {'id': identifier})
        try:
            with ExitStack() as stack:
                for tool in self.experiment.tools:
                    if tool.phase != 'observe':
                        stack.enter_context(tool.attach(self))
                yield
        except BaseException:
            self.failed = True
            raise
        finally:
            self.publish('trial_end', {'failed': self.failed})
            self.trial_id, self.event_id = None, None
            self.condition = 'default'

    def artifact(self, path, *, producer, description):
        """Register an existing output inside this run; never infer its producer."""
        path = Path(path).resolve()
        relative = path.relative_to(self.directory.resolve())
        digest = hashlib.sha256()
        with path.open('rb') as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b''):
                digest.update(block)
        row = {'path': str(relative), 'producer': producer, 'description': description,
               'sha256': digest.hexdigest(), 'bytes': path.stat().st_size}
        self.manifest['artifacts'].append(row)
        return row

    def import_artifact(self, path, *, name, producer, description):
        """Copy an external evaluator output without claiming UMI generated it."""
        if Path(name).name != name or name in ('', '.', '..'):
            raise ValueError('Artifact name must be a filename')
        destination = self.directory / 'external' / name
        destination.parent.mkdir(exist_ok=True)
        with Path(path).open('rb') as src, destination.open('xb') as dst:
            shutil.copyfileobj(src, dst)
        return self.artifact(destination, producer=producer, description=description)


class Experiment:
    """One execution, one subject instance, one new output directory.

    Protocols and tools are ordinary Python objects and can live in external
    packages. No concurrent reuse of a subject, tool, or provider is supported.
    """
    def __init__(self, *, subject, protocol, tools=(), output_dir,
                 instrumentation=None, metadata=None):
        self.subject, self.protocol = subject, protocol
        self.tools = tuple(tools)
        self.output_dir = Path(output_dir)
        self.instrumentation = instrumentation
        self.metadata = dict(metadata or {})

    def validate(self):
        if self.output_dir.exists():
            raise FileExistsError(self.output_dir)
        self.protocol.validate(self.subject)
        names = [tool.name for tool in self.tools]
        if len(set(names)) != len(names):
            raise ValueError('Tool names must be unique')
        phases = {'observe': 0, 'before_intervention': 1, 'intervene': 2,
                  'after_intervention': 3}
        previous = -1
        for tool in self.tools:
            if tool.phase not in phases or phases[tool.phase] < previous:
                raise ValueError('Order tools: observers, before-intervention recordings, '
                                 'interventions, after-intervention recordings')
            previous = phases[tool.phase]
            tool.validate(self)
        plan = {'subject': getattr(self.subject, 'identifier', _identity(self.subject)),
                'subject_type': _identity(self.subject),
                'protocol': {'identifier': self.protocol.identifier,
                             'implementation': _identity(self.protocol),
                             'config': self.protocol.describe()},
                'tools': [{'name': t.name, 'implementation': _identity(t),
                           'phase': t.phase, 'config': t.describe()} for t in self.tools],
                'instrumentation': (self.instrumentation.describe()
                                    if self.instrumentation else None),
                'metadata': self.metadata}
        json.dumps(plan, allow_nan=False)
        return plan

    def run(self):
        plan = self.validate()
        self.output_dir.mkdir(parents=True, exist_ok=False)
        manifest = {'schema_version': 1, 'run_id': uuid.uuid4().hex, 'status': 'running',
                    'started_utc': datetime.now(timezone.utc).isoformat(),
                    'plan': plan, 'artifacts': []}
        _write(self.output_dir / 'experiment.json', manifest)
        context = RunContext(self, manifest)
        try:
            with ExitStack() as stack:
                for tool in self.tools:
                    if tool.phase == 'observe':
                        stack.enter_context(tool.attach(context))
                value = self.protocol.run(self.subject, context)
                if context.failed:
                    raise RuntimeError('Experiment encountered a caught model or tool error')
                encoded = context.codec.encode(value)
                _write(self.output_dir / 'result.json', encoded)
                context.artifact(self.output_dir / 'result.json', producer='protocol',
                                 description='Protocol return value, RunRecord payload encoding')
            manifest['status'] = 'complete'
        except BaseException as error:
            context.failed = True
            manifest['status'] = 'failed'
            manifest['error'] = {'type': type(error).__name__, 'message': str(error)}
            raise
        finally:
            manifest['finished_utc'] = datetime.now(timezone.utc).isoformat()
            manifest['event_count'] = context.sequence
            _write(self.output_dir / 'experiment.json', manifest)
        return ExperimentResult(self.output_dir, value)
