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
from typing import Iterator

from brainscore.instrumentation import ObservedSession, observe
from brainscore.run_record import PayloadCodec, RunRecord
from brainscore.experiments._storage import check_output_directory
from brainscore_core.compatibility import CompatibilityMatch, validate_channels
from brainscore_core.contract import Subject


# These data fields affect built-in session evaluation. Never serialize a live
# environment, iterator, callback, or clock when recording a replay request.
from brainscore_core.streaming import Session

_SESSION_CONTEXT_FIELDS = Session.context_fields


def _identity(value):
    cls = type(value)
    return f'{cls.__module__}.{cls.__qualname__}'


_ERROR_CHAIN_LIMIT = 50


def _message(error):
    # Only called while a failure is already propagating. BaseException is
    # deliberate: an interrupt raised by a broken __str__ must not replace it.
    try:
        return str(error)
    except BaseException:
        return f'<unprintable {type(error).__name__}>'


def _error_chain(error, limit=_ERROR_CHAIN_LIMIT):
    """Flatten an exception graph, outermost first, without recursion.

    Each entry names its parent entry and how it relates: 'cause' (raise ...
    from), 'context' (raised while handling, unless suppressed) or 'group'
    (a member of an ExceptionGroup). An exception reached twice (a cycle or a
    shared child) is recorded once more with 'same_as' and not expanded. An
    entry whose links were not followed because of the limit has 'truncated'.
    """
    chain, seen, pending = [], {}, [(error, None, None)]
    while pending:
        current, parent, relation = pending.pop()
        if len(chain) >= limit:
            chain[parent]['truncated'] = True
            continue
        entry = {'type': type(current).__name__, 'message': _message(current),
                 'parent': parent, 'relation': relation}
        chain.append(entry)
        if id(current) in seen:
            entry['same_as'] = seen[id(current)]
            continue
        seen[id(current)] = index = len(chain) - 1
        links = []
        if current.__cause__ is not None:
            links.append((current.__cause__, 'cause'))
        elif current.__context__ is not None and not current.__suppress_context__:
            links.append((current.__context__, 'context'))
        if isinstance(current, BaseExceptionGroup):
            members = current.exceptions
            links.extend((member, 'group') for member in members[:limit])
            if len(members) > limit:
                entry['truncated'] = True
        # Stack is last-in first-out: push in reverse so links are visited in order.
        pending.extend((child, index, kind) for child, kind in reversed(links))
    return chain


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
    # Outputs this tool asks the subject for; a protocol that builds its own
    # sessions (TaskProtocol) requests them.
    output_channels = frozenset()

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
class Repetition:
    """One independent run of a condition: one session, between two resets.

    A protocol runs each condition once per repetition. Each repetition is one
    session (a visit), which may hold many trials (stimuli, each perhaps with
    several parts, then a response) or one continuous stream with no trials.
    The subject is reset before and after it, so nothing carries over between
    repetitions; within a session, memory follows the subject's traits.
    """
    condition: str
    identifier: object


class SessionProtocol:
    """Factory creates a fresh context-managed session for each independent repetition.

    The protocol owns repetition progression and resets. The factory owns session
    resource cleanup. Channel declarations describe the protocol, not a modality.

    The factory always receives Repetition(condition, identifier). Adding conditions
    does not change its argument. For a factory written to accept only an ID,
    set factory_argument='identifier' explicitly.
    Set collect_outputs=False to return None instead of collecting session history.
    Attach a recording tool if the episode's events must be kept.

    """
    def __init__(self, identifier, session_factory, *, conditions=None, repetitions=(0,),
                 input_channels=(), output_channels=(), metadata=None,
                 factory_argument='repetition', collect_outputs: bool = True):
        if not isinstance(collect_outputs, bool):
            raise TypeError('collect_outputs must be a bool')
        self.collect_outputs = collect_outputs
        self.identifier = identifier
        self.session_factory = session_factory
        self.repetitions = tuple(repetitions)
        self.conditions = tuple(conditions) if conditions is not None else ('default',)
        if factory_argument not in ('repetition', 'identifier'):
            raise ValueError("factory_argument must be 'repetition' or 'identifier'.")
        self.factory_argument = factory_argument
        self.input_channels = frozenset(input_channels)
        self.output_channels = frozenset(output_channels)
        self.metadata = dict(metadata or {})
        self._compatibility = None

    def accept_requests(self, channels):
        """Fail when attached recorders need outputs these sessions do not request."""
        missing = set(channels) - set(self.output_channels)
        if missing:
            raise ValueError(f'Recorders need {sorted(missing)}, but this protocol requests only '
                             f'{sorted(self.output_channels)}. Use TaskProtocol, or request them.')

    def describe(self):
        return {'conditions': self.conditions, 'repetitions': self.repetitions, 'input_channels': sorted(self.input_channels),
                'output_channels': sorted(self.output_channels), 'metadata': self.metadata,
                'factory_argument': self.factory_argument,
                'reset_policy': 'before and after each repetition; tools attach after initial reset',
                **({'compatibility': self._compatibility.as_attrs()}
                   if self._compatibility is not None else {}),
                **({} if self.collect_outputs else {'collect_outputs': False})}

    def validate(self, subject):
        self._compatibility = None
        if not isinstance(subject, Subject):
            raise TypeError('SessionProtocol requires a Subject (participate, reset and channels)')
        # Use scoring's channel rules; exploratory names remain usable with a warning.
        for channels, direction, owner in (
            (subject.in_channels, 'input', 'subject inputs'),
            (subject.out_channels, 'output', 'subject outputs'),
            (subject.required_channels, 'input', 'subject required inputs'),
            (self.input_channels, 'input', 'protocol inputs'),
            (self.output_channels, 'output', 'protocol outputs'),
        ):
            validate_channels(set(channels), direction, owner, allow_unregistered=True)
        if (not self.conditions or any(not isinstance(c, str) or not c for c in self.conditions)
                or len(set(self.conditions)) != len(self.conditions)):
            raise ValueError('Use nonempty, distinct condition names')
        if not self.repetitions or len(set(self.repetitions)) != len(self.repetitions):
            raise ValueError('Use nonempty, distinct, hashable repetition IDs')
        consumed = self.input_channels & set(subject.in_channels)
        missing_required = set(subject.required_channels) - self.input_channels
        missing_outputs = self.output_channels - set(subject.out_channels)
        problems = []
        if self.input_channels and not consumed:
            problems.append(f'takes in none of the protocol inputs {sorted(self.input_channels)}')
        if missing_required:
            problems.append(f'requires inputs absent from the protocol {sorted(missing_required)}')
        if missing_outputs:
            problems.append(f'cannot produce requested outputs {sorted(missing_outputs)}')
        if problems:
            raise ValueError(f'Subject {subject.identifier!r} is incompatible with protocol '
                             f'{self.identifier!r}: {"; ".join(problems)}. '
                             f'Subject in_channels={sorted(subject.in_channels)}, '
                             f'out_channels={sorted(subject.out_channels)}.')
        self._compatibility = CompatibilityMatch(
            subject=subject.identifier, benchmark=self.identifier,
            consumed_inputs=frozenset(consumed),
            dropped_inputs=frozenset(self.input_channels - set(subject.in_channels)),
            requested_outputs=frozenset(self.output_channels))

    def run(self, subject, context):
        results = []
        for condition in self.conditions:
            for repetition in self.repetitions:
                request = repetition if self.factory_argument == 'identifier' else Repetition(condition, repetition)
                try:
                    subject.reset()
                    with context.repetition(repetition, condition=condition), self.session_factory(request) as session:
                        if not isinstance(session, Session):
                            raise TypeError('Session factories must return a Session')
                        settings = {
                            name: getattr(session, name)
                            for name in _SESSION_CONTEXT_FIELDS
                            if getattr(session, name, None) is not None
                        }
                        if settings:
                            context.publish('session_context', settings)
                        subject.participate(ObservedSession(session, context))
                        if not session.complete:
                            raise RuntimeError('Environment session ended before episode completion')
                        if self.collect_outputs:
                            results.append(session.collect())
                finally:
                    subject.reset()
        return results if self.collect_outputs else None


class CallableProtocol:
    """Preserve an external evaluator's loop and scientific scoring.

    evaluate(subject, context) owns reset/seed/repetition semantics. Its selected
    public methods are observed, including errors caught by the evaluator.
    Instrumentation is scoped to the whole evaluator call. Use separate runs
    for baseline and intervention conditions.
    """
    def __init__(self, identifier, evaluate, *, methods=('process',), metadata=None):
        self.identifier, self.evaluate = identifier, evaluate
        self.methods = tuple(methods)
        self.metadata = dict(metadata or {})

    def accept_requests(self, channels):
        if channels:
            raise ValueError('CallableProtocol runs an evaluator with its own sessions, so recorders '
                             f'cannot request {sorted(channels)}. Use RecordInputsOutputs or RecordActivity.')

    def describe(self):
        return {'methods': self.methods, 'metadata': self.metadata,
                'reset_policy': 'owned by external evaluator'}

    def validate(self, subject):
        if not self.methods or len(set(self.methods)) != len(self.methods):
            raise ValueError('Select distinct methods to observe')
        # The external evaluator names the methods it calls, so checking for them is
        # checking its request, not discovering what the subject can do.
        for method in self.methods:
            if not callable(getattr(subject, method, None)):
                raise ValueError(f'Subject does not expose {method}')

    def run(self, subject, context):
        with context.repetition('external'), observe(subject, context, methods=self.methods):
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

    def neural(self, region, *, condition='default', repetition=None):
        """The NeuroidAssembly a RecordNeural tool saved for ``region``.

        ``attrs['recording']`` says how it was made. A recording made during a
        task is stamped by the subject; one without a stamp was plain viewing.
        """
        assembly = self._saved('neural', f'neural:{region}', condition, repetition)
        recording = assembly.attrs.get('recording') or {'task_type': 'passive', 'path': 'extractor'}
        return assembly.assign_attrs(recording={**recording, 'condition': condition})

    def behavior(self, *, condition='default', repetition=None, measure=None):
        """The BehavioralAssembly a RecordBehavior tool saved.

        ``measure`` picks one kind of answer: ``'choice'``, ``'probabilities'``,
        ``'confidence'``, ``'rating'`` or ``'text'``. By default, the main answer.
        """
        if measure is not None:
            return self._saved('behavior', f'behavior:{measure}', condition, repetition)
        for channel in ('behavior', 'behavior:choice', 'behavior:rating', 'behavior:text'):
            try:
                return self._saved('behavior', channel, condition, repetition)
            except LookupError:
                continue
        raise LookupError(f'No saved behavior for condition={condition!r}, repetition={repetition!r}.')

    def _saved(self, folder, channel, condition, repetition):
        path = self.directory / folder
        if not path.is_dir():
            raise FileNotFoundError(f'No {folder} record: attach Record{folder.title()} to the experiment.')
        # A tool's own record keeps each experiment envelope as the row payload.
        found = [row['payload'] for row in RunRecord(path).events()]
        found = [env for env in found
                 if env['kind'] == 'output' and env['payload'].channel == channel
                 and env['condition'] == condition and (repetition is None or env['repetition_id'] == repetition)]
        if not found:
            raise LookupError(f'No saved {channel!r} output for condition={condition!r}, repetition={repetition!r}.')
        if len(found) != 1:
            raise LookupError(f'{len(found)} saved {channel!r} outputs for condition={condition!r}, '
                              f'repetition={repetition!r}; pass condition= and repetition= to pick one.')
        return found[0]['payload'].payload


class RunContext:
    def __init__(self, experiment, manifest):
        self.experiment, self.manifest = experiment, manifest
        self.directory = experiment.output_dir
        self.codec = PayloadCodec(self.directory)
        self.listeners = []
        self.repetition_id = None
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
                    'condition': self.condition, 'repetition_id': self.repetition_id, 'event_id': event_id or self.event_id,
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
    def repetition(self, identifier, *, condition='default'):
        self.condition = condition
        self.repetition_id, self.event_id = identifier, None
        self.publish('repetition_start', {'id': identifier})
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
            self.publish('repetition_end', {'failed': self.failed})
            self.repetition_id, self.event_id = None, None
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
        check_output_directory(self.output_dir)
        requested = set().union(*(tool.output_channels for tool in self.tools))
        if requested or isinstance(self.protocol, SessionProtocol):
            # Protocols from other packages need accept_requests only to take recorders.
            self.protocol.accept_requests(requested)
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

    @contextmanager
    def open(self) -> Iterator[RunContext]:
        """Attach run observers while a caller or protocol owns execution.

        Enter ``context.repetition`` to attach repetition tools. This scope never runs
        the subject or resets it; the caller retains those responsibilities.
        """
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
                yield context
                if context.failed:
                    raise RuntimeError('Experiment encountered a caught model or tool error')
            manifest['status'] = 'complete'
        except BaseException as error:
            context.failed = True
            manifest['status'] = 'failed'
            manifest['error'] = {'type': type(error).__name__, 'message': _message(error)}
            # Narrow and deliberate: this only builds diagnostics for a failure
            # that is already propagating. Anything it raises, including an
            # interrupt, is dropped so the original exception is the one raised.
            try:
                manifest['error_chain'] = _error_chain(error)
            except BaseException:
                pass
            raise
        finally:
            manifest['finished_utc'] = datetime.now(timezone.utc).isoformat()
            manifest['event_count'] = context.sequence
            _write(self.output_dir / 'experiment.json', manifest)

    def run(self):
        with self.open() as context:
            value = self.protocol.run(self.subject, context)
            if context.failed:
                raise RuntimeError('Experiment encountered a caught model or tool error')
            encoded = context.codec.encode(value)
            _write(self.output_dir / 'result.json', encoded)
            context.artifact(self.output_dir / 'result.json', producer='protocol',
                             description='Protocol return value, RunRecord payload encoding')
        return ExperimentResult(self.output_dir, value)
