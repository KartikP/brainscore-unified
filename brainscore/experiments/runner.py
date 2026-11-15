"""Synchronous, single-subject experiment lifecycle and artifact provenance."""
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import threading
import time
import uuid
from typing import Iterator

from brainscore.instrumentation import Call, observed
from brainscore.run_record import PayloadCodec, RunRecord
from brainscore.experiments._storage import check_output_directory
from brainscore_core.compatibility import check_channel_compatibility
from brainscore_core.contract import Subject


# These data fields affect built-in session evaluation. Never serialize a live
# environment, iterator, callback, or clock when recording a replay request.
from brainscore_core.streaming import Session, StreamEvent

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
    requested_output_channels = frozenset()

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


def participate_observed(subject, session, context):
    """Run ``subject`` in ``session`` inside the current record repetition.

    The session's settings are published first and its input and output events
    are recorded between the repetition's boundaries, so ``replay_sessions``
    rebuilds exactly this session. Every protocol that runs a session uses this
    one block: ``SessionProtocol`` and the adapters for external libraries.
    """
    settings = {name: getattr(session, name) for name in _SESSION_CONTEXT_FIELDS
                if getattr(session, name, None) is not None}
    if settings:
        context.publish('session_context', settings)
    with observed(session, context):
        subject.participate(session)


def _publish_finalized_answers(session, context, collected=None):
    """Publish the stored answer of each behavior format of a completed session.

    While the runner owns the completed session, each addressed ``behavior:<format>``
    answer is collected once (``collected``, when the protocol collected the outputs)
    and published as one ``answer`` event, with the condition and repetition of the
    record. It holds core's stored form: each trial's value, state, reason and raw
    answer, one row per presented trial (core adds a missing row for a trial with no
    answer). Its meta carries the presented trials in presentation order with their
    stimulus columns (``trials``), so ``ExperimentResult.trials()`` builds one row per
    trial and occurrence from it. ``RecordBehavior`` keeps it. A session whose history
    limit dropped responses cannot be collected: it publishes none.
    """
    if session.dropped:
        return
    trials = _presented_trials(session)
    channels = [channel for channel in (session.outputs or ()) if channel.startswith('behavior:')]
    for channel in channels:
        if not any(event.channel == channel for event in session.emitted):
            continue
        if isinstance(collected, dict):
            answer = collected[channel]
        elif collected is not None and len(session.outputs) == 1:
            answer = collected
        else:
            answer = session.collect(channel)
        meta = {'finalized': True, **({'trials': trials} if trials else {})}
        context.publish('answer', StreamEvent(channel, answer, None, meta))


def _presented_trials(session):
    """The session's table in presentation order: ``stimulus_id`` and every stimulus
    column of plain scalar values (text, numbers, booleans, None), as lists; ``None``
    without a table. A numpy scalar becomes its Python value; a column with any other
    value (an array, a list) is left out, unconverted."""
    import numbers
    import numpy as np
    table = getattr(session, 'stimuli', None)
    if table is None or 'stimulus_id' not in getattr(table, 'columns', ()):
        return None

    def plain(value):
        return value is None or isinstance(value, (str, bool, numbers.Number))
    trials = {}
    for column in table.columns:
        values = [value.item() if isinstance(value, np.generic) else value
                  for value in table[column].tolist()]
        if column == 'stimulus_id' or all(plain(value) for value in values):
            trials[column] = values
    return trials


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
                 required_input_channels=(), requested_output_channels=(), metadata=None,
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
        self.required_input_channels = frozenset(required_input_channels)
        self.requested_output_channels = frozenset(requested_output_channels)
        self.metadata = dict(metadata or {})
        self._compatibility = None

    def accept_requests(self, channels):
        """Fail when attached recorders need outputs these sessions do not request."""
        missing = set(channels) - set(self.requested_output_channels)
        if missing:
            raise ValueError(f'Recorders need {sorted(missing)}, but this protocol requests only '
                             f'{sorted(self.requested_output_channels)}. Use TaskProtocol, or request them.')

    def describe(self):
        return {'conditions': self.conditions, 'repetitions': self.repetitions,
                'required_input_channels': sorted(self.required_input_channels),
                'requested_output_channels': sorted(self.requested_output_channels),
                'metadata': self.metadata,
                'factory_argument': self.factory_argument,
                'reset_policy': 'before and after each repetition; tools attach after initial reset',
                **({'compatibility': self._compatibility.as_attrs()}
                   if self._compatibility is not None else {}),
                **({} if self.collect_outputs else {'collect_outputs': False})}

    def validate(self, subject):
        """Check the subject against this protocol with the rule ``score()`` uses.

        The protocol is the benchmark side: it declares the inputs it sends and
        the outputs it requests, and core ``check_channel_compatibility``
        decides the match (Contract Rule 5.2), so an experiment and a benchmark
        agree on which pairs fit. An incompatible pair raises
        ``CompatibilityError``; a channel name outside the catalog must be
        registered with ``register_channel`` first.
        """
        self._compatibility = None
        if not isinstance(subject, Subject):
            raise TypeError('SessionProtocol requires a Subject (participate, reset and channels)')
        if (not self.conditions or any(not isinstance(c, str) or not c for c in self.conditions)
                or len(set(self.conditions)) != len(self.conditions)):
            raise ValueError('Use nonempty, distinct condition names')
        if not self.repetitions or len(set(self.repetitions)) != len(self.repetitions):
            raise ValueError('Use nonempty, distinct, hashable repetition IDs')
        self._compatibility = check_channel_compatibility(subject, self)

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
                        participate_observed(subject, session, context)
                        if not session.complete:
                            raise RuntimeError('Environment session ended before episode completion')
                        collected = session.collect() if self.collect_outputs else None
                        _publish_finalized_answers(session, context, collected)
                        if self.collect_outputs:
                            results.append(collected)
                except BaseException as error:
                    try:
                        subject.reset()
                    except Exception as reset_error:      # the first error stays the one raised
                        error.add_note(f'The reset of the subject also failed: {reset_error!r}')
                    raise
                else:
                    subject.reset()
        return results if self.collect_outputs else None


class _MethodCalls:
    """Record the named method calls of an external system for a with block.

    Only ``CallableProtocol(methods=[...])`` uses this, for a system that is not
    a subject (for example a policy server's ``infer``). The caller names the
    methods; there is no default. Nested calls on the same object count once.
    Restores the exact prior instance attributes on exit.
    """

    def __init__(self, subject, *observers, methods):
        self.subject = subject
        self.observers = observers
        self.methods = methods
        self._saved = {}
        self._local = threading.local()

    def _notify(self, name, *args):
        for observer in self.observers:
            callback = getattr(observer, name, None)
            if callback is not None:
                callback(*args)

    def _wrap(self, method, original):
        def wrapped(*args, **kwargs):
            if getattr(self._local, 'active', False):
                return original(*args, **kwargs)
            self._local.active = True
            call = Call(method, args, kwargs)
            start = time.perf_counter()
            try:
                self._notify('on_start', call)
                try:
                    result = original(*args, **kwargs)
                except BaseException as error:
                    call.duration_s = time.perf_counter() - start
                    self._notify('on_error', call, error)
                    raise
                call.duration_s = time.perf_counter() - start
                self._notify('on_result', call, result)
                return result
            finally:
                self._local.active = False
        return wrapped

    def __enter__(self):
        if self._saved:
            raise RuntimeError('An observer context cannot be entered twice')
        try:
            for method in self.methods:
                original = getattr(self.subject, method, None)
                if not callable(original):
                    continue
                state = vars(self.subject)
                self._saved[method] = (method in state, state.get(method))
                setattr(self.subject, method, self._wrap(method, original))
        except BaseException:
            self.__exit__(None, None, None)
            raise
        return self

    def __exit__(self, *exc):
        for method, (present, previous) in reversed(list(self._saved.items())):
            if present:
                setattr(self.subject, method, previous)
            else:
                delattr(self.subject, method)
        self._saved.clear()
        return False



class CallableProtocol:
    """Preserve an external evaluator's loop and scientific scoring.

    evaluate(subject, context) owns reset/seed/repetition semantics.
    Instrumentation is scoped to the whole evaluator call. Use separate runs
    for baseline and intervention conditions.

    The evaluator runs the subject through ``participate``. It owns the
    repetitions too: it runs each session with
    ``brainscore.harnesses.adapter.run_request`` (or ``participate_observed``
    inside ``context.repetition(identifier)``), so the record holds the input
    and output events of each session between its own boundaries. It names those
    identifiers in ``repetitions``, so tools can select them.

    For a system that is not a subject (a policy server, an external library
    object), ``methods`` names the methods to record, for example
    ``methods=['infer']``; the one repetition is then ``'external'``. There is
    no default method: a subject is watched at its sessions.
    """
    def __init__(self, identifier, evaluate, *, methods=(), metadata=None,
                 repetitions=None):
        self.identifier, self.evaluate = identifier, evaluate
        if isinstance(methods, str):
            raise TypeError('methods is a list of method names, for example methods=[\'infer\']')
        self.methods = tuple(methods)
        self.metadata = dict(metadata or {})
        if self.methods:
            if repetitions is not None:
                raise ValueError("An evaluator with observed methods runs one 'external' "
                                 'repetition; it cannot name others')
            self.repetitions = ('external',)
        else:
            self.repetitions = tuple(repetitions or ())

    def accept_requests(self, channels):
        if channels:
            raise ValueError('CallableProtocol runs an evaluator with its own sessions, so recorders '
                             f'cannot request {sorted(channels)}. Use RecordInputsOutputs or RecordActivity.')

    def describe(self):
        described = {'methods': self.methods, 'metadata': self.metadata,
                     'reset_policy': 'owned by external evaluator'}
        if not self.methods:
            described['repetitions'] = self.repetitions
        return described

    def validate(self, subject):
        # An empty selection is allowed: such an evaluator records its sessions.
        if len(set(self.methods)) != len(self.methods):
            raise ValueError('Select distinct methods to observe')
        # The external evaluator names the methods it calls, so checking for them is
        # checking its request, not discovering what the subject can do.
        for method in self.methods:
            if not callable(getattr(subject, method, None)):
                raise ValueError(f'Subject does not expose {method}')

    def run(self, subject, context):
        if not self.methods:
            # The evaluator records sessions, so it also marks where each one
            # starts and ends: one context.repetition per session, which
            # replay_sessions turns back into one session each.
            return self.evaluate(subject, context)
        with context.repetition('external'), _MethodCalls(subject, context, methods=self.methods):
            return self.evaluate(subject, context)


def _per_trial_coords(assembly):
    """Every per-presentation coordinate, including levels of a presentation MultiIndex
    (where this xarray version keeps the stimulus set's columns)."""
    import pandas as pd
    found = {}
    index = assembly.indexes.get('presentation')
    if isinstance(index, pd.MultiIndex):
        found.update({name: index.get_level_values(name).values for name in index.names})
    found.update({name: coord.values for name, coord in assembly.coords.items()
                  if coord.dims == ('presentation',) and name != 'presentation'})
    found.pop('stimulus_id', None)
    return found


# How each answer was read, by the coordinate core stores, and its column suffix.
_DETAILS = {'response_state': 'state', 'response_reason': 'reason', 'raw_answer': 'raw',
            'answer_valid': 'valid', 'parse_error': 'error'}


def _occurrences(stimulus_ids):
    """``(stimulus_id, k)`` for each row: its k-th occurrence, the session's matching rule."""
    seen, keys = {}, []
    for stimulus_id in stimulus_ids:
        keys.append((stimulus_id, seen.get(stimulus_id, 0)))
        seen[stimulus_id] = seen[stimulus_id] + 1 if stimulus_id in seen else 1
    return keys


def _rows_of(events):
    """Each answer row of ``events`` (one format, in order) with its key ``(stimulus_id, k)``:
    ``(key, assembly, row)`` for an assembly row, ``(key, event, None)`` for a one-cell event."""
    found = []
    for event in events:
        payload = event.payload
        if hasattr(payload, 'dims'):
            found += [(stimulus_id, payload, row)
                      for row, stimulus_id in enumerate(payload['stimulus_id'].values)]
        else:
            found.append((event.meta.get('stimulus_id'), event, None))
    keys = _occurrences([stimulus_id for stimulus_id, _, _ in found])
    return [(key, source, row) for key, (_, source, row) in zip(keys, found)]


def _answer_columns(channel, rows, keys, columns):
    """Add ``channel``'s value and detail columns, one value per table key; ``rows`` are
    its answer rows (``_rows_of``), matched to the keys by stimulus and occurrence."""
    import numpy as np
    by_key = {key: (source, row) for key, source, row in rows}
    # A bare behavior answer (a legacy task) keeps its earlier name: 'choice' when it
    # has a choice axis, else 'answer'.
    fmt = (channel.split(':', 1)[1] if ':' in channel else
           'choice' if any(row is not None and 'choice' in source.dims for _, source, row in rows)
           else 'answer')
    picked = [by_key.get(key) for key in keys]
    per_trial = {}
    for source, row in filter(None, picked):
        if row is not None and id(source) not in per_trial:
            per_trial[id(source)] = {name: list(values) for name, values in _per_trial_coords(source).items()}

    def value(item):
        if item is None:
            return None
        source, row = item
        if row is None:                         # a one-cell event: the cell as sent
            return source.payload
        values = np.asarray(source.values)
        if 'choice' in source.dims and fmt not in ('probabilities', 'log_probabilities'):
            labels = list(source['choice'].values)
            return labels[int(np.argmax(values[row]))] if np.isfinite(values[row]).all() else None
        return values[row]
    if any(item is not None and item[1] is not None and 'choice' in item[0].dims
           for item in picked) and fmt in ('probabilities', 'log_probabilities'):
        prefix = 'p' if fmt == 'probabilities' else 'logp'
        labels = next(list(item[0]['choice'].values) for item in picked if item and item[1] is not None)
        for i, label in enumerate(labels):
            columns[f'{prefix}({label})'] = [float(value(item)[i]) if item is not None else np.nan
                                            for item in picked]
    else:
        columns[fmt] = [value(item) for item in picked]
    for name, suffix in _DETAILS.items():
        if any(name in coords for coords in per_trial.values()):
            columns[f'{fmt}_{suffix}'] = [
                per_trial[id(item[0])].get(name, [None] * (item[1] + 1))[item[1]]
                if item is not None and item[1] is not None else None for item in picked]


def _stimulus_columns(rows, keys, columns):
    """Stimulus columns of the answer rows (the first row that has a key gives its values)."""
    found = {}
    for key, source, row in rows:
        if row is None:
            continue
        for name, values in _per_trial_coords(source).items():
            if name not in _DETAILS:
                found.setdefault(name, {}).setdefault(key, list(values)[row])
    for name, by_key in found.items():
        columns.setdefault(name, [by_key.get(key) for key in keys])


def _finalized_table(envelopes):
    """One session's table from its finalized answers: one row per presented trial."""
    trials = next((env['payload'].meta.get('trials') for env in envelopes
                   if env['payload'].meta.get('trials')), None)
    first = envelopes[0]['payload'].payload
    order = list(trials['stimulus_id']) if trials else list(first['stimulus_id'].values)
    keys = _occurrences(order)
    columns = {'stimulus_id': order}
    if trials:
        columns.update({name: list(values) for name, values in trials.items() if name != 'stimulus_id'})
    else:
        _stimulus_columns(_rows_of([envelopes[0]['payload']]), keys, columns)
    for env in envelopes:
        _answer_columns(env['payload'].channel, _rows_of([env['payload']]), keys, columns)
    return columns


def _observed_table(envelopes):
    """One session's table from the observed answer events (the fallback): every observed
    answer row is a table row, in the order it was observed; the formats are matched by
    stimulus and occurrence as in a completed session. A trial never observed has no row."""
    named = {env['payload'].channel for env in envelopes} - {'behavior'}
    by_channel = {}
    for env in envelopes:
        channel = env['payload'].channel
        if channel == 'behavior' and named:
            continue               # the main answer is also saved under its own name
        by_channel.setdefault(channel, []).append(env['payload'])
    rows = {channel: _rows_of(events) for channel, events in by_channel.items()}
    keys = list(dict.fromkeys(key for found in rows.values() for key, _, _ in found))
    columns = {'stimulus_id': [stimulus_id for stimulus_id, _ in keys]}
    for found in rows.values():
        _stimulus_columns(found, keys, columns)
    for channel, found in rows.items():
        _answer_columns(channel, found, keys, columns)
    return columns


@dataclass(frozen=True)
class ExperimentResult:
    """What a run saved, read back from ``directory``.

    A run made without ``output_dir`` lives in a temporary folder that is deleted
    when this result is no longer used; ``save(path)`` keeps a copy.
    """
    directory: Path
    value: object
    _temporary: object = None          # the TemporaryDirectory that holds a run without output_dir

    @classmethod
    def open(cls, path) -> 'ExperimentResult':
        """Reopen a run saved earlier (an ``output_dir`` or a ``result.save(path)`` copy),
        with the same ``behavior``, ``neural`` and ``trials`` readers."""
        directory = Path(path)
        if not (directory / 'experiment.json').is_file():
            raise FileNotFoundError(f'{directory} is not a saved experiment run (no experiment.json).')
        saved = directory / 'result.json'
        value = PayloadCodec(directory).decode(json.loads(saved.read_text())) if saved.is_file() else None
        return cls(directory, value)

    @property
    def temporary(self) -> bool:
        """Whether the run is deleted with this result unless saved."""
        return self._temporary is not None

    def save(self, path) -> 'ExperimentResult':
        """Copy the run to ``path`` (which must not exist yet) and return the result there."""
        destination = Path(path)
        check_output_directory(destination)
        shutil.copytree(self.directory, destination)
        return ExperimentResult(destination, self.value)

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
        ``'rating'``, ``'rt'`` or ``'text'``. By default, the main answer.
        """
        if measure is not None:
            return self._saved('behavior', f'behavior:{measure}', condition, repetition)
        for channel in ('behavior', 'behavior:choice', 'behavior:rating', 'behavior:text'):
            try:
                return self._saved('behavior', channel, condition, repetition)
            except LookupError:
                continue
        raise LookupError(f'No saved behavior for condition={condition!r}, repetition={repetition!r}.')

    def trials(self):
        """Every saved answer as a table: one row per presented trial.

        Columns: ``condition``, ``repetition``, ``stimulus_id``, the stimulus set's own
        columns (e.g. a noise level), then one column per kind of answer the run saved:
        ``choice`` (the chosen label), ``p(<label>)`` for each label's probability,
        ``logp(<label>)`` for each log probability, ``rt``, ``rating``, ``text``. For each
        format, ``<answer>_state`` and ``<answer>_reason`` are the state core stored for
        the trial (``valid``, ``invalid``, ``missing``) and its reason;
        ``<answer>_raw``, ``<answer>_valid`` and ``<answer>_error`` are the raw response,
        whether the model side could read it, and why not, when the answer has them. A
        trial with no usable answer has an empty value (``None`` or NaN).

        For a completed session the rows come from the finalized answers the runner
        saves: one row for each presented trial and occurrence, in presentation order,
        each format matched by stimulus identifier and occurrence. A session with no
        finalized answer (one that ended with an error, a run saved before finalized
        answers existed, or a history limit that dropped responses) falls back to the
        observed answer events, one table per event as before. The column ``rows_from``
        says which: ``'finalized answers'`` or ``'observed events'``.
        """
        import pandas as pd
        path = self.directory / 'behavior'
        if not path.is_dir():
            raise FileNotFoundError('No behavior record: attach RecordBehavior to the experiment.')
        envelopes = [row['payload'] for row in RunRecord(path).events()]
        keys, outputs, finalized = [], {}, {}
        for env in envelopes:
            if env['kind'] not in ('output', 'answer'):
                continue
            key = (env['condition'], env['repetition_id'])
            if key not in keys:
                keys.append(key)
            (finalized if env['kind'] == 'answer' else outputs).setdefault(key, []).append(env)
        frames = []
        for condition, repetition in keys:
            key = (condition, repetition)
            if key in finalized:
                columns, source = _finalized_table(finalized[key]), 'finalized answers'
            else:
                columns, source = _observed_table(outputs[key]), 'observed events'
            frames.append(pd.DataFrame(columns).assign(condition=condition, repetition=repetition,
                                                       rows_from=source))
        if not frames:
            raise LookupError('The behavior record holds no answers.')
        table = pd.concat(frames, ignore_index=True)
        first = ['condition', 'repetition', 'stimulus_id']
        last = ['rows_from']
        return table[first + [c for c in table.columns if c not in first + last] + last]

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
        self.directory = experiment.run_directory
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

    With ``output_dir``, the run is written there; the folder must not exist yet.
    Without it, each run is written to a temporary folder that its result keeps
    until the result is no longer used; call ``result.save(path)`` to keep it.

    Protocols and tools are ordinary Python objects and can live in external
    packages. No concurrent reuse of a subject, tool, or provider is supported.
    """
    def __init__(self, *, subject, protocol, tools=(), output_dir=None,
                 instrumentation=None, metadata=None):
        self.subject, self.protocol = subject, protocol
        self.tools = tuple(tools)
        self.output_dir = Path(output_dir) if output_dir is not None else None
        self.run_directory = self.output_dir      # where the current run is written
        self._temporary = None
        self.instrumentation = instrumentation
        self.metadata = dict(metadata or {})

    def validate(self):
        if self.output_dir is not None:
            check_output_directory(self.output_dir)
        requested = set().union(*(tool.requested_output_channels for tool in self.tools))
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
                'subject_traits': sorted(getattr(self.subject, 'traits', ())),
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
        if self.output_dir is None:
            # A fresh temporary folder per run, so the same Experiment can run again.
            self._temporary = tempfile.TemporaryDirectory(prefix='umi-run-')
            self.run_directory = Path(self._temporary.name) / 'run'
        self.run_directory.mkdir(parents=True, exist_ok=False)
        manifest = {'schema_version': 1, 'run_id': uuid.uuid4().hex, 'status': 'running',
                    'started_utc': datetime.now(timezone.utc).isoformat(),
                    'plan': plan, 'artifacts': []}
        _write(self.run_directory / 'experiment.json', manifest)
        context = RunContext(self, manifest)
        try:
            with ExitStack() as stack:
                for tool in self.tools:
                    if tool.phase == 'observe':
                        stack.enter_context(tool.attach(context))
                yield context
                if context.failed:
                    raise RuntimeError('Experiment encountered a caught subject or tool error')
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
            _write(self.run_directory / 'experiment.json', manifest)

    def run(self):
        with self.open() as context:
            value = self.protocol.run(self.subject, context)
            if context.failed:
                raise RuntimeError('Experiment encountered a caught subject or tool error')
            encoded = context.codec.encode(value)
            _write(self.run_directory / 'result.json', encoded)
            context.artifact(self.run_directory / 'result.json', producer='protocol',
                             description='Protocol return value, RunRecord payload encoding')
        return ExperimentResult(self.run_directory, value, self._temporary)
