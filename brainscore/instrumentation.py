"""Composable observation and scoped interventions for experiment tools.

Observers implement on_start(call), on_result(call, result), and/or
on_error(call, error). Call arguments are live objects; persistent recorders
must snapshot them in on_start. Observation failures propagate explicitly.
Use distinct subject instances for concurrent experiments.
"""

from contextlib import contextmanager
from dataclasses import dataclass, field
import threading
import time
import uuid


@dataclass
class Call:
    method: str
    args: tuple
    kwargs: dict
    event_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    started_ns: int = field(default_factory=time.time_ns)
    duration_s: float = 0.0


class observe:
    """Attach independent observers to public evaluation calls for a with block.

    Nested public calls on the same subject count once per observer. Restores
    the exact prior instance attributes, including another tool's wrappers.
    Contexts must be exited in reverse attachment order.
    """

    def __init__(self, subject, *observers, methods=('process', 'look_at', 'digest_text')):
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


@contextmanager
def intervene(subject, state_change):
    """Apply one intervention and remove only its handle, including on error."""
    from brainscore_core.events import StateChange
    applied = subject.process(state_change)
    try:
        yield applied
    finally:
        subject.process(StateChange(kind='reset', handle_id=applied.handle_id))


class ObservedSession:
    """Record session inputs and outputs without changing the session protocol.

    The sink implements record(direction, event). Pass this proxy to interact().
    Its properties and collect() are delegated to the original session.
    """

    def __init__(self, session, sink):
        self.session, self.sink = session, sink

    def __getattr__(self, name):
        return getattr(self.session, name)

    def next_input(self):
        event = self.session.next_input()
        if event is not None:
            self.sink.record('input', event)
        return event

    def emit(self, event):
        self.sink.record('output', event)
        return self.session.emit(event)
