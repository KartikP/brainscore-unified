"""Watch a run: the inputs and outputs of a session.

``observed(session, sink)`` is the one way to watch what a subject receives
and sends. Layers are read with ``TorchInstrumentation`` and ``RecordActivity``
(``brainscore.experiments``); an intervention is sent with
``apply_state_change`` or the ``Intervene`` tool.
"""

from contextlib import contextmanager
from dataclasses import dataclass, field
import time
import uuid


@dataclass
class Call:
    """One method call of an external system that is not a subject.

    ``CallableProtocol(methods=[...])`` records these; a subject is watched at
    its sessions with ``observed``.
    """
    method: str
    args: tuple
    kwargs: dict
    event_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    started_ns: int = field(default_factory=time.time_ns)
    duration_s: float = 0.0


@contextmanager
def observed(session, sink):
    """Record the session's inputs and outputs to ``sink.record(direction, event)``.

    Wrap the call that runs a subject: ``with observed(session, sink):
    subject.participate(session)``. Inputs are recorded as the source delivers
    them, before the subject's input filter; outputs as the subject responds.
    A closed-loop source also reports each action proposal and receipt, with its
    status; a failed action goes to ``sink.on_error`` too.
    """
    last = []

    def on_error(event, error):
        report = getattr(sink, 'on_error', None)
        if callable(report):
            report(Call('emit', (event,), {}), error)

    def on_action(event):
        last[:] = [event]
        sink.record('output', event)

    def on_output(event):
        if not (last and event is last[0]):  # a receipt is already recorded
            sink.record('output', event)

    with session.observe(lambda event: sink.record('input', event), on_output,
                         on_error, on_action):
        yield session
