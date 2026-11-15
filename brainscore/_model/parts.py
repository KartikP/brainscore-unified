"""The parts a session used, noted where each part is called, at the time of the call.

``BrainScoreModel.participate`` opens one log per session (``session_log``). Each call
site notes what it called and for which requested outputs: every extractor call
(``dispatch.call_extractor``; a shared or fused pass included), a live recording
(``recording.open_recordings``), the readout fit and each readout pass, the answer, the
reply, the likelihood and the policy. The outputs a call serves come from the code that
runs it (``serving``), never from a second choice after the run. At the end of the
session the model turns the log into its provenance entry (``model_helpers.provenance``).

A note costs one dictionary update per call (a set union of the call's layer names);
the facts of a part (its settings fingerprints, where its parameters are) are read once
per session, when the entry is made. Outside a session, or for a session that fails, no
log is kept or read.
"""
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Dict, Iterable, Iterator, Optional, Tuple

_log: ContextVar[Optional['PartLog']] = ContextVar('brainscore_part_log', default=None)
_serving: ContextVar[Tuple[str, ...]] = ContextVar('brainscore_part_serving', default=())

# The parts that answer a behavior format: a conversion of their answer is noted on them.
ANSWERING = ('answer', 'reply', 'likelihood', 'readout', 'policy')


class PartLog:
    """The calls of one session: ``(part, id(object), serves)`` -> its record."""

    def __init__(self) -> None:
        self.records: Dict[tuple, dict] = {}
        self.objects: Dict[int, object] = {}   # keeps each noted object for the facts at the end
        self.settings: Dict[int, set] = {}     # a caching wrapper's settings fingerprints, per call

    def note(self, part: str, obj, layers: Iterable[str] = (), serves: Optional[Iterable[str]] = None) -> None:
        serves = tuple(sorted(set(_serving.get() if serves is None else serves)))
        key = (part, id(obj), serves)
        record = self.records.get(key)
        if record is None:
            record = self.records[key] = {'part': part, 'object': id(obj), 'serves': serves,
                                          'layers': set(), 'conversion': {}}
            self.objects[id(obj)] = obj
        record['layers'].update(layers)

    def conversion(self, channel: str, conversion: Optional[str]) -> None:
        """The conversion applied to ``channel``'s answer, on the answering part that served it."""
        if conversion is None:
            return
        for record in self.records.values():
            if record['part'] in ANSWERING and channel in record['serves']:
                record['conversion'][channel] = conversion


def active() -> Optional[PartLog]:
    """The log of the running session, or None."""
    return _log.get()


@contextmanager
def session_log() -> Iterator[PartLog]:
    """A fresh log for one session; the previous one (a nested session) comes back after it."""
    token, serving = _log.set(PartLog()), _serving.set(())
    try:
        yield _log.get()
    finally:
        _serving.reset(serving)
        _log.reset(token)


@contextmanager
def serving(channels: Iterable[str]) -> Iterator[None]:
    """The requested outputs that the calls inside the block serve."""
    token = _serving.set(tuple(channels))
    try:
        yield
    finally:
        _serving.reset(token)


@contextmanager
def unnoted() -> Iterator[None]:
    """Calls inside the block are not noted (a shared pass whose requests note themselves)."""
    token = _log.set(None)
    try:
        yield
    finally:
        _log.reset(token)


def note(part: str, obj, layers: Iterable[str] = (), serves: Optional[Iterable[str]] = None) -> None:
    """Note one call of ``obj`` as ``part`` in the running session's log (nothing outside a session)."""
    log = _log.get()
    if log is not None:
        log.note(part, obj, layers, serves)


def serving_now() -> Tuple[str, ...]:
    return _serving.get()
