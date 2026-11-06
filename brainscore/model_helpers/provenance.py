"""Attach model provenance at the response boundary without changing values."""
from copy import deepcopy
from dataclasses import replace
from typing import Mapping, Any

from brainscore_core.contract import Subject
from brainscore_core.streaming import Session


class _ProvenanceSession:
    def __init__(self, session, provenance):
        self._session, self._provenance = session, provenance

    def __getattr__(self, name):
        return getattr(self._session, name)

    @property
    def complete(self):
        return self._session.complete

    @complete.setter
    def complete(self, value):
        self._session.complete = value

    def next_input(self):
        return self._session.next_input()

    def respond(self, event):
        from xarray import DataArray

        payload = event.payload
        if isinstance(payload, DataArray):
            payload = payload.assign_attrs(provenance=deepcopy(self._provenance))
        self._session.respond(replace(event, payload=payload,
            meta={**event.meta, 'provenance': deepcopy(self._provenance)}))

    emit = respond


class ProvenanceSubject(Subject):
    """A response-metadata decorator; all declarations and lifecycle delegate.

    This adds no model ability or evaluation path. Assemblies share their data
    with the original response and receive a separate attrs dictionary. Each
    response gets its own metadata copy so consumers cannot rewrite later runs.
    """

    def __init__(self, subject: Subject, provenance: Mapping[str, Any]) -> None:
        self._subject = subject
        self._provenance = deepcopy(dict(provenance))

    @property
    def identifier(self) -> str:
        return self._subject.identifier

    @property
    def in_channels(self) -> set[str]:
        return self._subject.in_channels

    @property
    def out_channels(self) -> set[str]:
        return self._subject.out_channels

    @property
    def required_channels(self) -> set[str]:
        return self._subject.required_channels

    @property
    def traits(self) -> set[str]:
        return self._subject.traits

    def participate(self, session: Session) -> None:
        self._subject.participate(_ProvenanceSession(session, self._provenance))

    def reset(self) -> None:
        self._subject.reset()
