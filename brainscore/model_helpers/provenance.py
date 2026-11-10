"""Attach model provenance at the response boundary without changing values."""
from copy import deepcopy
from dataclasses import replace
from typing import Mapping, Any

from brainscore_core.contract import Subject
from brainscore_core.streaming import Session


def _attach(provenance, event):
    from xarray import DataArray

    payload = event.payload
    if isinstance(payload, DataArray):
        payload = payload.assign_attrs(provenance=deepcopy(provenance))
    return replace(event, payload=payload,
                   meta={**event.meta, 'provenance': deepcopy(provenance)})


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
        with session.rewriting_responses(lambda event: _attach(self._provenance, event)):
            self._subject.participate(session)

    def reset(self) -> None:
        self._subject.reset()
