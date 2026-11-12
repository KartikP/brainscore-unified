"""Explicit integration boundary between the lab and Python experiments."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Any, Callable


@dataclass(frozen=True)
class Target:
    """An explicitly exposed recording/intervention address, never discovered."""

    name: str
    layer: str
    units: int


@dataclass
class LabProfile:
    """A trusted Python factory supplies one subject and its experiment setup.

    make_session receives the existing experiment Repetition. summarize receives the
    ExperimentResult and returns a small JSON report, never activation tensors.
    The SessionProtocol owns all resets. The lab does not wrap model forwards.
    """

    subject: Any
    title: str
    description: str
    task: str
    input_channels: tuple[str, ...]
    output_channels: tuple[str, ...]
    make_session: Callable
    summarize: Callable
    targets: tuple[Target, ...] = ()
    instrumentation: Any = None
    provenance: dict | None = None

    def describe(self) -> dict:
        return {
            "identifier": self.subject.identifier,
            "title": self.title,
            "description": self.description,
            "task": self.task,
            "in_channels": sorted(self.subject.in_channels),
            "out_channels": sorted(self.subject.out_channels),
            "required_channels": sorted(self.subject.required_channels),
            "traits": sorted(self.subject.traits),
            "inputs": list(self.input_channels),
            "outputs": list(self.output_channels),
            "targets": [vars(target) for target in self.targets],
            "provenance": self.provenance or {},
            "avatar_seed": hashlib.sha256(self.subject.identifier.encode()).hexdigest()[:16],
            "calibration": True,
        }
