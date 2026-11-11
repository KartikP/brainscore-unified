"""Shared checks for what every benchmark declares (Contract Rules 5.1, 5.2, 5.12).

A benchmark states what it sends (``required_input_channels``), what it asks
for (``requested_output_channels``) and how its trials relate
(``trial_memory``). Pre-flight compares these with a subject's declarations
by exact channel name, so the tests below need no model, no data and no session.
"""
import pytest

from brainscore_core.compatibility import CompatibilityError, check_compatibility
from brainscore_core.contract import Subject


class Declared(Subject):
    """A subject that declares channels and does nothing else."""

    identifier = 'declared-only'

    def __init__(self, in_channels, out_channels):
        self._in = frozenset(in_channels)
        self._out = frozenset(out_channels)

    @property
    def in_channels(self):
        return self._in

    @property
    def out_channels(self):
        return self._out

    def participate(self, session):
        raise AssertionError('pre-flight must not run a session')


def assert_declares(benchmark, inputs, outputs, protocol='none'):
    """The declarations are present and exactly these."""
    assert set(benchmark.required_input_channels) == set(inputs)
    assert set(benchmark.requested_output_channels) == set(outputs)
    assert benchmark.trial_memory == protocol


def assert_preflight(benchmark):
    """A matching subject passes; a subject without the inputs or an output is N/A."""
    inputs = set(benchmark.required_input_channels)
    outputs = set(benchmark.requested_output_channels)
    assert check_compatibility(Declared(inputs, outputs), benchmark).kind == 'full'
    with pytest.raises(CompatibilityError, match='takes in none of the inputs'):
        check_compatibility(Declared(set(), outputs), benchmark)
    for channel in sorted(outputs):
        with pytest.raises(CompatibilityError, match='cannot produce output channels'):
            check_compatibility(Declared(inputs, outputs - {channel}), benchmark)
