"""Unified Brain-Score benchmarks.

Benchmarks here reach a subject only through a session (a core helper such as
`neural_response`, which calls `participate`) and declare what they send
(`required_input_channels`), what they ask for (`requested_output_channels`)
and how their trials relate (`trial_memory`).

Each builds its score with `benchmark_score`, which writes the attributes every
unified score carries (`raw`, `ceiling`, `normalized`, `metric`,
`benchmark_version`); `not_estimated()` is the ceiling of a benchmark that
measures none.
"""
from ._score import benchmark_score, not_estimated  # noqa: F401

from brainscore import _import_plugins

_import_plugins(__name__)    # every plugin folder here registers by its presence
