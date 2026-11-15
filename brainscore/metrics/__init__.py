"""Metrics for the unified interface.

Each registered metric is a plugin folder with its registry line, imported when the
package loads: ``topographic`` (the topographic-organization metrics, TDANN / TopoLM
fold-in) and ``direct_comparison`` (a per-unit comparison with no fitted readout).
Import a metric class from its plugin, for example
``from brainscore.metrics.topographic import TopographicMetric``. The tests of the metrics live under ``tests/``.
``encoding`` is the shared regression kit that benchmarks compose; it registers nothing.
"""
from brainscore import _import_plugins

_import_plugins(__name__)    # every plugin folder here registers by its presence
