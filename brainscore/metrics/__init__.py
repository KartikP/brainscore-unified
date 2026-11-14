"""Metrics for the unified interface.

Each registered metric is a plugin folder with its registry line: ``topographic`` (the
topographic-organization metrics, TDANN / TopoLM fold-in) and ``direct_comparison`` (a
per-unit comparison with no fitted readout). The tests of the metrics live under ``tests/``.
``encoding`` is the shared regression kit that benchmarks compose; it registers nothing.
"""
from .topographic import *  # noqa: F401,F403
from .topographic import __all__ as _topographic
from .direct_comparison import *  # noqa: F401,F403
from .direct_comparison import __all__ as _direct_comparison

__all__ = [*_topographic, *_direct_comparison]
