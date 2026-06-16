"""Registers your benchmark in brainscore.benchmark_registry so
load_benchmark('your-benchmark') works.

To install: copy this folder to ``brainscore/benchmarks/your_name/``, then add
``from . import your_name`` to ``brainscore/benchmarks/__init__.py``.
"""
from . import benchmark  # noqa: F401  registration happens at import time
