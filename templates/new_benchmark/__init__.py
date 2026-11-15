"""Registers your benchmark in brainscore.benchmark_registry so
load_benchmark('your-benchmark') works.

To install: copy this folder to ``brainscore/benchmarks/your_name/``; it registers by
its presence (no other file changes).
"""
from . import benchmark  # noqa: F401  registration happens at import time
