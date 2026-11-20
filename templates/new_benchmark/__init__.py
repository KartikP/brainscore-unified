"""Registers your benchmark in brainscore.benchmark_registry so
load_benchmark('your-benchmark') works.

To install: copy this folder to ``brainscore/benchmarks/your_name/``; it registers by
its presence (no other file changes).

Write one literal line for each identifier, here and nowhere else: core's plugin
runner reads these lines. ``benchmark.py`` loads data and metrics; ``_load`` imports
it when the benchmark is built, not at the top of this file, so ``import brainscore``
stays light.
"""
from brainscore import benchmark_registry


def _load(**configuration):
    from .benchmark import YourBenchmark
    return YourBenchmark(**configuration)


benchmark_registry['your-benchmark'] = _load
