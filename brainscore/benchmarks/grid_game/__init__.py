"""Closed-loop grid-game benchmark — registered variant.

The closed-loop reach-the-goal game as a Brain-Score ``Benchmark`` (scored
through a closed-loop session). See ``benchmark.py``.
"""
from brainscore import benchmark_registry


def _load(**configuration):
    from .benchmark import GridGameBenchmark
    return GridGameBenchmark(**configuration)


benchmark_registry['GridGame-reach-5x5'] = lambda: _load(size=5)
