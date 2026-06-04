"""Embodied grid-game benchmark — registered variant.

The closed-loop reach-the-goal game as a Brain-Score ``Benchmark`` (scored via
``process(EnvironmentStep)``). See ``benchmark.py``.
"""
from brainscore import benchmark_registry
from .benchmark import GridGameBenchmark

benchmark_registry['GridGame-reach-5x5'] = lambda: GridGameBenchmark(size=5)
