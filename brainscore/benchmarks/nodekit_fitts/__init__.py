"""Fitts pointing task run in nodekit's browser runtime (cursor traces).

Models and humans play the same nodekit site and produce the same pointer
traces. See ``benchmark.py`` and ``brainscore.harnesses.nodekit_browser``.
"""
from brainscore import benchmark_registry
from .benchmark import NodekitFittsBenchmark

benchmark_registry['Nodekit-fitts-pointing'] = lambda: NodekitFittsBenchmark()
