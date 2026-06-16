"""Topographic-organization benchmark (the spatial-alignment axis).

Exposes the reusable :class:`TopographicBenchmark` + ``make_topographic_benchmark`` factory.
No concrete instance is registered yet: a real one needs a topographic model + a surface
fMRI assembly with voxel coordinates (stage on EC2, then register the instance here, mirroring
``benchmarks/lahner2024``). The wiring is validated offline in
``tests/test_topographic_benchmark.py``.
"""
from .benchmark import TopographicBenchmark, make_topographic_benchmark

__all__ = ['TopographicBenchmark', 'make_topographic_benchmark']
