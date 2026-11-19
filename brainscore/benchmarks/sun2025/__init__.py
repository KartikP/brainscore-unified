"""Register the Sun 2025 motion benchmark; the data loads when the benchmark is built."""
from brainscore import benchmark_registry


def _load(**configuration):
    from .benchmark import load_benchmark
    return load_benchmark(**configuration)


benchmark_registry['Sun2025-motion-vector-error'] = _load
