"""Register the Sun motion-perception benchmark without loading study data."""
from brainscore import benchmark_registry

IDENTIFIER = 'Sun2025-motion-vector-error'


def _load(**configuration):
    from .benchmark import load_benchmark
    return load_benchmark(**configuration)


benchmark_registry[IDENTIFIER] = _load
