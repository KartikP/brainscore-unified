"""Register explicitly named EWoK evaluation protocols."""
from functools import partial
from brainscore import benchmark_registry


def _load(mode: str, **configuration):
    from .benchmark import EWoKBenchmark
    return EWoKBenchmark(mode=mode, **configuration)


for _mode in ('logprobs', 'choice'):
    benchmark_registry[f'EWoK-core-1.0-{_mode}'] = partial(_load, _mode)
