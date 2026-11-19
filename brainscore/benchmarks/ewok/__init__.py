"""Register explicitly named EWoK evaluation protocols."""
from functools import partial
from brainscore import benchmark_registry


def _load(mode: str, **configuration):
    from .benchmark import EWoKBenchmark
    return EWoKBenchmark(mode=mode, **configuration)


benchmark_registry['EWoK-core-1.0-logprobs'] = partial(_load, 'logprobs')
benchmark_registry['EWoK-core-1.0-choice'] = partial(_load, 'choice')
