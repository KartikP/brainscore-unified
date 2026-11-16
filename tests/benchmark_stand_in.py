"""One stand-in benchmark for the tests of the score builders.

It carries every attribute a real benchmark always has (``BenchmarkBase``:
``identifier``, ``version``, ``ceiling``, ``parent``, ``bibtex``), so a score builder
that starts to read one more of them does not break the tests that use it.
"""
from brainscore_core.benchmarks import BenchmarkBase


class StandInBenchmark(BenchmarkBase):
    """A benchmark with the fields of ``BenchmarkBase`` and the extra attributes a test gives."""

    def __init__(self, ceiling, version=1, identifier='stand-in-benchmark', **attributes):
        super().__init__(identifier=identifier, ceiling=ceiling, version=version, parent='stand-in')
        for name, value in attributes.items():
            setattr(self, name, value)

    def __call__(self, candidate):
        raise NotImplementedError('a stand-in for the score builders; it runs no subject')


def without_init(cls, identifier, ceiling, version, **attributes):
    """An instance of a real benchmark class built without ``__init__`` (which loads
    data), with the fields of ``BenchmarkBase`` and the extra attributes a test gives."""
    benchmark = object.__new__(cls)
    BenchmarkBase.__init__(benchmark, identifier=identifier, ceiling=ceiling, version=version,
                           parent='stand-in')
    for name, value in attributes.items():
        setattr(benchmark, name, value)
    return benchmark
