"""Frozen copy of ``brainscore/benchmarks/_score.py`` at commit 9442779.

The reference that ``test_benchmark_score_no_ceiling.py`` compares the current
score builder with: every score value must stay the same.
"""
from typing import Optional

from brainscore_core.metrics import Score


def not_estimated() -> Score:
    """The ceiling of a benchmark that measures none: the score is the raw value."""
    ceiling = Score(float('nan'))
    ceiling.attrs['status'] = 'not_estimated'
    return ceiling


def benchmark_score(benchmark, value: float, *, raw: float, metric: str, normalized: bool,
                    ceiling: Optional[Score] = None, **attrs) -> Score:
    """The score of ``benchmark`` with the common attributes, then ``attrs``.

    ``ceiling`` defaults to ``benchmark.ceiling``; pass it only when the score
    used another reference (for example the median of per-voxel ceilings).
    """
    if not isinstance(normalized, bool):
        raise TypeError('normalized must be True or False')
    if ceiling is None:
        ceiling = benchmark.ceiling
    score = Score(value)
    score.attrs['raw'] = raw if isinstance(raw, Score) else Score(float(raw))
    score.attrs['ceiling'] = ceiling if isinstance(ceiling, Score) else Score(float(ceiling))
    score.attrs['normalized'] = normalized
    score.attrs['metric'] = metric
    score.attrs['benchmark_version'] = benchmark.version
    score.attrs.update(attrs)
    return score
