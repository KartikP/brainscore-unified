"""The attributes every unified benchmark score carries, written in one place.

A benchmark builds its score with :func:`benchmark_score`; ``score()`` then adds
what pre-flight and the run know. A reader finds the same facts under the same
keys on every unified score:

- ``raw``: a ``Score``, the metric value before any ceiling normalization;
- ``ceiling``: a ``Score``, the reference the benchmark holds. When no ceiling
  is measured it is :func:`not_estimated` (NaN, ``attrs['status'] ==
  'not_estimated'``); LeBel2023, the induced-dyslexia benchmarks and every
  Lahner2024 variant except ``Lahner2024-fMRI-naturalistic-visualROI`` keep
  ``Score(1.0)``;
- ``normalized``: ``True`` when the score is the raw value divided by a measured
  ceiling, ``False`` when the score is the raw value (Communication Rule 2.4:
  the two never share a column);
- ``metric``: what was measured;
- ``benchmark_version``: the benchmark's version.

``score()`` adds ``benchmark_identifier``, ``model_identifier``,
``compatibility``, ``trial_memory`` and the channel provenance. A benchmark keeps
its own extra attributes beside these.
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
