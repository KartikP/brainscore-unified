"""The attributes every unified benchmark score carries, written in one place.

A benchmark builds its score with :func:`benchmark_score`; ``score()`` then adds
what pre-flight and the run know. A reader finds the same facts under the same
keys on every unified score:

- ``raw``: a ``Score``, the metric value before any ceiling normalization;
- ``ceiling``: a ``Score``, the reference the benchmark holds. When no ceiling
  is measured it is :func:`not_estimated` (NaN, ``attrs['status'] ==
  'not_estimated'``); a ceiling of 1.0 never means "none";
- ``normalized``: ``True`` when the score is the raw value divided by a measured
  ceiling, ``False`` when the score is the raw value (Communication Rule 2.4:
  the two never share a column). It follows from the ceiling;
- ``metric``: what was measured;
- ``benchmark_version``: the benchmark's version.

``score()`` adds ``benchmark_identifier``, ``model_identifier``,
``compatibility``, ``trial_memory`` and the channel provenance. A benchmark keeps
its own extra attributes beside these.
"""
import math
from typing import Optional

from brainscore_core.metrics import Score


# The attributes benchmark_score writes; a caller cannot set them.
_DERIVED = ('raw', 'ceiling', 'normalized', 'metric', 'benchmark_version')


def not_estimated() -> Score:
    """The ceiling of a benchmark that measures none: the score is the raw value."""
    ceiling = Score(float('nan'))
    ceiling.attrs['status'] = 'not_estimated'
    return ceiling


def benchmark_score(benchmark, *, raw: float, metric: str, ceiling: Optional[Score] = None,
                    value: Optional[float] = None, **attrs) -> Score:
    """The score of ``benchmark`` with the common attributes, then ``attrs``.

    The value is ``raw / ceiling`` when the ceiling is measured and ``raw`` when
    it is :func:`not_estimated`; ``normalized`` says which. ``ceiling`` defaults
    to ``benchmark.ceiling``; pass it only when the score used another reference
    (for example the median of per-voxel ceilings). Pass ``value`` only when the
    benchmark normalizes each target on its own (for example ``median(r_i / c_i)``);
    it needs a measured ceiling. ``attrs`` cannot hold the attributes this
    function writes (``normalized``, ``benchmark_version``, ...), and a NaN ceiling
    must be :func:`not_estimated`.
    """
    reserved = sorted(set(attrs) & set(_DERIVED))
    if reserved:
        raise TypeError(f'benchmark_score writes {reserved} itself; do not pass them')
    if ceiling is None:
        ceiling = benchmark.ceiling
    if not isinstance(ceiling, Score):
        ceiling = Score(float(ceiling))
    normalized = not math.isnan(float(ceiling))
    if not normalized and ceiling.attrs.get('status') != 'not_estimated':
        raise ValueError('the ceiling is NaN but not not_estimated(); pass not_estimated() '
                         'for a benchmark that measures no ceiling')
    if value is None:
        value = float(raw) / float(ceiling) if normalized else float(raw)
    elif not normalized:
        raise ValueError('value is for a per-target normalization; it needs a measured ceiling')
    score = Score(value)
    score.attrs['raw'] = raw if isinstance(raw, Score) else Score(float(raw))
    score.attrs['ceiling'] = ceiling
    score.attrs['normalized'] = normalized
    score.attrs['metric'] = metric
    score.attrs['benchmark_version'] = benchmark.version
    score.attrs.update(attrs)
    return score
