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
- ``benchmark_version``: the benchmark's version;
- ``leaderboard_eligible``: written when the benchmark declares it (a benchmark for
  local use only declares ``leaderboard_eligible = False``); storage refuses such a
  score.

A behavior score also carries what its metric did with each trial (Contract Rule
6.2). The benchmark passes its stored answers (``answers=``) and its policy in one
sentence (``answer_policy=``); :func:`benchmark_score` reads the state core gave each
trial (``response_state``) and writes:

- ``n_trials``, ``n_valid``, ``n_invalid``, ``n_missing``: the counts of the states,
  summed over the answers; ``counts_by_condition`` when the answers are given by
  condition;
- ``answer_policy``: what the metric does with an invalid and a missing trial;
- ``result_validity``: ``'no limit stated'`` when the benchmark passes no
  ``invalid_limit`` (the default; no check was made). With a limit: ``'invalid'``
  when the share of invalid and missing trials is above it or when there is no
  trial, and ``result_validity_reason`` then gives the share and the limit or says
  that no trial was answered; ``'valid'`` otherwise. Such a result is returned, not
  raised.

A metric selects its valid rows with ``answer['response_state'].values == 'valid'``.

``score()`` adds ``benchmark_identifier``, ``model_identifier``,
``compatibility``, ``trial_memory`` and the channel provenance. A benchmark keeps
its own extra attributes beside these.

The voxel benchmarks share two functions here: :func:`noise_ceiling` (the voxelwise
noise ceiling from repeated trials) and :func:`voxel_score` (the median over voxels
of r, or of r / c with a ceiling per voxel).
"""
import math
from collections.abc import Mapping
from numbers import Real
from typing import Optional

import numpy as np

from brainscore_core.metrics import Score


# The attributes benchmark_score writes; a caller cannot set them.
_DERIVED = ('raw', 'ceiling', 'normalized', 'metric', 'benchmark_version', 'leaderboard_eligible',
            'n_trials', 'n_valid', 'n_invalid', 'n_missing', 'counts_by_condition',
            'answer_policy', 'result_validity', 'result_validity_reason')
_STATES = ('valid', 'invalid', 'missing')


def not_estimated() -> Score:
    """The ceiling of a benchmark that measures none: the score is the raw value."""
    ceiling = Score(float('nan'))
    ceiling.attrs['status'] = 'not_estimated'
    return ceiling


def _state_counts(answer) -> dict:
    if isinstance(answer, Score):
        # A behavior score of a benchmark this benchmark runs: its counts were read
        # from the states of its own answer.
        try:
            return {key: int(answer.attrs[key]) for key in ('n_trials', 'n_valid', 'n_invalid', 'n_missing')}
        except KeyError:
            raise ValueError('a behavior score passed as answers carries n_trials, n_valid, '
                             'n_invalid and n_missing') from None
    try:
        states = [str(state) for state in answer['response_state'].values]
    except KeyError:
        raise ValueError('a stored behavior answer has a response_state for each trial; this '
                         'answer has none (bare behavior has no states)') from None
    unknown = sorted(set(states) - set(_STATES))
    if unknown:
        raise ValueError(f'unknown trial states {unknown}; the states are {_STATES}')
    return {'n_trials': len(states), **{f'n_{state}': states.count(state) for state in _STATES}}


def _answer_attrs(answers, answer_policy, invalid_limit) -> dict:
    """The counts, the policy and the result validity of a behavior score."""
    if not answer_policy or not isinstance(answer_policy, str):
        raise TypeError('a behavior score states its answer_policy: what its metric does '
                        'with an invalid and with a missing trial')
    if invalid_limit is not None and not (
            isinstance(invalid_limit, Real) and not isinstance(invalid_limit, bool)
            and 0 <= invalid_limit <= 1):
        raise ValueError(f'invalid_limit is a share of the trials: pass a number from 0 to 1 '
                         f'(or None for no limit), not {invalid_limit!r}')
    by_condition = None
    if isinstance(answers, Mapping):
        by_condition = {str(condition): _state_counts(answer) for condition, answer in answers.items()}
        parts = list(by_condition.values())
    else:
        parts = [_state_counts(answer) for answer in
                 (answers if isinstance(answers, (list, tuple)) else [answers])]
    attrs = {key: sum(part[key] for part in parts)
             for key in ('n_trials', 'n_valid', 'n_invalid', 'n_missing')}
    if by_condition is not None:
        attrs['counts_by_condition'] = by_condition
    attrs['answer_policy'] = answer_policy
    attrs['result_validity'] = 'no limit stated' if invalid_limit is None else 'valid'
    if invalid_limit is not None and not attrs['n_trials']:
        # A check against the limit needs answered trials; with none it did not run.
        attrs['result_validity'] = 'invalid'
        attrs['result_validity_reason'] = 'no trial was answered: the answers hold zero trials'
    elif invalid_limit is not None:
        share = (attrs['n_invalid'] + attrs['n_missing']) / attrs['n_trials']
        if share > invalid_limit:
            attrs['result_validity'] = 'invalid'
            attrs['result_validity_reason'] = (
                f'{share:g} of the trials are invalid or missing, above the limit {invalid_limit:g}')
    return attrs


def benchmark_score(benchmark, *, raw: float, metric: str, ceiling: Optional[Score] = None,
                    value: Optional[float] = None, answers=None, answer_policy: Optional[str] = None,
                    invalid_limit: Optional[float] = None, **attrs) -> Score:
    """The score of ``benchmark`` with the common attributes, then ``attrs``.

    The value is ``raw / ceiling`` when the ceiling is measured and ``raw`` when
    it is :func:`not_estimated`; ``normalized`` says which. ``ceiling`` defaults
    to ``benchmark.ceiling``; pass it only when the score used another reference
    (for example the median of per-voxel ceilings). Pass ``value`` only when the
    benchmark normalizes each target on its own (for example ``median(r_i / c_i)``);
    it needs a measured ceiling. ``attrs`` cannot hold the attributes this
    function writes (``normalized``, ``benchmark_version``, ...), and a NaN ceiling
    must be :func:`not_estimated`.

    ``answers`` (a stored behavior answer, a list of them, or a mapping of condition to
    answer; a behavior score of a benchmark this benchmark runs stands for its answer)
    adds the counts of the trial states, ``answer_policy`` (required with it)
    and ``result_validity`` against ``invalid_limit`` (a share; none by default, and
    then ``result_validity`` is ``'no limit stated'``).
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
    eligible = getattr(benchmark, 'leaderboard_eligible', None)
    if eligible is not None:
        score.attrs['leaderboard_eligible'] = eligible
    if answers is not None:
        score.attrs.update(_answer_attrs(answers, answer_policy, invalid_limit))
    elif answer_policy is not None or invalid_limit is not None:
        raise TypeError('answer_policy and invalid_limit describe the answers; pass answers=')
    score.attrs.update(attrs)
    return score


def noise_ceiling(responses: np.ndarray, image: np.ndarray, viewing: np.ndarray, n: int = 3) -> tuple:
    """``(noise ceiling per voxel in percent, images used)`` from trials x voxels; ``viewing`` is the
    place of each trial among the answered trials of its image (``things.answered_viewing`` of the
    CNeuroMod plugin, 0: no response). An image counts when it has exactly ``n`` answered trials,
    also when shown more often.

    The method of St-Laurent et al. (2026), Scientific Data 13:141 (p.5-6), as the paper's
    ``GLMsingle_noiseceilings.py`` computes it: the responses of each viewing z-scored per voxel
    over the images; the noise variance is the variance over the ``n`` viewings (n - 1), averaged
    over images; signal = 1 - noise (not below 0); NC = 100 ncsnr^2 / (ncsnr^2 + 1/n). A shift and
    scale per voxel cancel. The ceiling of the mean of the ``n`` viewings as a correlation is
    ``sqrt(NC / 100)``.
    """
    image = np.asarray(image).astype(str)
    labels, counts = np.unique(image[viewing > 0], return_counts=True)
    used = np.isin(image, labels[counts == n])
    cube = np.stack([responses[used & (viewing == place)][np.argsort(image[used & (viewing == place)], kind='stable')]
                     for place in range(1, n + 1)], axis=1)                         # images x places x voxels
    cube = (cube - cube.mean(axis=0)) / cube.std(axis=0)                            # per repetition, over images
    noise = np.sqrt(np.mean(np.var(cube, axis=1, ddof=1), axis=0))
    snr = np.sqrt(np.clip(1 - noise ** 2, 0, None)) / noise
    return np.nan_to_num(100 * snr ** 2 / (snr ** 2 + 1 / n)), int(np.sum(counts == n))


def voxel_score(benchmark, per_voxel_r, ceiling_per_voxel: Optional[np.ndarray] = None) -> Score:
    """The score of a voxel benchmark from the held-out r of each scored voxel.

    ``per_voxel_r`` holds one value per scored voxel, in voxel order. One rule
    for every voxel benchmark (Lahner2024, StLaurent2026): a voxel
    counts when its r is finite and, given ``ceiling_per_voxel``, its ceiling is
    finite and above zero. A voxel with a constant response has no correlation;
    a voxel with a reliability at or below zero has no ceiling. Such a voxel is
    left out of the median and the mean and counted in ``n_voxels_excluded``,
    as the Algonauts2025 benchmark leaves out a constant parcel. The attribute
    ``per_voxel_r`` keeps every scored voxel (NaN where r is undefined).

    With ``ceiling_per_voxel`` the score is ``median(r_i / c_i)``, not
    ``median(r_i) / median(c_i)``: the repository convention. Without it the
    raw median is divided by ``benchmark.ceiling`` when it is measured.
    """
    r = np.asarray(per_voxel_r)
    counts = np.isfinite(r)
    if ceiling_per_voxel is not None:
        ceiling_per_voxel = np.asarray(ceiling_per_voxel, dtype=float)
        counts &= np.isfinite(ceiling_per_voxel) & (ceiling_per_voxel > 0)
    kept = r[counts]
    median_r = float(np.median(kept))
    if ceiling_per_voxel is not None:
        score = benchmark_score(
            benchmark, value=float(np.median(np.asarray(kept, dtype=float) / ceiling_per_voxel[counts])),
            raw=median_r, metric='median_per_voxel_pearson_r',
            ceiling=Score(float(np.median(ceiling_per_voxel[counts]))),
            ceiling_aggregation='per_voxel')
    else:
        score = benchmark_score(
            benchmark, raw=median_r, metric='median_per_voxel_pearson_r',
            ceiling_aggregation='scalar')
    score.attrs['mean_r'] = float(np.mean(kept))
    score.attrs['n_voxels_scored'] = int(counts.sum())
    score.attrs['n_voxels_excluded'] = int(r.size - counts.sum())
    score.attrs['per_voxel_r'] = r   # exposed for bootstrap CIs and surface maps
    return score
