"""Fit a linear map from features to recorded data, and score it per target.

This is the one encoding fit of the repository. Every benchmark that maps model
features to recorded responses with a ridge uses it. A target is one column of
the recorded data: a voxel, a vertex or a parcel.

A fit has three parts, and the benchmark hands in each one:

- a solver, the arithmetic of one ridge solve: ``SklearnRidge`` or
  ``NormalEquations``;
- a penalty: a number, or a chooser that ``grid_search`` builds from a list of
  candidates and an inner split;
- the outer folds, a list of ``(train_rows, test_rows)``.

``ridge(penalty, solver)`` joins a penalty and a solver into a fit.
``cross_validated_r(features, Y, folds, fit)`` runs the fit on each fold and
returns the Pearson r of each target on the held-out rows. The Pearson r is
``brainscore_core.metrics.per_unit_pearson``; this module has no second one.

Features are one 2-D array, or a list of 2-D arrays (bands) with the same rows.
A banded penalty has one value per band.

The settings of the solvers and choosers exist to keep existing scores
bit-for-bit (Contract Rules 10). Each one reproduces the arithmetic of one
benchmark; the docstrings name the benchmark. A new benchmark takes the
defaults: ``NormalEquations()`` (an intercept, float64 solves) with penalties
chosen on held-out groups (``group_kfold``).
"""
from dataclasses import dataclass
from itertools import islice, product
from numbers import Real
from typing import Callable, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from brainscore_core.metrics import per_unit_pearson

__all__ = [
    'Chosen', 'NormalEquations', 'SklearnRidge',
    'cross_validated_r', 'fixed', 'grid_search', 'group_kfold',
    'held_out_r', 'penalty_grid', 'random_group_split',
    'ridge', 'row_kfold', 'summarize',
]


# -- Penalties chosen ----------------------------------------------------------

@dataclass(frozen=True, eq=False)
class Chosen:
    """The penalties of one fit: the candidates and the index chosen for each target."""

    candidates: Tuple[Tuple[float, ...], ...]
    index: np.ndarray

    def per_target(self) -> np.ndarray:
        """``(n_targets, n_bands)``: the penalties of each target."""
        return np.asarray(self.candidates, dtype=float)[self.index]

    @property
    def shared(self) -> Tuple[float, ...]:
        """The penalty tuple, when every target has the same one."""
        distinct = np.unique(self.index)
        if len(distinct) != 1:
            raise ValueError(f'{len(distinct)} penalty tuples were chosen, not one.')
        return self.candidates[int(distinct[0])]


# -- Splits ------------------------------------------------------------------

def row_kfold(n_rows: int, n_splits: int = 5, random_state: int = 0):
    """Shuffled KFold over rows; yields ``(train_positions, test_positions)``."""
    from sklearn.model_selection import KFold

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    yield from kf.split(np.arange(n_rows))


def group_kfold(groups, n_splits: int = 5, random_state: int = 0,
                n_used: Optional[int] = None):
    """Shuffled KFold over the distinct groups; yields boolean ``(train, test)`` masks.

    All rows of one group (a run, a story, a stimulus) stay on one side.
    ``n_used`` keeps only the first folds: Algonauts chooses its penalties on
    the first fold alone.
    """
    groups = np.asarray(groups)
    distinct = np.unique(groups)
    folds = row_kfold(len(distinct), n_splits=n_splits, random_state=random_state)
    for train_pos, test_pos in islice(folds, n_used):
        yield (np.isin(groups, distinct[train_pos]),
               np.isin(groups, distinct[test_pos]))


def random_group_split(groups, fraction: float = 0.25, seed: int = 0):
    """One split that holds out a random ``fraction`` of the groups (at least one)."""
    groups = np.asarray(groups)
    distinct = np.unique(groups)
    n_check = max(1, int(len(distinct) * fraction))
    check_groups = np.random.default_rng(seed).choice(distinct, n_check, replace=False)
    check = np.isin(groups, check_groups)
    return [(~check, check)]


def _empty(rows) -> bool:
    rows = np.asarray(rows)
    return not rows.any() if rows.dtype == bool else rows.size == 0


# -- Solvers -----------------------------------------------------------------

def _as_bands(features) -> List[np.ndarray]:
    return list(features) if isinstance(features, (list, tuple)) else [features]


def _stack(bands: Sequence[np.ndarray]) -> np.ndarray:
    return bands[0] if len(bands) == 1 else np.concatenate(bands, axis=1)


def _project(gram, cross, target):
    """``cross @ gram⁻¹ @ target``, solved against whichever side is narrower.

    ``gram`` is symmetric, so the product may be grouped either way. Voxelwise
    encoding has far more targets than held-out rows, so solving against
    ``cross.T`` is cheaper there (measured 1.3-1.8x). Near parity the copy for
    LAPACK costs more than it saves; the 3x threshold comes from that
    measurement. The two groupings round differently.
    """
    if target.shape[1] >= 3 * cross.shape[0]:
        return np.linalg.solve(gram, cross.T).T @ target
    return cross @ np.linalg.solve(gram, target)


class SklearnRidge:
    """scikit-learn's ``Ridge`` with an intercept, at one fixed penalty.

    The Lahner2024 benchmarks fit with it at a penalty of 1. scikit-learn's
    solver rounds differently from ``NormalEquations``, so it stays a solver of
    its own. It does not take part in a grid search.
    """

    def prepare(self, train_bands, Y_train, test_bands):
        return _SklearnFit(_stack(train_bands), Y_train, _stack(test_bands))


class _SklearnFit:
    y_mean = None

    def __init__(self, X, Y, X_test):
        self._X, self._Y, self._X_test = X, Y, X_test

    def predict(self, penalties, columns=None, centred=False):
        if centred or len(set(penalties)) != 1:
            raise ValueError('SklearnRidge takes one fixed penalty for all features; '
                             'use NormalEquations for a grid search or a banded penalty.')
        from sklearn.linear_model import Ridge
        Y = self._Y if columns is None else self._Y[:, columns]
        return Ridge(alpha=penalties[0]).fit(self._X, Y).predict(self._X_test)


class NormalEquations:
    """Ridge solved in closed form, ``(XᵀX + diag(λ)) W = XᵀY``, with one λ per band.

    The Gram matrix is computed once per fit; only its diagonal changes from one
    penalty to the next.

    Args:
        centre: what is centred on the training rows. ``'both'`` centres the
            features and the targets and adds the target mean back: a ridge with
            an unpenalized intercept (Algonauts2025). ``'features'`` centres
            only the features and fits no intercept (LeBel2023). ``'none'``
            centres nothing (the former Lahner2024 ``banded`` analysis).
        solve_dtype: the precision of the solve. The Gram matrix is computed in
            the precision of the inputs and then converted. ``None`` keeps the
            input precision (LeBel2023 solves in float32).
        shape_aware: solve in the cheaper form for the shape: the dual form
            ``X(XᵀX + αI)⁻¹ = (XXᵀ + αI)⁻¹X`` when features outnumber rows, and
            ``_project`` for many targets. Same estimator, other rounding
            (LeBel2023). The dual form takes one penalty for all features.
    """

    CENTRES = ('none', 'features', 'both')

    def __init__(self, *, centre: str = 'both', solve_dtype=np.float64,
                 shape_aware: bool = False):
        if centre not in self.CENTRES:
            raise ValueError(f'centre must be one of {self.CENTRES}; got {centre!r}.')
        self.centre = centre
        self.solve_dtype = solve_dtype
        self.shape_aware = shape_aware

    def prepare(self, train_bands, Y_train, test_bands):
        return _NormalFit(self, train_bands, Y_train, test_bands)


class _NormalFit:
    def __init__(self, solver: NormalEquations, train_bands, Y, test_bands):
        X, X_test = _stack(train_bands), _stack(test_bands)
        self._widths = [band.shape[1] for band in train_bands]
        self._solver = solver
        self.y_mean = None
        if solver.centre != 'none':
            x_mean = X.mean(0)
            X, X_test = X - x_mean, X_test - x_mean
        if solver.centre == 'both':
            self.y_mean = Y.mean(0)
            Y = Y - self.y_mean
        self._X, self._Y = X, Y
        self._dual = solver.shape_aware and X.shape[1] > X.shape[0]
        if self._dual:
            self._gram, self._cross = X @ X.T, X_test @ X.T
        else:
            self._gram, self._cross = X.T @ X, X_test
        self._full_target = None

    def _target(self, columns):
        if columns is None:
            if self._full_target is None:
                self._full_target = self._Y if self._dual else self._X.T @ self._Y
            return self._full_target
        Y = self._Y[:, columns]
        return Y if self._dual else self._X.T @ Y

    def _penalty_diagonal(self, penalties, dtype):
        if self._dual:
            if len(set(penalties)) != 1:
                raise ValueError('The dual form takes one penalty for all features.')
            return np.full(self._gram.shape[0], penalties[0], dtype=dtype)
        return np.concatenate([np.full(width, penalty, dtype=dtype)
                               for width, penalty in zip(self._widths, penalties)])

    def predict(self, penalties, columns=None, centred=False):
        """Predictions of the test rows for one penalty tuple.

        ``centred=True`` leaves out the target mean; the inner choice compares
        such predictions with targets centred the same way.
        """
        dtype = self._solver.solve_dtype
        gram = self._gram.astype(dtype) if dtype is not None else self._gram.copy()
        gram[np.diag_indices_from(gram)] += self._penalty_diagonal(penalties, gram.dtype)
        target = self._target(columns)
        if self._solver.shape_aware:
            predictions = _project(gram, self._cross, target)
        else:
            predictions = self._cross @ np.linalg.solve(gram, target)
        if not centred and self.y_mean is not None:
            predictions = predictions + (self.y_mean if columns is None
                                         else self.y_mean[columns])
        return predictions


# -- Penalty choosers --------------------------------------------------------

def penalty_grid(values: Sequence[float], n_bands: int,
                 n_candidates: Optional[int] = None, random_state: int = 0):
    """The penalty tuples to try: the full grid, or a random sample of it.

    With many bands the full grid is too large (11 values and 2 bands give 121
    tuples). A random search, as himalaya's ``solver='random_search'`` does,
    tries ``n_candidates`` tuples: every tuple with one shared penalty, then
    random draws from the grid.
    """
    values = list(values)
    if n_candidates is None or len(values) ** n_bands <= n_candidates:
        return list(product(values, repeat=n_bands))
    rng = np.random.default_rng(random_state)
    tuples = {(value,) * n_bands for value in values}
    while len(tuples) < n_candidates:
        tuples.add(tuple(float(a) for a in rng.choice(values, size=n_bands)))
    return sorted(tuples)


def fixed(alpha: float):
    """A chooser that gives every band and every target the penalty ``alpha``."""
    def choose(solver, train_bands, Y_train, groups):
        return Chosen(((alpha,) * len(train_bands),),
                      np.zeros(Y_train.shape[1], dtype=int))
    return choose


def grid_search(candidates: Iterable[Sequence[float]], split: Callable, *,
                per_target: bool = False, undefined_r: str = 'zero'):
    """A chooser that picks the candidate with the best Pearson r on an inner split.

    Args:
        candidates: penalty tuples, one value per band, tried in this order;
            the first best one wins.
        split: ``split(groups) -> [(fit_rows, check_rows), ...]`` over the
            training rows. ``groups`` has one label per training row (row
            numbers when the benchmark gives none). The score of a candidate is
            its r on the check rows, averaged over the splits.
        per_target: one candidate per target (LeBel2023 ``per_voxel_alpha``)
            instead of one for all targets.
        undefined_r: a target without a correlation (a constant column) counts
            as r = 0 (``'zero'``) or is left out of the mean over targets
            (``'skip'``, Algonauts2025).
    """
    candidates = tuple(tuple(c) for c in candidates)
    if undefined_r not in ('zero', 'skip'):
        raise ValueError(f"undefined_r must be 'zero' or 'skip'; got {undefined_r!r}.")

    def choose(solver, train_bands, Y_train, groups):
        groups = np.arange(len(Y_train)) if groups is None else groups
        total, n_splits = None, 0
        for fit_rows, check_rows in split(groups):
            if _empty(fit_rows) or _empty(check_rows):
                continue
            prepared = solver.prepare([band[fit_rows] for band in train_bands],
                                      Y_train[fit_rows],
                                      [band[check_rows] for band in train_bands])
            Y_check = Y_train[check_rows]
            if prepared.y_mean is not None:
                Y_check = Y_check - prepared.y_mean
            scores = np.full((len(candidates), Y_train.shape[1]) if per_target
                             else len(candidates), np.nan)
            for i, penalties in enumerate(candidates):
                try:
                    predictions = prepared.predict(penalties, centred=True)
                except np.linalg.LinAlgError:
                    continue                      # left NaN: never chosen
                r = per_unit_pearson(Y_check, predictions)
                if undefined_r == 'zero':
                    r = np.nan_to_num(r, nan=0.0)
                if per_target:
                    scores[i] = r
                else:
                    scores[i] = np.mean(r) if undefined_r == 'zero' else np.nanmean(r)
            total = scores if total is None else total + scores
            n_splits += 1
        if n_splits == 0:
            raise ValueError('The inner split gave no pair of non-empty fit and check rows.')
        ranked = total / n_splits
        ranked = np.where(np.isnan(ranked), -np.inf, ranked)
        if not np.isfinite(ranked).any():
            raise np.linalg.LinAlgError('No penalty candidate gave a usable fit.')
        if per_target:
            index = np.argmax(ranked, axis=0)
        else:
            index = np.full(Y_train.shape[1], int(np.argmax(ranked)))
        return Chosen(candidates, index)

    return choose


# -- Fits ----------------------------------------------------------------------

def ridge(penalty, solver):
    """A fit: choose the penalties on the training rows, then refit and predict.

    ``penalty`` is a number or a chooser (``fixed``, ``grid_search``). The
    returned fit is ``fit(train_bands, Y_train, test_bands, groups=None) ->
    (predictions, Chosen)``. Targets with different penalties are refitted in
    groups, one solve per distinct tuple.
    """
    choose = fixed(penalty) if isinstance(penalty, Real) else penalty

    def fit(train_bands, Y_train, test_bands, groups=None):
        train_bands, test_bands = _as_bands(train_bands), _as_bands(test_bands)
        chosen = choose(solver, train_bands, Y_train, groups)
        prepared = solver.prepare(train_bands, Y_train, test_bands)
        distinct = np.unique(chosen.index)
        if len(distinct) == 1:
            return prepared.predict(chosen.candidates[int(distinct[0])]), chosen
        predictions = None
        for index in distinct:
            columns = np.flatnonzero(chosen.index == index)
            part = prepared.predict(chosen.candidates[int(index)], columns=columns)
            if predictions is None:
                predictions = np.empty((part.shape[0], Y_train.shape[1]), dtype=part.dtype)
            predictions[:, columns] = part
        return predictions, chosen

    return fit


# -- Scores ------------------------------------------------------------------

def held_out_r(Y, predictions) -> np.ndarray:
    """Per-target Pearson r over the rows that received a prediction.

    Rows without one hold NaN. When every row has one, the arrays are used as
    they are, so the result does not depend on a copy.
    """
    predicted = ~np.isnan(predictions[:, 0])
    if predicted.all():
        return per_unit_pearson(Y, predictions)
    if not predicted.any():
        return np.full(Y.shape[1], np.nan, dtype=np.float32)
    return per_unit_pearson(Y[predicted], predictions[predicted])


def cross_validated_r(features, Y, folds, fit, *, groups=None,
                      target_batch: Optional[int] = None):
    """Held-out Pearson r of each target, and the penalties chosen in each fit.

    Args:
        features: ``(n_rows, n_features)`` or a list of bands with the same rows.
        Y: ``(n_rows, n_targets)`` recorded data.
        folds: ``(train_rows, test_rows)`` pairs, as masks or positions.
        fit: from ``ridge``.
        groups: one label per row, handed to the inner split (stories, stimuli).
        target_batch: fit the targets in batches of this size, to bound memory.
            A chooser then chooses per batch, so a shared penalty depends on
            the batch (finding C2). The Lahner2024 time-resolved benchmark
            batches at a fixed penalty, where the batch changes only float32
            rounding (``SklearnRidge``).

    Returns ``(r, chosen)``: ``r`` has one value per target (NaN without a
    correlation); ``chosen`` lists what each fit returned, batch by batch and
    fold by fold.
    """
    bands = _as_bands(features)
    folds = list(folds)
    n_targets = Y.shape[1]
    if target_batch is None:
        batches = [None]
    else:
        batches = [slice(start, min(start + target_batch, n_targets))
                   for start in range(0, n_targets, target_batch)]
    r, chosen = None, []
    for columns in batches:
        Y_part = Y if columns is None else Y[:, columns]
        held = np.full_like(Y_part, np.nan)
        for train, test in folds:
            predictions, fold_chosen = fit(
                [band[train] for band in bands], Y_part[train],
                [band[test] for band in bands],
                None if groups is None else groups[train])
            held[test] = predictions
            chosen.append(fold_chosen)
        part = held_out_r(Y_part, held)
        if columns is None:
            return part, chosen
        if r is None:
            r = np.empty(n_targets, dtype=part.dtype)
        r[columns] = part
    return r, chosen


def summarize(r) -> Tuple[np.ndarray, float, float]:
    """The finite values of ``r``, their median and their mean."""
    finite = r[~np.isnan(r)]
    return finite, float(np.median(finite)), float(np.mean(finite))
