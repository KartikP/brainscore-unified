"""Banded ridge for the Algonauts 2025 benchmark.

This module is pure: it needs no model and no fMRI files, so it is tested on
small synthetic arrays.

A band is the features of one model region, one row per scan (TR), in the row
order of the assembly. ``score_banded`` scores the bands with run-held-out
5-fold cross-validation. In each outer fold it:

1. fits the reduction of each band (``FeatureProjection``, truncated SVD) on
   the rows of the training runs only, and applies it to every row;
2. stacks the reduced band over the stimulus window and shifts it by the HRF
   delay, inside each run (the ``stack`` argument);
3. chooses one penalty per band on held-out runs inside the training runs
   (``choose_penalties``);
4. fits the banded ridge on the training runs and predicts the test runs.

The method follows multi-layer encoding with banded ridge (himalaya,
Dupre la Tour et al. 2022). With one band it is a ridge with a tuned penalty.
"""
from itertools import product

import numpy as np

from brainscore.benchmarks._scoring_utils import (
    kfold_positions,
    valid_prediction_per_unit_pearson,
)

from .projection import FeatureProjection

# The score names its reduction protocol so a reader can see that no test row
# was used to fit it.
PROJECTION_PROTOCOL = 'svd_fitted_on_training_runs_per_fold'


def _alpha_tuples(grid, n_bands, n_candidates, random_state):
    """The penalty tuples to try: the full grid, or a random sample of it.

    With many bands the full grid is too large (5 values and 4 bands give 625
    tuples). A random search, as himalaya's ``solver='random_search'`` does,
    tries ``n_candidates`` tuples: every tuple with one shared penalty, then
    random draws from the grid.
    """
    if n_candidates is None or len(grid) ** n_bands <= n_candidates:
        return list(product(grid, repeat=n_bands))
    rng = np.random.default_rng(random_state)
    tuples = {(alpha,) * n_bands for alpha in grid}
    while len(tuples) < n_candidates:
        tuples.add(tuple(float(a) for a in rng.choice(grid, size=n_bands)))
    return sorted(tuples)


def _penalty_diagonal(widths, penalties):
    return np.concatenate([np.full(width, penalty, dtype=np.float32)
                           for width, penalty in zip(widths, penalties)])


def _solve(X, Y, widths, penalties):
    """Banded ridge weights: one penalty for all columns of each band."""
    XtX = X.T @ X
    XtX[np.diag_indices_from(XtX)] += _penalty_diagonal(widths, penalties)
    return np.linalg.solve(XtX, X.T @ Y)


def choose_penalties(X, Y, run_idx, widths, tuples, *, n_splits, random_state):
    """The penalty tuple with the best mean Pearson r on held-out runs.

    The runs of ``run_idx`` are split ``n_splits`` ways; the first split gives
    the inner training runs and the inner held-out runs. ``X^T X`` is computed
    once; only the penalty diagonal changes from tuple to tuple.
    """
    runs = np.unique(run_idx)
    fit_pos, check_pos = next(kfold_positions(
        len(runs), n_splits=n_splits, random_state=random_state))
    fit = np.isin(run_idx, runs[fit_pos])
    check = np.isin(run_idx, runs[check_pos])
    XtX_base = X[fit].T @ X[fit]
    XtY = X[fit].T @ Y[fit]
    Y_check = Y[check] - Y[check].mean(0)
    Y_check_var = (Y_check ** 2).sum(0)

    best_score, best = -np.inf, None
    for penalties in tuples:
        XtX = XtX_base.copy()
        XtX[np.diag_indices_from(XtX)] += _penalty_diagonal(widths, penalties)
        try:
            W = np.linalg.solve(XtX, XtY)
        except np.linalg.LinAlgError:
            continue
        pred = X[check] @ W
        pred = pred - pred.mean(0)
        num = (Y_check * pred).sum(axis=0)
        den = np.sqrt(Y_check_var * (pred ** 2).sum(0))
        with np.errstate(divide='ignore', invalid='ignore'):
            r = np.where(den > 0, num / den, 0.0)
        score = float(np.mean(r))
        if score > best_score:
            best_score, best = score, penalties
    if best is None:
        raise np.linalg.LinAlgError('No penalty tuple gave a solvable banded ridge.')
    return best


def reduce_and_stack(bands, fit_rows, stack, dim_cap, projections=None):
    """Reduce each band and stack it; return ``(blocks, projections)``.

    Each band gets its own ``FeatureProjection``, fitted on ``bands[b][fit_rows]``
    unless ``projections`` gives fitted ones. The reduction is applied to every
    row, then ``stack`` stacks the stimulus window and shifts by the HRF delay.
    """
    blocks, fitted = [], {}
    for name, (features, feature_ids) in bands.items():
        projection = (projections[name] if projections is not None
                      else FeatureProjection(dim_cap).fit(features[fit_rows], feature_ids))
        fitted[name] = projection
        blocks.append(stack(projection.transform(features, feature_ids)))
    return blocks, fitted


def score_banded(bands, Y, run_idx, keep, stack, *, dim_cap, alpha_grid,
                 n_candidates, n_splits=5, random_state=0):
    """Per-parcel Pearson r of the banded ridge on held-out runs.

    Args:
        bands: ``{band: (features, feature_ids)}``; ``features`` is
            ``(n_TRs, n_units)``, one row per TR of the assembly.
        Y: ``(n_TRs, n_parcels)`` recorded BOLD, same rows.
        run_idx: ``(n_TRs,)`` run id of each row.
        keep: ``(n_TRs,)`` bool; False for the excluded samples at run edges.
        stack: ``(n_TRs, k) -> (n_TRs, window * k)``; stacks within runs.

    Returns ``(per_parcel_r, info)``. ``info`` lists the bands and the
    penalty of each band in each fold.
    """
    names = list(bands)
    tuples = _alpha_tuples(list(alpha_grid), len(names), n_candidates, random_state)
    run_kept = run_idx[keep]
    Y_kept = Y[keep]
    runs = np.unique(run_kept)
    held = np.full(Y_kept.shape, np.nan, dtype=np.float32)
    chosen = []
    for fold, (train_pos, test_pos) in enumerate(kfold_positions(
            len(runs), n_splits=n_splits, random_state=random_state)):
        train_runs, test_runs = runs[train_pos], runs[test_pos]
        # The reduction sees the rows of the training runs only. The test rows
        # are only transformed with it.
        blocks, _ = reduce_and_stack(bands, np.isin(run_idx, train_runs), stack, dim_cap)
        X = np.concatenate(blocks, axis=1)[keep]
        widths = [block.shape[1] for block in blocks]
        train = np.isin(run_kept, train_runs)
        test = np.isin(run_kept, test_runs)
        # The inner choice uses the reduction fitted on all outer training rows,
        # so the inner held-out runs took part in that fit. The outer test runs
        # took part in nothing.
        penalties = choose_penalties(X[train], Y_kept[train], run_kept[train], widths,
                                     tuples, n_splits=n_splits, random_state=fold)
        chosen.append(dict(zip(names, (float(p) for p in penalties))))
        W = _solve(X[train], Y_kept[train], widths, penalties)
        held[test] = (X[test] @ W).astype(np.float32)
    info = {'bands': names, 'alpha_grid': list(alpha_grid),
            'alpha_candidates': len(tuples), 'chosen_alphas_per_fold': chosen}
    return valid_prediction_per_unit_pearson(Y_kept, held), info


def fit_predict_banded(train_bands, Y, run_idx, keep, stack, predict_bands, predict_stack,
                       *, dim_cap, alpha_grid, n_candidates, n_splits=5, random_state=0):
    """Fit on all training rows and predict the rows of ``predict_bands``.

    The reduction is fitted on the training rows, the penalties are chosen on
    held-out training runs, and one banded ridge is fitted on every kept
    training row. Returns ``(predictions, penalties)``.
    """
    names = list(train_bands)
    if list(predict_bands) != names:
        raise ValueError(f'Prediction bands {list(predict_bands)} differ from the '
                         f'training bands {names}.')
    tuples = _alpha_tuples(list(alpha_grid), len(names), n_candidates, random_state)
    blocks, projections = reduce_and_stack(
        train_bands, np.ones(len(run_idx), dtype=bool), stack, dim_cap)
    X = np.concatenate(blocks, axis=1)[keep]
    widths = [block.shape[1] for block in blocks]
    penalties = choose_penalties(X, Y[keep], run_idx[keep], widths, tuples,
                                 n_splits=n_splits, random_state=random_state)
    W = _solve(X, Y[keep], widths, penalties)
    predict_blocks, _ = reduce_and_stack(predict_bands, None, predict_stack, dim_cap,
                                         projections=projections)
    predictions = (np.concatenate(predict_blocks, axis=1) @ W).astype(np.float32)
    return predictions, dict(zip(names, (float(p) for p in penalties)))
