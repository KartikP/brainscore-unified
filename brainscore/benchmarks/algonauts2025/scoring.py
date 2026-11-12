"""Banded ridge for the Algonauts 2025 benchmark.

This module is pure: it needs no model and no fMRI files, so it is tested on
small synthetic arrays.

A band is the features of one model region, one row per scan (TR), in the row
order of the assembly. ``score_banded`` scores the bands with 5-fold
cross-validation over stimuli: every run of one stimulus (a repeated Movie10
clip has two runs) stays on one side of a split. In each outer fold it:

1. prepares each band on the training rows of the fold only
   (``prepare_bands``): it centres the band with the training mean, reduces it
   with a truncated SVD (a PCA, because the band is centred), and standardizes
   the components with the training mean and scale. The same transform is
   applied to every row;
2. stacks the prepared band over the stimulus window and shifts it by the HRF
   delay, inside each run (the ``stack`` argument);
3. chooses, for each parcel, the penalty tuple (one penalty per band) with the
   best Pearson r on held-out stimuli inside the training stimuli
   (``choose_penalties``), as himalaya does;
4. fits a banded ridge with an intercept on the training rows, one solve per
   distinct chosen tuple, and predicts the test rows.

The method follows multi-layer encoding with banded ridge (himalaya,
Dupre la Tour et al. 2022). With one band it is a ridge with an intercept and a
penalty tuned per parcel. All solves are in float64.
"""
from itertools import product

import numpy as np

from brainscore.benchmarks._scoring_utils import (
    kfold_positions,
    valid_prediction_per_unit_pearson,
)

from .projection import FeatureProjection

# The score names its preparation protocol so a reader can see that no test row
# was used to fit it.
PROJECTION_PROTOCOL = 'centred_svd_and_scale_fitted_on_training_stimuli'
PENALTY_PROTOCOL = 'per_parcel_on_one_inner_split_of_held_out_stimuli'


def _alpha_tuples(grid, n_bands, n_candidates, random_state):
    """The penalty tuples to try: the full grid, or a random sample of it.

    With many bands the full grid is too large (11 values and 2 bands give 121
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
    return np.concatenate([np.full(width, penalty, dtype=np.float64)
                           for width, penalty in zip(widths, penalties)])


def _centred(X, Y, rows):
    """``X`` and ``Y`` minus the means of ``rows`` (float64), and those means."""
    x_mean, y_mean = X[rows].mean(0), Y[rows].mean(0)
    return X - x_mean, Y - y_mean, x_mean, y_mean


def _pearson_per_column(Y, P):
    Yc, Pc = Y - Y.mean(0), P - P.mean(0)
    num = (Yc * Pc).sum(0)
    den = np.sqrt((Yc ** 2).sum(0) * (Pc ** 2).sum(0))
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.where(den > 0, num / den, -np.inf)


def choose_penalties(X, Y, groups, widths, tuples, *, n_splits, random_state):
    """For each parcel, the index in ``tuples`` with the best Pearson r on held-out stimuli.

    The stimuli of ``groups`` are split ``n_splits`` ways; the first split
    gives the inner training stimuli and the inner held-out stimuli. The ridge
    has an intercept (centred by the inner training rows). ``X^T X`` is
    computed once; only the penalty diagonal changes from tuple to tuple.
    """
    stimuli = np.unique(groups)
    fit_pos, check_pos = next(kfold_positions(
        len(stimuli), n_splits=n_splits, random_state=random_state))
    fit = np.isin(groups, stimuli[fit_pos])
    check = np.isin(groups, stimuli[check_pos])
    Xc, Yc, _, _ = _centred(X, Y, fit)
    XtX_base = Xc[fit].T @ Xc[fit]
    XtY = Xc[fit].T @ Yc[fit]
    scores = np.full((len(tuples), Y.shape[1]), -np.inf)
    for index, penalties in enumerate(tuples):
        XtX = XtX_base.copy()
        XtX[np.diag_indices_from(XtX)] += _penalty_diagonal(widths, penalties)
        try:
            W = np.linalg.solve(XtX, XtY)
        except np.linalg.LinAlgError:
            continue
        scores[index] = _pearson_per_column(Yc[check], Xc[check] @ W)
    if not np.isfinite(scores).any():
        raise np.linalg.LinAlgError('No penalty tuple gave a solvable banded ridge.')
    return scores.argmax(axis=0)


def prepare_bands(bands, fit_rows, dim_cap, prepared=None):
    """Centre, reduce and standardize each band; return ``(prepared, fitted)``.

    ``fitted[band]`` is ``(mean, projection, component mean, component scale)``,
    fitted on ``bands[band][fit_rows]`` unless ``prepared`` gives fitted ones.
    """
    out, fitted = {}, {}
    for name, (features, feature_ids) in bands.items():
        features = np.asarray(features, dtype=np.float64)
        if prepared is None:
            mean = features[fit_rows].mean(0)
            projection = FeatureProjection(dim_cap).fit(features[fit_rows] - mean, feature_ids)
            components = projection.transform(features[fit_rows] - mean, feature_ids)
            c_mean, c_scale = components.mean(0), components.std(0)
            c_scale = np.where(c_scale > 0, c_scale, 1.0)
        else:
            mean, projection, c_mean, c_scale = prepared[name]
        components = projection.transform(features - mean, feature_ids)
        out[name] = (components - c_mean) / c_scale
        fitted[name] = (mean, projection, c_mean, c_scale)
    return out, fitted


def _design(prepared, keep, stack):
    blocks = [stack(Z) for Z in prepared.values()]
    return np.concatenate(blocks, axis=1)[keep], [block.shape[1] for block in blocks]


def _fit_predict(X, Y, train, X_test, widths, tuples, choice):
    """Ridge with an intercept, one solve per distinct chosen tuple."""
    Xc, Yc, x_mean, y_mean = _centred(X, Y, train)
    XtX_base = Xc[train].T @ Xc[train]
    predictions = np.empty((len(X_test), Y.shape[1]))
    for index in np.unique(choice):
        parcels = np.flatnonzero(choice == index)
        XtX = XtX_base.copy()
        XtX[np.diag_indices_from(XtX)] += _penalty_diagonal(widths, tuples[index])
        W = np.linalg.solve(XtX, Xc[train].T @ Yc[train][:, parcels])
        predictions[:, parcels] = (X_test - x_mean) @ W + y_mean[parcels]
    return predictions


def _choice_summary(choice, tuples, names, grid):
    """What the score records about the per-parcel choice of one fit."""
    edges = (min(grid), max(grid))
    at_edge = np.array([any(p in edges for p in tuples[i]) for i in choice])
    chosen, counts = np.unique(choice, return_counts=True)
    return {'n_distinct_tuples': int(len(chosen)),
            'share_of_parcels_at_grid_edge': float(at_edge.mean()),
            'tuples': [{'penalties': dict(zip(names, (float(p) for p in tuples[i]))),
                        'n_parcels': int(n)} for i, n in zip(chosen, counts)]}


def score_banded(bands, Y, groups, keep, stack, *, dim_cap, alpha_grid,
                 n_candidates, n_splits=5, random_state=0):
    """Per-parcel Pearson r of the banded ridge on held-out stimuli.

    Args:
        bands: ``{band: (features, feature_ids)}``; ``features`` is
            ``(n_TRs, n_units)``, one row per TR of the assembly.
        Y: ``(n_TRs, n_parcels)`` recorded BOLD, same rows.
        groups: ``(n_TRs,)`` stimulus of each row; the folds keep each
            stimulus on one side.
        keep: ``(n_TRs,)`` bool; False for the excluded samples at run edges.
        stack: ``(n_TRs, k) -> (n_TRs, window * k)``; stacks within runs.

    Returns ``(per_parcel_r, info)``.
    """
    names = list(bands)
    tuples = _alpha_tuples(list(alpha_grid), len(names), n_candidates, random_state)
    Y = np.asarray(Y, dtype=np.float64)
    groups_kept, Y_kept = groups[keep], Y[keep]
    stimuli = np.unique(groups_kept)
    held = np.full(Y_kept.shape, np.nan, dtype=np.float32)
    folds, components = [], None
    for fold, (train_pos, test_pos) in enumerate(kfold_positions(
            len(stimuli), n_splits=n_splits, random_state=random_state)):
        train_stimuli, test_stimuli = stimuli[train_pos], stimuli[test_pos]
        # The preparation sees the rows of the training stimuli only. The test
        # rows are only transformed with it.
        prepared, _ = prepare_bands(bands, np.isin(groups, train_stimuli), dim_cap)
        components = {name: int(Z.shape[1]) for name, Z in prepared.items()}
        X, widths = _design(prepared, keep, stack)
        train = np.isin(groups_kept, train_stimuli)
        test = np.isin(groups_kept, test_stimuli)
        # The inner choice uses the preparation fitted on all outer training
        # stimuli, so the inner held-out stimuli took part in that fit. The outer
        # test stimuli took part in nothing.
        choice = choose_penalties(X[train], Y_kept[train], groups_kept[train], widths,
                                  tuples, n_splits=n_splits, random_state=fold)
        held[test] = _fit_predict(X[train], Y_kept[train], np.ones(int(train.sum()), bool),
                                  X[test], widths, tuples, choice)
        folds.append(_choice_summary(choice, tuples, names, alpha_grid))
    info = {'bands': names, 'components_per_band': components,
            'alpha_grid': [float(a) for a in alpha_grid],
            'alpha_candidates': len(tuples), 'penalties_per_fold': folds}
    return valid_prediction_per_unit_pearson(Y_kept, held), info


def fit_predict_banded(train_bands, Y, groups, keep, stack, predict_bands, predict_stack,
                       *, dim_cap, alpha_grid, n_candidates, n_splits=5, random_state=0):
    """Fit on all training rows and predict every row of ``predict_bands``.

    The preparation is fitted on the training rows, the penalties are chosen
    per parcel on held-out training stimuli, and one banded ridge with an
    intercept is fitted on every kept training row. Returns
    ``(predictions, summary)``.
    """
    names = list(train_bands)
    if list(predict_bands) != names:
        raise ValueError(f'Prediction bands {list(predict_bands)} differ from the '
                         f'training bands {names}.')
    tuples = _alpha_tuples(list(alpha_grid), len(names), n_candidates, random_state)
    Y = np.asarray(Y, dtype=np.float64)[keep]
    prepared, fitted = prepare_bands(train_bands, np.ones(len(groups), dtype=bool), dim_cap)
    X, widths = _design(prepared, keep, stack)
    choice = choose_penalties(X, Y, groups[keep], widths, tuples,
                              n_splits=n_splits, random_state=random_state)
    predict_prepared, _ = prepare_bands(predict_bands, None, dim_cap, prepared=fitted)
    X_predict, _ = _design(predict_prepared, np.ones(len(next(iter(predict_prepared.values()))),
                                                     dtype=bool), predict_stack)
    predictions = _fit_predict(X, Y, np.ones(len(X), bool), X_predict, widths, tuples, choice)
    return predictions.astype(np.float32), _choice_summary(choice, tuples, names, alpha_grid)
