"""Banded ridge for the Algonauts 2025 benchmark.

This module is pure: it needs no model and no fMRI files, so it is tested on
small synthetic arrays.

A band is the features of one model region, one row per scan (TR), in the row
order of the assembly. ``score_banded`` scores the bands with 5-fold
cross-validation over stimuli: every run of one stimulus (a repeated Movie10
clip has two runs) stays on one side of a split. In each outer fold it:

1. prepares each band on the training rows of the fold only
   (``prepare_bands``): it centres the band with the training mean, reduces it
   with a truncated SVD (a PCA, because the band is centred), and divides the
   band by one scale (the root mean variance of its components on the training
   rows), so the penalties of different bands compare. The components keep
   their relative sizes: giving every component the same size makes the weak
   components, which are mostly noise, count as much as the strong ones. The
   same transform is applied to every row;
2. stacks the prepared band over the stimulus window and shifts it by the HRF
   delay, inside each run (the ``stack`` argument);
3. chooses the penalty tuple (one penalty per band) with the best mean Pearson
   r over the parcels on held-out stimuli inside the training stimuli (the
   first of 5 folds over the training stimuli). All parcels share the tuple: a
   choice for each parcel on one inner split is noisy and scored lower on real
   data. A parcel without a correlation (a constant one) is left out of the
   mean;
4. fits a banded ridge with an intercept on the training rows and predicts the
   test rows.

Steps 3 and 4 are ``brainscore.metrics.encoding``: ``_banded_fit`` gives its
settings. The method follows multi-layer encoding with banded ridge (himalaya,
Dupre la Tour et al. 2022). With one band it is a ridge with an intercept and a
tuned penalty. All solves are in float64.
"""
from functools import partial

import numpy as np

from brainscore.metrics.encoding import (
    NormalEquations,
    grid_search,
    group_kfold,
    held_out_r,
    penalty_grid,
    ridge,
    row_kfold,
)

from .projection import FeatureProjection

# The score names its preparation protocol so a reader can see that no test row
# was used to fit it.
PROJECTION_PROTOCOL = 'centred_svd_and_scale_fitted_on_training_stimuli'
PENALTY_PROTOCOL = 'one_tuple_for_all_parcels_on_one_inner_split_of_held_out_stimuli'


def _banded_fit(names, alpha_grid, n_candidates, n_splits, random_state, inner_random_state):
    """The banded ridge of one fit: an intercept, one penalty tuple for all parcels.

    The tuple is chosen on the first of ``n_splits`` folds over the training
    stimuli (seed ``inner_random_state``), from ``penalty_grid``.
    """
    tuples = penalty_grid(alpha_grid, len(names), n_candidates, random_state)
    split = partial(group_kfold, n_splits=n_splits, random_state=inner_random_state,
                    n_used=1)
    return ridge(grid_search(tuples, split, undefined_r='skip'), NormalEquations())


def prepare_bands(bands, fit_rows, dim_cap, prepared=None):
    """Centre, reduce and standardize each band; return ``(prepared, fitted)``.

    ``fitted[band]`` is ``(mean, projection, component mean, band scale)``,
    fitted on ``bands[band][fit_rows]`` unless ``prepared`` gives fitted ones.
    """
    out, fitted = {}, {}
    for name, (features, feature_ids) in bands.items():
        features = np.asarray(features, dtype=np.float64)
        if prepared is None:
            mean = features[fit_rows].mean(0)
            projection = FeatureProjection(dim_cap).fit(features[fit_rows] - mean, feature_ids)
            components = projection.transform(features[fit_rows] - mean, feature_ids)
            # One scale for the band, not one for each component.
            c_mean = components.mean(0)
            c_scale = float(np.sqrt(components.var(0).mean())) or 1.0
        else:
            mean, projection, c_mean, c_scale = prepared[name]
        components = projection.transform(features - mean, feature_ids)
        out[name] = (components - c_mean) / c_scale
        fitted[name] = (mean, projection, c_mean, c_scale)
    return out, fitted


def _design(prepared, keep, stack):
    """One stacked band per prepared band, on the kept rows."""
    return [stack(Z)[keep] for Z in prepared.values()]


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
    Y = np.asarray(Y, dtype=np.float64)
    groups_kept, Y_kept = groups[keep], Y[keep]
    stimuli = np.unique(groups_kept)
    held = np.full(Y_kept.shape, np.nan, dtype=np.float32)
    folds, components, chosen = [], None, None
    for fold, (train_pos, test_pos) in enumerate(row_kfold(
            len(stimuli), n_splits=n_splits, random_state=random_state)):
        train_stimuli, test_stimuli = stimuli[train_pos], stimuli[test_pos]
        # The preparation sees the rows of the training stimuli only. The test
        # rows are only transformed with it.
        prepared, _ = prepare_bands(bands, np.isin(groups, train_stimuli), dim_cap)
        components = {name: int(Z.shape[1]) for name, Z in prepared.items()}
        design = _design(prepared, keep, stack)
        train = np.isin(groups_kept, train_stimuli)
        test = np.isin(groups_kept, test_stimuli)
        # The inner choice uses the preparation fitted on all outer training
        # stimuli, so the inner held-out stimuli took part in that fit. The outer
        # test stimuli took part in nothing.
        fit = _banded_fit(names, alpha_grid, n_candidates, n_splits, random_state,
                          inner_random_state=fold)
        held[test], chosen = fit([band[train] for band in design], Y_kept[train],
                                 [band[test] for band in design], groups_kept[train])
        folds.append(_choice_summary(chosen.index, chosen.candidates, names, alpha_grid))
    info = {'bands': names, 'components_per_band': components,
            'alpha_grid': [float(a) for a in alpha_grid],
            'alpha_candidates': len(chosen.candidates), 'penalties_per_fold': folds}
    return held_out_r(Y_kept, held), info


def fit_predict_banded(train_bands, Y, groups, keep, stack, predict_bands, predict_stack,
                       *, dim_cap, alpha_grid, n_candidates, n_splits=5, random_state=0):
    """Fit on all training rows and predict every row of ``predict_bands``.

    The preparation is fitted on the training rows, the penalties are chosen
    on held-out training stimuli, and one banded ridge with an
    intercept is fitted on every kept training row. Returns
    ``(predictions, summary)``.
    """
    names = list(train_bands)
    if list(predict_bands) != names:
        raise ValueError(f'Prediction bands {list(predict_bands)} differ from the '
                         f'training bands {names}.')
    Y = np.asarray(Y, dtype=np.float64)[keep]
    prepared, fitted = prepare_bands(train_bands, np.ones(len(groups), dtype=bool), dim_cap)
    design = _design(prepared, keep, stack)
    predict_prepared, _ = prepare_bands(predict_bands, None, dim_cap, prepared=fitted)
    predict_design = _design(predict_prepared, np.ones(
        len(next(iter(predict_prepared.values()))), dtype=bool), predict_stack)
    fit = _banded_fit(names, alpha_grid, n_candidates, n_splits, random_state,
                      inner_random_state=random_state)
    predictions, chosen = fit(design, Y, predict_design, groups[keep])
    return (predictions.astype(np.float32),
            _choice_summary(chosen.index, chosen.candidates, names, alpha_grid))
