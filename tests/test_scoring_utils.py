"""Shared benchmark scoring utility tests."""

import numpy as np

from brainscore.benchmarks._scoring_utils import (
    kfold_positions,
    kfold_ridge_predictions,
    masked_ridge_predictions,
    pearson_summary,
    run_kfold_masks,
    summed_group_kfold_ridge_predictions,
    valid_prediction_per_unit_pearson,
)


def _reference_kfold_ridge_predictions(X, Y, alpha=1.0):
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import KFold

    held = np.zeros_like(Y)
    for tr, te in KFold(n_splits=5, shuffle=True,
                        random_state=0).split(np.arange(X.shape[0])):
        held[te] = Ridge(alpha=alpha).fit(X[tr], Y[tr]).predict(X[te])
    return held


def test_kfold_positions_match_sklearn_order():
    from sklearn.model_selection import KFold

    got = list(kfold_positions(12, n_splits=4, random_state=7))
    expected = list(KFold(n_splits=4, shuffle=True,
                          random_state=7).split(np.arange(12)))
    for (g_tr, g_te), (e_tr, e_te) in zip(got, expected):
        np.testing.assert_array_equal(g_tr, e_tr)
        np.testing.assert_array_equal(g_te, e_te)


def test_run_kfold_masks_keep_runs_together():
    run_idx = np.repeat(np.arange(6), 3)
    for train_mask, test_mask in run_kfold_masks(
            run_idx, n_splits=3, random_state=1):
        assert not set(run_idx[train_mask]) & set(run_idx[test_mask])
        assert train_mask.dtype == bool and test_mask.dtype == bool


def test_kfold_ridge_predictions_matches_reference():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((30, 5))
    Y = rng.standard_normal((30, 4))
    got = kfold_ridge_predictions(X, Y, alpha=3.0, dtype=None)
    ref = _reference_kfold_ridge_predictions(X, Y, alpha=3.0)
    np.testing.assert_array_equal(got, ref)


def test_masked_ridge_predictions_matches_reference_masks():
    rng = np.random.default_rng(1)
    X = rng.standard_normal((24, 4))
    Y = rng.standard_normal((24, 3))
    run_idx = np.repeat(np.arange(8), 3)
    masks = list(run_kfold_masks(run_idx, n_splits=4, random_state=0))

    got = masked_ridge_predictions(X, Y, masks, alpha=2.0, dtype=None)

    from sklearn.linear_model import Ridge
    ref = np.full_like(Y, np.nan)
    for tr, te in masks:
        ref[te] = Ridge(alpha=2.0).fit(X[tr], Y[tr]).predict(X[te])
    np.testing.assert_array_equal(got, ref)


def test_summed_group_kfold_ridge_predictions_matches_reference_sum():
    rng = np.random.default_rng(2)
    groups = [rng.standard_normal((30, 3)), rng.standard_normal((30, 2))]
    Y = rng.standard_normal((30, 4))
    got = summed_group_kfold_ridge_predictions(
        groups, Y, alpha=1.5, dtype=None)
    ref = sum(_reference_kfold_ridge_predictions(g, Y, alpha=1.5)
              for g in groups)
    np.testing.assert_array_equal(got, ref)


def test_valid_prediction_per_unit_pearson_drops_unpredicted_rows():
    rng = np.random.default_rng(3)
    Y = rng.standard_normal((20, 4))
    pred = Y.copy()
    pred[:5] = np.nan
    r = valid_prediction_per_unit_pearson(Y, pred)
    np.testing.assert_allclose(r, np.ones(4))


def test_pearson_summary_drops_nan_units():
    Y = np.array([[1., 0.], [2., 0.], [3., 0.]])
    pred = np.array([[1., 1.], [2., 1.], [3., 1.]])
    per_unit, median_r, mean_r = pearson_summary(Y, pred)
    np.testing.assert_allclose(per_unit, [1.0])
    assert median_r == 1.0
    assert mean_r == 1.0
