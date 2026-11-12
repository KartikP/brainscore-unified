"""Tests for brainscore.benchmarks.algonauts2025.scoring, on small synthetic arrays.

- The reduction of a fold is fitted on the training runs of that fold only.
- With one band, the banded ridge is a ridge whose penalty is chosen on held-out
  runs inside the training runs.
- With many bands, a random search tries a fixed number of penalty tuples.
- Held-out prediction uses the basis and the penalties of the training films.
"""
import numpy as np
import pytest

from brainscore.benchmarks._scoring_utils import (
    kfold_positions,
    valid_prediction_per_unit_pearson,
)
from brainscore.benchmarks.algonauts2025 import scoring
from brainscore.benchmarks.algonauts2025.projection import FeatureProjection
from brainscore.benchmarks.algonauts2025.scoring import (
    _alpha_tuples,
    fit_predict_banded,
    score_banded,
)

GRID = (1.0, 10.0, 100.0, 1000.0, 10000.0)


def _synth(n_runs=10, per_run=20, n_feat=12, n_parcels=6, seed=0):
    rng = np.random.default_rng(seed)
    n = n_runs * per_run
    run_idx = np.repeat(np.arange(n_runs), per_run)
    X = rng.standard_normal((n, n_feat)).astype(np.float32)
    W = rng.standard_normal((n_feat, n_parcels))
    Y = (X @ W + 2.0 * rng.standard_normal((n, n_parcels))).astype(np.float32)
    return X, Y, run_idx


def _same(X):
    return X


class _Recorded(FeatureProjection):
    """Records what each fit saw: its rows and its components."""

    fits = []

    def fit(self, features, feature_ids=None):
        super().fit(features, feature_ids)
        _Recorded.fits.append((features.copy(), None if self.svd_ is None
                               else self.svd_.components_.copy()))
        return self


def test_the_reduction_of_a_fold_never_sees_the_rows_of_its_test_runs(monkeypatch):
    X, Y, run_idx = _synth(n_feat=30)
    keep = np.ones(len(run_idx), dtype=bool)
    monkeypatch.setattr(scoring, 'FeatureProjection', _Recorded)

    _Recorded.fits = []
    score_banded({'a': (X, None)}, Y, run_idx, keep, _same, dim_cap=4,
                 alpha_grid=GRID, n_candidates=None)
    first = list(_Recorded.fits)
    runs = np.unique(run_idx)
    folds = list(kfold_positions(len(runs), n_splits=5, random_state=0))
    assert len(first) == len(folds)
    for (fitted, _), (train_pos, _) in zip(first, folds):
        np.testing.assert_array_equal(fitted, X[np.isin(run_idx, runs[train_pos])])

    # Change the features of the test runs of fold 0: the components fitted for
    # fold 0 do not change.
    test_rows = np.isin(run_idx, runs[folds[0][1]])
    changed = X.copy()
    changed[test_rows] = 100.0 * np.random.default_rng(5).standard_normal(
        (int(test_rows.sum()), X.shape[1]))
    _Recorded.fits = []
    score_banded({'a': (changed, None)}, Y, run_idx, keep, _same, dim_cap=4,
                 alpha_grid=GRID, n_candidates=None)
    np.testing.assert_array_equal(_Recorded.fits[0][1], first[0][1])
    assert not np.array_equal(_Recorded.fits[1][1], first[1][1])   # fold 1 trains on them


def _reference_ridge(X, Y, run_idx, grid):
    """Ridge without intercept; the penalty is chosen on the first inner split."""
    from sklearn.linear_model import Ridge
    runs = np.unique(run_idx)
    held = np.full(Y.shape, np.nan, dtype=np.float32)
    chosen = []
    for fold, (train_pos, test_pos) in enumerate(kfold_positions(len(runs), 5, 0)):
        train_runs = runs[train_pos]
        train = np.isin(run_idx, train_runs)
        test = np.isin(run_idx, runs[test_pos])
        fit_pos, check_pos = next(kfold_positions(len(train_runs), 5, fold))
        fit = np.isin(run_idx, train_runs[fit_pos])
        check = np.isin(run_idx, train_runs[check_pos])
        best, best_alpha = -np.inf, None
        for alpha in grid:
            pred = Ridge(alpha=alpha, fit_intercept=False).fit(X[fit], Y[fit]).predict(X[check])
            r = float(np.mean(np.nan_to_num(
                valid_prediction_per_unit_pearson(Y[check], pred))))
            if r > best:
                best, best_alpha = r, alpha
        chosen.append(best_alpha)
        ridge = Ridge(alpha=best_alpha, fit_intercept=False).fit(X[train], Y[train])
        held[test] = ridge.predict(X[test])
    return valid_prediction_per_unit_pearson(Y, held), chosen


def test_one_band_is_a_ridge_with_the_penalty_chosen_on_inner_held_out_runs():
    X, Y, run_idx = _synth(n_feat=8)
    keep = np.ones(len(run_idx), dtype=bool)
    got, info = score_banded({'IT': (X, None)}, Y, run_idx, keep, _same, dim_cap=100,
                             alpha_grid=GRID, n_candidates=60)
    ref, chosen = _reference_ridge(X, Y, run_idx, GRID)
    assert info['bands'] == ['IT']
    assert [fold['IT'] for fold in info['chosen_alphas_per_fold']] == chosen
    assert len(set(chosen)) >= 1 and info['alpha_candidates'] == len(GRID)
    np.testing.assert_allclose(got, ref, atol=1e-4)


def test_excluded_rows_are_not_scored_and_runs_stay_whole():
    X, Y, run_idx = _synth(n_feat=8)
    keep = np.ones(len(run_idx), dtype=bool)
    keep[::20] = False      # the first TR of every run
    got, _ = score_banded({'IT': (X, None)}, Y, run_idx, keep, _same, dim_cap=100,
                          alpha_grid=GRID, n_candidates=60)
    ref, _ = _reference_ridge(X[keep], Y[keep], run_idx[keep], GRID)
    np.testing.assert_allclose(got, ref, atol=1e-4)


def test_two_bands_get_their_own_penalties():
    Xv, Y, run_idx = _synth(n_feat=8, seed=1)
    noise = np.random.default_rng(9).standard_normal((len(run_idx), 40)).astype(np.float32)
    keep = np.ones(len(run_idx), dtype=bool)
    per_parcel, info = score_banded({'IT': (Xv, None), 'A1': (noise, None)}, Y, run_idx, keep,
                                    _same, dim_cap=10, alpha_grid=GRID, n_candidates=60)
    assert info['bands'] == ['IT', 'A1']
    assert info['alpha_candidates'] == 25      # the full grid is small enough
    assert all(set(fold) == {'IT', 'A1'} for fold in info['chosen_alphas_per_fold'])
    # The noise band is shrunk harder than the band that carries the signal.
    assert all(fold['A1'] >= fold['IT'] for fold in info['chosen_alphas_per_fold'])
    assert per_parcel.shape == (Y.shape[1],) and np.median(per_parcel) > 0.3


def test_many_bands_use_a_random_search_over_the_penalties():
    assert len(_alpha_tuples(GRID, 2, None, 0)) == 25          # the full grid
    assert len(_alpha_tuples(GRID, 2, 60, 0)) == 25            # small enough: still the full grid
    tuples = _alpha_tuples(GRID, 4, 60, 0)
    assert len(tuples) == 60 and tuples == _alpha_tuples(GRID, 4, 60, 0)   # sampled, repeatable
    assert all((alpha,) * 4 in tuples for alpha in GRID)       # every shared penalty is tried


def test_prediction_uses_the_training_basis_and_penalties():
    X, Y, run_idx = _synth(n_feat=30)
    keep = np.ones(len(run_idx), dtype=bool)
    new = np.random.default_rng(3).standard_normal((15, 30)).astype(np.float32)
    ids = [f'u{i}' for i in range(30)]
    preds, penalties = fit_predict_banded(
        {'IT': (X, ids)}, Y, run_idx, keep, _same, {'IT': (new, ids)}, _same,
        dim_cap=5, alpha_grid=GRID, n_candidates=60)
    basis = FeatureProjection(5).fit(X, ids)
    Z = basis.transform(X, ids)
    W = scoring._solve(Z, Y, [5], [penalties['IT']])
    np.testing.assert_allclose(preds, basis.transform(new, ids) @ W, rtol=1e-4, atol=1e-4)
    with pytest.raises(ValueError, match='identities/order'):
        fit_predict_banded({'IT': (X, ids)}, Y, run_idx, keep, _same,
                           {'IT': (new, ids[::-1])}, _same,
                           dim_cap=5, alpha_grid=GRID, n_candidates=60)
    with pytest.raises(ValueError, match='differ from the training bands'):
        fit_predict_banded({'IT': (X, ids)}, Y, run_idx, keep, _same, {'V4': (new, ids)}, _same,
                           dim_cap=5, alpha_grid=GRID, n_candidates=60)
