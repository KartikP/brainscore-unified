"""Tests for brainscore.benchmarks.algonauts2025.scoring, on small synthetic arrays.

- The preparation of a fold (centring, reduction, scale) is fitted on the
  training stimuli of that fold only.
- The folds keep every run of a repeated stimulus on one side.
- With one band, the banded ridge is scikit-learn's ridge with an intercept, its
  penalty chosen per parcel on held-out stimuli inside the training stimuli.
- Each parcel gets its own penalties: parcels that depend on different bands
  get different tuples and are both predicted well.
- With many bands, a random search tries a fixed number of penalty tuples.
- Held-out prediction uses the preparation and the penalties of the training films.
"""
import numpy as np
import pytest

from brainscore.benchmarks._scoring_utils import (
    kfold_positions,
    valid_prediction_per_unit_pearson,
)
from brainscore.benchmarks.algonauts2025 import scoring
from brainscore.benchmarks.algonauts2025.benchmark import _Algonauts2025Base
from brainscore.benchmarks.algonauts2025.projection import FeatureProjection
from brainscore.benchmarks.algonauts2025.scoring import (
    _alpha_tuples,
    fit_predict_banded,
    score_banded,
)

GRID = _Algonauts2025Base.BANDED_ALPHA_GRID


def _synth(n_stimuli=10, per_stimulus=20, n_feat=12, n_parcels=6, seed=0, offset=5.0):
    rng = np.random.default_rng(seed)
    n = n_stimuli * per_stimulus
    groups = np.repeat([f's{i}' for i in range(n_stimuli)], per_stimulus)
    # Features and BOLD are not centred, so a missing intercept would show.
    X = (rng.standard_normal((n, n_feat)) + offset).astype(np.float32)
    W = rng.standard_normal((n_feat, n_parcels))
    Y = (X @ W + 3.0 * rng.standard_normal((n, n_parcels)) + 100.0).astype(np.float32)
    return X, Y, groups


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


def test_the_preparation_of_a_fold_never_sees_the_rows_of_its_test_stimuli(monkeypatch):
    X, Y, groups = _synth(n_feat=30)
    keep = np.ones(len(groups), dtype=bool)
    monkeypatch.setattr(scoring, 'FeatureProjection', _Recorded)

    _Recorded.fits = []
    score_banded({'a': (X, None)}, Y, groups, keep, _same, dim_cap=4,
                 alpha_grid=GRID, n_candidates=None)
    first = list(_Recorded.fits)
    stimuli = np.unique(groups)
    folds = list(kfold_positions(len(stimuli), n_splits=5, random_state=0))
    assert len(first) == len(folds)
    for (fitted, _), (train_pos, _) in zip(first, folds):
        train = X[np.isin(groups, stimuli[train_pos])].astype(np.float64)
        # Centred with the training mean before the reduction.
        np.testing.assert_allclose(fitted, train - train.mean(0), atol=1e-9)

    # Change the features of the test stimuli of fold 0: the components fitted
    # for fold 0 do not change.
    test_rows = np.isin(groups, stimuli[folds[0][1]])
    changed = X.copy()
    changed[test_rows] = 100.0 * np.random.default_rng(5).standard_normal(
        (int(test_rows.sum()), X.shape[1]))
    _Recorded.fits = []
    score_banded({'a': (changed, None)}, Y, groups, keep, _same, dim_cap=4,
                 alpha_grid=GRID, n_candidates=None)
    np.testing.assert_array_equal(_Recorded.fits[0][1], first[0][1])
    assert not np.array_equal(_Recorded.fits[1][1], first[1][1])   # fold 1 trains on them


def test_both_runs_of_a_repeated_stimulus_stay_on_one_side_of_every_split(monkeypatch):
    X, Y, groups = _synth(n_stimuli=9)
    # Stimulus s0 is shown twice: two runs, one stimulus id, the same features.
    groups = np.concatenate([groups, np.repeat('s0', 20)])
    run = np.concatenate([np.repeat(np.arange(9), 20), np.repeat(9, 20)])
    X = np.concatenate([X, X[:20]])
    Y = np.concatenate([Y, Y[:20]])
    keep = np.ones(len(groups), dtype=bool)
    sides = []

    def recording(X_, Y_, groups_, *args, **kwargs):
        sides.append(set(groups_))
        return real(X_, Y_, groups_, *args, **kwargs)

    real = scoring.choose_penalties
    monkeypatch.setattr(scoring, 'choose_penalties', recording)
    score_banded({'a': (X, None)}, Y, groups, keep, _same, dim_cap=100,
                 alpha_grid=GRID, n_candidates=None)
    assert len(sides) == 5
    for train_stimuli in sides:
        runs_in_train = set(run[np.isin(groups, list(train_stimuli))])
        assert ({0, 9} <= runs_in_train) or not ({0, 9} & runs_in_train)


def _reference_ridge(X, Y, groups, grid):
    """scikit-learn ridge with an intercept on data prepared by the training rows.

    The preparation has no reduction here (the band is narrower than the cap):
    centre with the training mean, then divide by one scale (the root mean
    variance of the columns; a rotation does not change it). The penalty is the
    one with the best mean r over the parcels on the first inner split.
    """
    from sklearn.linear_model import Ridge
    stimuli = np.unique(groups)
    held = np.full(Y.shape, np.nan)
    chosen = []
    for fold, (train_pos, test_pos) in enumerate(kfold_positions(len(stimuli), 5, 0)):
        train_stimuli = stimuli[train_pos]
        train = np.isin(groups, train_stimuli)
        test = np.isin(groups, stimuli[test_pos])
        Z = X.astype(np.float64) - X[train].mean(0)
        Z = Z / np.sqrt(Z[train].var(0).mean())
        fit_pos, check_pos = next(kfold_positions(len(train_stimuli), 5, fold))
        fit = np.isin(groups, train_stimuli[fit_pos])
        check = np.isin(groups, train_stimuli[check_pos])
        r = np.array([valid_prediction_per_unit_pearson(
            Y[check], Ridge(alpha=a).fit(Z[fit], Y[fit]).predict(Z[check])) for a in grid])
        best = np.full(Y.shape[1], np.asarray(grid)[np.nanargmax(np.nanmean(r, axis=1))])
        chosen.append(best)
        for p, alpha in enumerate(best):
            held[test, p] = Ridge(alpha=alpha).fit(Z[train], Y[train, p]).predict(Z[test])
    return valid_prediction_per_unit_pearson(Y, held), chosen


def test_one_band_is_a_ridge_with_an_intercept_and_a_tuned_penalty():
    X, Y, groups = _synth(n_feat=8)
    keep = np.ones(len(groups), dtype=bool)
    got, info = score_banded({'IT': (X, None)}, Y, groups, keep, _same, dim_cap=100,
                             alpha_grid=GRID, n_candidates=60)
    ref, chosen = _reference_ridge(X, Y, groups, GRID)
    assert info['bands'] == ['IT'] and info['components_per_band'] == {'IT': 8}
    assert info['alpha_candidates'] == len(GRID)
    for fold, best in zip(info['penalties_per_fold'], chosen):
        counts = {entry['penalties']['IT']: entry['n_parcels'] for entry in fold['tuples']}
        alphas, n = np.unique(best, return_counts=True)
        assert counts == dict(zip(alphas.tolist(), n.tolist()))
        assert fold['n_distinct_tuples'] == len(alphas)
    np.testing.assert_allclose(got, ref, atol=1e-6)


def test_all_parcels_share_one_penalty_tuple_and_bands_get_their_own_penalty():
    rng = np.random.default_rng(0)
    groups = np.repeat([f's{i}' for i in range(12)], 25)
    A = rng.standard_normal((len(groups), 60))
    B = rng.standard_normal((len(groups), 60))             # band B carries nothing
    Y = (np.stack([A[:, :3].sum(1), A[:, 3:6].sum(1)], axis=1)
         + rng.standard_normal((len(groups), 2)))
    keep = np.ones(len(groups), dtype=bool)
    r, info = score_banded({'A': (A, None), 'B': (B, None)}, Y, groups, keep, _same,
                           dim_cap=100, alpha_grid=GRID, n_candidates=60)
    assert np.all(r > 0.7)
    for fold in info['penalties_per_fold']:
        assert fold['n_distinct_tuples'] == 1 and fold['tuples'][0]['n_parcels'] == 2
        penalties = fold['tuples'][0]['penalties']
        assert penalties['B'] > penalties['A']             # the empty band is shrunk harder


def test_a_band_keeps_the_relative_sizes_of_its_components():
    from brainscore.benchmarks.algonauts2025.scoring import prepare_bands
    rng = np.random.default_rng(0)
    X = rng.standard_normal((200, 6)) * np.array([10.0, 5.0, 2.0, 1.0, 0.5, 0.1])
    prepared, _ = prepare_bands({'A': (X, None)}, np.ones(200, bool), 100)
    spread = prepared['A'].std(0)
    assert spread[0] > 10 * spread[-1]                     # not one size for every component
    np.testing.assert_allclose(np.sqrt((spread ** 2).mean()), 1.0, atol=1e-6)   # one scale for the band


def test_excluded_rows_are_not_scored():
    X, Y, groups = _synth(n_feat=8)
    keep = np.ones(len(groups), dtype=bool)
    keep[::20] = False      # the first TR of every stimulus
    got, _ = score_banded({'IT': (X, None)}, Y, groups, keep, _same, dim_cap=100,
                          alpha_grid=GRID, n_candidates=60)
    # Every kept row is predicted exactly once; the dropped rows play no part.
    ref, _ = _reference_ridge(X[keep], Y[keep], groups[keep], GRID)
    # The preparation is fitted on all training rows, kept or not; the reference on
    # the kept rows only, so the two agree closely but not exactly.
    np.testing.assert_allclose(got, ref, atol=0.02)


def test_many_bands_use_a_random_search_over_the_penalties():
    assert len(_alpha_tuples(GRID, 1, 60, 0)) == 11          # one band: the full grid
    assert len(_alpha_tuples(GRID, 2, None, 0)) == 121       # no limit: the full grid
    tuples = _alpha_tuples(GRID, 4, 60, 0)
    assert len(tuples) == 60 and tuples == _alpha_tuples(GRID, 4, 60, 0)   # sampled, repeatable
    assert all((alpha,) * 4 in tuples for alpha in GRID)     # every shared penalty is tried


def test_prediction_uses_the_training_preparation_and_penalties():
    X, Y, groups = _synth(n_feat=30)
    keep = np.ones(len(groups), dtype=bool)
    new = (np.random.default_rng(3).standard_normal((15, 30)) + 5.0).astype(np.float32)
    ids = [f'u{i}' for i in range(30)]
    preds, summary = fit_predict_banded(
        {'IT': (X, ids)}, Y, groups, keep, _same, {'IT': (new, ids)}, _same,
        dim_cap=5, alpha_grid=GRID, n_candidates=60)
    assert preds.shape == (15, Y.shape[1])
    assert sum(entry['n_parcels'] for entry in summary['tuples']) == Y.shape[1]
    # The same preparation by hand: centre and reduce on the training rows, scale.
    Xd = X.astype(np.float64)
    mean = Xd.mean(0)
    basis = FeatureProjection(5).fit(Xd - mean, ids)
    Z = basis.transform(Xd - mean, ids)
    c_mean, c_scale = Z.mean(0), Z.std(0)
    Zn = (Z - c_mean) / c_scale
    Zp = (basis.transform(new - mean, ids) - c_mean) / c_scale
    from sklearn.linear_model import Ridge
    chosen = [entry['penalties']['IT'] for entry in summary['tuples']]
    candidates = [Ridge(alpha=alpha).fit(Zn, Y).predict(Zp) for alpha in chosen]
    for parcel in range(Y.shape[1]):
        assert any(np.allclose(preds[:, parcel], c[:, parcel], rtol=1e-4, atol=1e-3)
                   for c in candidates)
    with pytest.raises(ValueError, match='identities/order'):
        fit_predict_banded({'IT': (X, ids)}, Y, groups, keep, _same,
                           {'IT': (new, ids[::-1])}, _same,
                           dim_cap=5, alpha_grid=GRID, n_candidates=60)
    with pytest.raises(ValueError, match='differ from the training bands'):
        fit_predict_banded({'IT': (X, ids)}, Y, groups, keep, _same, {'V4': (new, ids)}, _same,
                           dim_cap=5, alpha_grid=GRID, n_candidates=60)


def test_a_constant_parcel_does_not_stop_the_choice_of_penalties():
    X, Y, groups = _synth(n_parcels=1)
    both = np.concatenate([Y, np.full_like(Y, 7.0)], axis=1)   # a constant parcel has no correlation
    widths, tuples = [X.shape[1]], [(a,) for a in GRID]
    kwargs = dict(n_splits=5, random_state=0)
    alone = scoring.choose_penalties(X, Y, groups, widths, tuples, **kwargs)
    with_constant = scoring.choose_penalties(X, both, groups, widths, tuples, **kwargs)
    assert list(with_constant) == [alone[0]] * 2
