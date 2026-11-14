"""Tests for ``brainscore.metrics.encoding``, the one encoding fit.

Three kinds of test:

- correctness: each solver setting against scikit-learn or the textbook
  closed form, on small data;
- no leak: the rows of a held-out split never reach the fit;
- identity: every benchmark family gives exactly the numbers of the code it
  replaced (``tests/_encoding_reference.py``, copied from commit 97fce99), on
  synthetic data of realistic shape. Equality is ``np.array_equal``, never
  ``allclose``, and the chosen penalties must match too.
"""
from functools import partial

import numpy as np
import pytest

from tests import _encoding_reference as old
from brainscore_core.metrics import per_unit_pearson
from brainscore.metrics.encoding import (
    NormalEquations,
    SklearnRidge,
    cross_validated_r,
    fixed,
    grid_search,
    group_kfold,
    held_out_r,
    penalty_grid,
    random_group_split,
    ridge,
    row_kfold,
    summarize,
)

pytestmark = pytest.mark.unit


def _data(n=120, p=6, k=4, seed=0, offset=3.0, dtype=np.float64):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, p)) + offset
    Y = X @ rng.standard_normal((p, k)) + rng.standard_normal((n, k)) + 10.0
    return X.astype(dtype), Y.astype(dtype)


def _same(a, b):
    """Bit-for-bit equal arrays (NaN equal to NaN), same shape."""
    a, b = np.asarray(a), np.asarray(b)
    assert a.shape == b.shape
    assert np.array_equal(a, b, equal_nan=True)


# -- Correctness -------------------------------------------------------------

class TestSolvers:

    def test_sklearn_ridge_is_sklearn(self):
        from sklearn.linear_model import Ridge
        X, Y = _data()
        pred, chosen = ridge(2.0, SklearnRidge())(X[:90], Y[:90], X[90:])
        _same(pred, Ridge(alpha=2.0).fit(X[:90], Y[:90]).predict(X[90:]))
        assert chosen.shared == (2.0,)

    def test_intercept_matches_sklearn(self):
        from sklearn.linear_model import Ridge
        X, Y = _data()
        pred, _ = ridge(5.0, NormalEquations(centre='both'))(X[:90], Y[:90], X[90:])
        expected = Ridge(alpha=5.0).fit(X[:90], Y[:90]).predict(X[90:])
        np.testing.assert_allclose(pred, expected, rtol=1e-9, atol=1e-9)

    @pytest.mark.parametrize('centre', ['none', 'features'])
    def test_no_intercept_matches_sklearn(self, centre):
        from sklearn.linear_model import Ridge
        X, Y = _data()
        X_train, X_test = X[:90], X[90:]
        if centre == 'features':
            mean = X_train.mean(0)
            X_train, X_test = X_train - mean, X_test - mean
        pred, _ = ridge(5.0, NormalEquations(centre=centre))(X[:90], Y[:90], X[90:])
        expected = Ridge(alpha=5.0, fit_intercept=False).fit(X_train, Y[:90]).predict(X_test)
        np.testing.assert_allclose(pred, expected, rtol=1e-9, atol=1e-9)

    def test_banded_penalty_is_the_textbook_solution(self):
        X, Y = _data(p=7)
        bands = [X[:, :3], X[:, 3:]]
        lam = np.array([1.0] * 3 + [100.0] * 4)
        pred, _ = ridge(grid_search([(1.0, 100.0)], random_group_split),
                        NormalEquations(centre='none'))(
            [b[:90] for b in bands], Y[:90], [b[90:] for b in bands])
        expected = X[90:] @ np.linalg.solve(X[:90].T @ X[:90] + np.diag(lam),
                                            X[:90].T @ Y[:90])
        np.testing.assert_allclose(pred, expected, rtol=1e-9, atol=1e-9)

    @pytest.mark.parametrize('n,p', [(200, 50), (50, 200), (80, 80)])
    def test_shape_aware_is_the_same_estimator(self, n, p):
        X, Y = _data(n=n + 15, p=p, k=40)
        plain = ridge(3.0, NormalEquations(centre='features'))(X[:n], Y[:n], X[n:])[0]
        aware = ridge(3.0, NormalEquations(centre='features', shape_aware=True))(
            X[:n], Y[:n], X[n:])[0]
        np.testing.assert_allclose(aware, plain, rtol=1e-8, atol=1e-8)

    def test_wide_features_take_the_dual_form(self, monkeypatch):
        """Guard the cost: no (p, p) solve when p is much larger than n."""
        X, Y = _data(n=45, p=500, k=3)
        seen, original = [], np.linalg.solve
        monkeypatch.setattr(np.linalg, 'solve',
                            lambda a, b: (seen.append(a.shape), original(a, b))[1])
        ridge(1.0, NormalEquations(centre='features', shape_aware=True))(X[:40], Y[:40], X[40:])
        assert seen == [(40, 40)]

    def test_solve_dtype(self):
        X, Y = _data(dtype=np.float32)
        kept = ridge(1.0, NormalEquations(centre='features', solve_dtype=None))(X, Y, X)[0]
        upcast = ridge(1.0, NormalEquations(centre='features'))(X, Y, X)[0]
        assert kept.dtype == np.float32 and upcast.dtype == np.float64

    def test_sklearn_refuses_a_search(self):
        X, Y = _data()
        with pytest.raises(ValueError, match='one fixed penalty'):
            ridge(grid_search([(1.0,), (10.0,)], random_group_split), SklearnRidge())(X, Y, X)

    def test_unknown_centre(self):
        with pytest.raises(ValueError, match='centre'):
            NormalEquations(centre='targets')


class TestChoosers:

    def test_grid_search_picks_the_best_candidate(self):
        """The chosen penalty is the one with the best mean r on the check rows."""
        from sklearn.linear_model import Ridge
        rng = np.random.default_rng(1)
        X = rng.standard_normal((150, 30))
        Y = X[:, :2] @ rng.standard_normal((2, 5)) + 2.0 * rng.standard_normal((150, 5))
        grid = [(a,) for a in (1e-3, 1.0, 1e2, 1e4)]
        split = partial(random_group_split, fraction=0.3, seed=4)
        chosen = grid_search(grid, split)(NormalEquations(), [X], Y, None)
        (fit_rows, check_rows), = split(np.arange(150))
        means = [np.mean(per_unit_pearson(
            Y[check_rows], Ridge(alpha=a).fit(X[fit_rows], Y[fit_rows]).predict(X[check_rows])))
            for (a,) in grid]
        assert chosen.shared == grid[int(np.argmax(means))]

    def test_per_target_choice(self):
        """A noisy target takes a stronger penalty than a clean one."""
        rng = np.random.default_rng(2)
        groups = np.repeat(np.arange(10), 20)
        X = rng.standard_normal((200, 40))
        clean = X[:, :5].sum(1)
        noisy = 0.2 * X[:, 5] + 5.0 * rng.standard_normal(200)
        Y = np.stack([clean, noisy], axis=1)
        grid = [(a,) for a in (1e-2, 1e1, 1e4)]
        chosen = grid_search(grid, partial(group_kfold, n_splits=4), per_target=True)(
            NormalEquations(), [X], Y, groups)
        alphas = chosen.per_target()[:, 0]
        assert alphas[1] > alphas[0]
        with pytest.raises(ValueError, match='not one'):
            chosen.shared

    def test_constant_target_zero_or_skip(self):
        X, Y = _data(k=1)
        both = np.concatenate([Y, np.full_like(Y, 7.0)], axis=1)
        grid = [(a,) for a in (1e-2, 1.0, 1e2, 1e4)]
        for undefined in ('zero', 'skip'):
            choose = grid_search(grid, random_group_split, undefined_r=undefined)
            alone = choose(NormalEquations(), [X], Y, None).shared
            assert choose(NormalEquations(), [X], both, None).shared == alone

    def test_fixed(self):
        chosen = fixed(3.0)(None, [np.zeros((4, 2)), np.zeros((4, 1))], np.zeros((4, 5)), None)
        assert chosen.shared == (3.0, 3.0)
        _same(chosen.per_target(), np.full((5, 2), 3.0))

    def test_penalty_grid(self):
        grid = [10.0 ** e for e in range(-2, 9)]
        assert len(penalty_grid(grid, 1, 60)) == 11
        assert penalty_grid(grid, 2) == list(__import__('itertools').product(grid, repeat=2))
        tuples = penalty_grid(grid, 4, 60, 0)
        assert len(tuples) == 60 and tuples == penalty_grid(grid, 4, 60, 0)
        assert all((a,) * 4 in tuples for a in grid)


class TestSplits:

    def test_group_kfold_keeps_groups_together_and_covers_each_row_once(self):
        groups = np.repeat([f's{i}' for i in range(10)], 7)
        counts = np.zeros(len(groups), dtype=int)
        for train, test in group_kfold(groups, n_splits=5):
            assert not set(groups[train]) & set(groups[test])
            counts += test
        assert (counts == 1).all()
        assert len(list(group_kfold(groups, n_splits=5, n_used=1))) == 1

    def test_row_kfold_is_sklearn(self):
        from sklearn.model_selection import KFold
        expected = KFold(4, shuffle=True, random_state=7).split(np.arange(12))
        for (a, b), (c, d) in zip(row_kfold(12, 4, 7), expected):
            _same(a, c)
            _same(b, d)

    def test_random_group_split_holds_out_whole_groups(self):
        groups = np.repeat(np.arange(8), 5)
        (fit, check), = random_group_split(groups, fraction=0.25, seed=3)
        assert len(set(groups[check])) == 2
        assert not set(groups[fit]) & set(groups[check])


class TestNoLeak:

    def test_held_out_targets_never_reach_the_fit(self):
        X, Y = _data(n=100)
        groups = np.repeat(np.arange(10), 10)
        folds = list(group_kfold(groups, n_splits=5))
        fit = ridge(grid_search([(1.0,), (100.0,)], partial(group_kfold, n_splits=4)),
                    NormalEquations())

        def run(targets):
            seen = []

            def capturing(*args):
                predictions, chosen = fit(*args)
                seen.append(predictions)
                return predictions, chosen
            cross_validated_r(X, targets, folds, capturing, groups=groups)
            return seen

        changed = Y.copy()
        changed[folds[0][1]] = 1e6 * np.random.default_rng(9).standard_normal(
            (int(folds[0][1].sum()), Y.shape[1]))
        _same(run(Y)[0], run(changed)[0])

    def test_the_inner_split_sees_only_training_groups(self):
        X, Y = _data(n=100)
        groups = np.repeat(np.arange(10), 10)
        seen = []

        def split(train_groups):
            pairs = list(group_kfold(train_groups, n_splits=4))
            for fit_rows, check_rows in pairs:
                assert not set(train_groups[fit_rows]) & set(train_groups[check_rows])
            seen.append(set(train_groups))
            return pairs

        folds = list(group_kfold(groups, n_splits=5))
        cross_validated_r(X, Y, folds, ridge(grid_search([(1.0,)], split), NormalEquations()),
                          groups=groups)
        for (train, test), train_groups in zip(folds, seen):
            assert train_groups == set(groups[train])
            assert not train_groups & set(groups[test])


class TestScores:

    def test_rows_without_a_prediction_are_left_out(self):
        rng = np.random.default_rng(3)
        Y = rng.standard_normal((20, 4))
        P = Y.copy()
        P[:5] = np.nan
        np.testing.assert_allclose(held_out_r(Y, P), np.ones(4))
        assert np.isnan(held_out_r(Y, np.full_like(Y, np.nan))).all()

    def test_summarize_drops_nan(self):
        finite, median, mean = summarize(np.array([1.0, np.nan, 0.5]))
        _same(finite, [1.0, 0.5])
        assert (median, mean) == (0.75, 0.75)


# -- Identity with the replaced code -----------------------------------------

def _neural(n, k, seed, fortran=False, dtype=np.float64):
    rng = np.random.default_rng(seed)
    Y = rng.standard_normal((n, k)).astype(dtype)
    return np.asfortranarray(Y) if fortran else Y


class TestIdentityLahnerGLM:
    """``Lahner2024-fMRI-naturalistic`` and ``-visualROI``: scikit-learn ridge,
    penalty 1, 5 shuffled folds over videos."""

    @pytest.mark.parametrize('fortran', [False, True])
    @pytest.mark.parametrize('dtype', [np.float32, np.float64])
    def test_kfold_ridge(self, fortran, dtype):
        rng = np.random.default_rng(0)
        features = rng.standard_normal((205, 48)).astype(dtype)
        neural = _neural(205, 700, 1, fortran=fortran, dtype=dtype)
        # The whole-cortex variant takes the transposed assembly as it is (F order).
        neural[:, 3] = 2.0                       # a constant voxel
        ref_r, ref_median, ref_mean = old.lahner_glm_score(features, neural)
        r, _ = cross_validated_r(features, neural, row_kfold(len(features), n_splits=5,
                                                             random_state=0),
                                 ridge(1.0, SklearnRidge()))
        finite, median, mean = summarize(r)
        _same(finite, ref_r)
        assert (median, mean) == (ref_median, ref_mean)

    def test_split_half_pearson(self):
        rng = np.random.default_rng(2)
        A, B = rng.standard_normal((300, 500)), rng.standard_normal((300, 500))
        A[:, 7] = 1.0
        _same(np.nan_to_num(per_unit_pearson(A, B), nan=0.0), old.lahner_split_half_pearson(A, B))


def _tr_data(n_runs=20, per_run=60, n_feat=48, n_vox=4500, n_participants=4, seed=4):
    rng = np.random.default_rng(seed)
    n = n_runs * per_run
    X = rng.standard_normal((n, n_feat)).astype(np.float32)
    Y = (X[:, :4] @ rng.standard_normal((4, n_vox))
         + 3.0 * rng.standard_normal((n, n_vox))).astype(np.float32)
    run_idx = np.repeat(np.arange(n_runs), per_run).astype(np.int32)
    participants = np.repeat([f'p{i}' for i in range(n_participants)],
                             n // n_participants).astype(object)
    return X, Y, run_idx, participants


class TestIdentityLahnerTimeResolved:
    """``Lahner2024-fMRI-naturalistic-timeresolved{,-visualROI,-improved,
    -improved-visualROI}``: scikit-learn ridge at penalty 1, voxel batches of 2000,
    through the two fold rules."""

    def test_run_blocks(self):
        from brainscore.benchmarks.lahner2024 import RunBlocks
        X, Y, run_idx, participants = _tr_data()
        new = RunBlocks(4).per_voxel_r(X, Y, run_idx, participants)
        _same(new, old.tr_cross_validated_ridge(X, Y, run_idx, n_held_out=4))
        assert new.dtype == np.float32

    def test_run_blocks_within_participant(self):
        from brainscore.benchmarks.lahner2024 import ParticipantRunBlocks
        X, Y, run_idx, participants = _tr_data()
        new = ParticipantRunBlocks(2).per_voxel_r(X, Y, run_idx, participants)
        _same(new, old.tr_cross_validated_ridge_within_participant(
            X, Y, run_idx, participants, n_held_out_per_participant=2))


def _stories(n_stories=8, per_story=30, n_feat=40, n_targets=400, seed=5):
    rng = np.random.default_rng(seed)
    n = n_stories * per_story
    X = rng.standard_normal((n, n_feat)).astype(np.float32)
    signal = X[:, :3] @ rng.standard_normal((3, n_targets))
    noise = rng.standard_normal((n, n_targets)) * np.linspace(0.3, 30.0, n_targets)
    Y = (signal + noise).astype(np.float32)
    Y[:, 11] = 0.0                                 # a constant vertex
    stories = np.repeat([f's{i}' for i in range(n_stories)], per_story)
    return X, Y, stories


class TestIdentityLeBel:
    """``LeBel2023-UTS03-encoding{,-smoke,-contextwindow,-languagemask}``: story
    folds, a penalty shared by all vertices chosen on held-out stories, float32,
    dual form for wide features. ``per_voxel_alpha`` is a constructor option."""

    @pytest.mark.parametrize('n_feat', [40, 400])       # primal, dual
    @pytest.mark.parametrize('per_voxel_alpha', [False, True])
    def test_folds(self, n_feat, per_voxel_alpha):
        from brainscore.benchmarks.lebel2023.benchmark import DEFAULT_ALPHA_GRID, _encoding_fit
        X, Y, stories = _stories(n_feat=n_feat)
        grid = (1e-2, 1e0, 1e2, 1e3, 1e4) if n_feat == 40 else DEFAULT_ALPHA_GRID
        (ref_r, ref_median, ref_mean), ref_alphas = old.lebel_folds(
            X, Y, stories, grid, per_voxel_alpha)
        r, chosen = cross_validated_r(X, Y, group_kfold(stories, 5, 0),
                                      _encoding_fit(grid, per_voxel_alpha, 0), groups=stories)
        finite, median, mean = summarize(r)
        _same(finite, ref_r)
        assert (median, mean) == (ref_median, ref_mean)
        assert [np.median(fold.per_target()[:, 0]).item() for fold in chosen] == ref_alphas
        if per_voxel_alpha:
            assert any(len(np.unique(fold.index)) > 1 for fold in chosen)


def _algonauts_synth(n_stimuli=12, per_stimulus=25, seed=0):
    rng = np.random.default_rng(seed)
    n = n_stimuli * per_stimulus
    groups = np.repeat([f's{i}' for i in range(n_stimuli)], per_stimulus)
    A = (rng.standard_normal((n, 30)) + 5.0).astype(np.float32)
    B = (rng.standard_normal((n, 20)) - 2.0).astype(np.float32)
    Y = (A[:, :4] @ rng.standard_normal((4, 50)) + 3.0 * rng.standard_normal((n, 50))
         + 100.0).astype(np.float32)
    Y[:, 5] = 7.0                                  # a constant parcel
    return {'A': (A, None), 'B': (B, None)}, Y, groups


def _window(Z):
    """A two-row stimulus window, like the benchmark's ``stack``."""
    return np.concatenate([Z, np.roll(Z, 1, axis=0)], axis=1)


class TestIdentityAlgonauts:
    """Courtois NeuroMod Friends (``Algonauts2025-friends-*``, version 2):
    the preparation per fold, the penalty tuple on the first inner fold of
    stimuli, a banded ridge with an intercept."""

    @pytest.mark.parametrize('names,n_candidates', [(['A'], 60), (['A', 'B'], None),
                                                    (['A', 'B'], 7)])
    def test_score_banded(self, names, n_candidates):
        from brainscore.benchmarks.algonauts2025.benchmark import Algonauts2025
        from brainscore.benchmarks.algonauts2025.scoring import score_banded
        bands, Y, groups = _algonauts_synth()
        bands = {name: bands[name] for name in names}
        keep = np.ones(len(groups), dtype=bool)
        keep[::25] = False
        kwargs = dict(dim_cap=12, alpha_grid=Algonauts2025.BANDED_ALPHA_GRID,
                      n_candidates=n_candidates)
        new_r, new_info = score_banded(bands, Y, groups, keep, _window, **kwargs)
        ref_r, ref_info = old.algonauts_score_banded(bands, Y, groups, keep, _window, **kwargs)
        _same(new_r, ref_r)
        assert new_info == ref_info

    def test_fit_predict_banded(self):
        from brainscore.benchmarks.algonauts2025.benchmark import Algonauts2025
        from brainscore.benchmarks.algonauts2025.scoring import fit_predict_banded
        bands, Y, groups = _algonauts_synth()
        new_bands, _, _ = _algonauts_synth(seed=8)
        keep = np.ones(len(groups), dtype=bool)
        kwargs = dict(dim_cap=12, alpha_grid=Algonauts2025.BANDED_ALPHA_GRID,
                      n_candidates=60)
        new = fit_predict_banded(bands, Y, groups, keep, _window, new_bands, _window, **kwargs)
        ref = old.algonauts_fit_predict_banded(bands, Y, groups, keep, _window, new_bands,
                                               _window, **kwargs)
        _same(new[0], ref[0])
        assert new[1] == ref[1]
