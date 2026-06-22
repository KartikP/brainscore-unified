"""Tests for brainscore.tools.banded_ridge.

The load-bearing test is ``test_matches_reference_2group_bitforbit``: it
re-implements the exact prior hand-rolled 2-group banded ridge that lived
inline in the Lahner2024 multimodal benchmarks and asserts the shared
function reproduces it bit-for-bit. If the refactor ever drifts, this fails.
"""
import numpy as np

from brainscore.tools.banded_ridge import (
    banded_ridge_fit_predict,
    ridge_fit_predict,
)

GRID = (1.0, 10.0, 100.0, 1000.0, 10000.0)


def _reference_banded_2group(train_groups, test_groups, Y_train, alpha_grid):
    """Verbatim copy of the prior inline implementation (2 groups, seed 0,
    20% val split, nested α loop). The regression oracle."""
    rng = np.random.default_rng(0)
    n_train = train_groups[0].shape[0]
    n_val = max(1, int(round(n_train * 0.2)))
    perm = rng.permutation(n_train)
    val_idx = perm[:n_val]
    inner_idx = perm[n_val:]
    sizes = [g.shape[1] for g in train_groups]

    def stack(groups, idx):
        return np.concatenate([g[idx] for g in groups], axis=1)

    X_inner = stack(train_groups, inner_idx)
    X_val = stack(train_groups, val_idx)
    Y_inner = Y_train[inner_idx]
    Y_val = Y_train[val_idx]
    XtX = X_inner.T @ X_inner
    XtY = X_inner.T @ Y_inner
    Yc = Y_val - Y_val.mean(axis=0)
    Yv = (Yc ** 2).sum(axis=0)
    best, best_alpha = -np.inf, (1.0, 1.0)
    for av in alpha_grid:
        for aa in alpha_grid:
            lam = np.concatenate([np.full(sizes[0], av), np.full(sizes[1], aa)])
            W = np.linalg.solve(XtX + np.diag(lam), XtY)
            yp = X_val @ W
            yp = yp - yp.mean(axis=0)
            num = (Yc * yp).sum(axis=0)
            den = np.sqrt(Yv * (yp ** 2).sum(axis=0))
            with np.errstate(divide='ignore', invalid='ignore'):
                r = np.where(den > 0, num / den, 0.0)
            s = float(np.mean(r))
            if s > best:
                best, best_alpha = s, (av, aa)
    av, aa = best_alpha
    lam = np.concatenate([np.full(sizes[0], av), np.full(sizes[1], aa)])
    Xtr = stack(train_groups, slice(None))
    Xte = stack(test_groups, slice(None))
    W = np.linalg.solve(Xtr.T @ Xtr + np.diag(lam), Xtr.T @ Y_train)
    return Xte @ W, best_alpha


def test_matches_reference_2group_bitforbit():
    rng = np.random.default_rng(42)
    gv_tr, ga_tr = rng.standard_normal((60, 8)), rng.standard_normal((60, 5))
    gv_te, ga_te = rng.standard_normal((20, 8)), rng.standard_normal((20, 5))
    Y = rng.standard_normal((60, 4))
    pred_ref, a_ref = _reference_banded_2group(
        [gv_tr, ga_tr], [gv_te, ga_te], Y, GRID)
    pred_new, a_new = banded_ridge_fit_predict(
        [gv_tr, ga_tr], [gv_te, ga_te], Y, GRID)
    assert a_new == a_ref
    assert np.array_equal(pred_new, pred_ref)  # bit-for-bit, not just close


def test_noise_band_is_shrunk_harder_than_signal_band():
    rng = np.random.default_rng(0)
    G0 = rng.standard_normal((200, 6))                  # signal band
    W_true = rng.standard_normal((6, 3))
    Y = G0 @ W_true + 0.01 * rng.standard_normal((200, 3))
    G1 = rng.standard_normal((200, 6))                  # pure-noise band
    G0_te, G1_te = rng.standard_normal((50, 6)), rng.standard_normal((50, 6))
    pred, (a0, a1) = banded_ridge_fit_predict(
        [G0, G1], [G0_te, G1_te], Y, GRID)
    assert pred.shape == (50, 3)
    assert a1 >= a0          # noise band penalized at least as hard as signal


def test_generic_three_groups_runs():
    rng = np.random.default_rng(1)
    tr = [rng.standard_normal((40, 4)) for _ in range(3)]
    te = [rng.standard_normal((10, 4)) for _ in range(3)]
    Y = rng.standard_normal((40, 2))
    pred, best = banded_ridge_fit_predict(tr, te, Y, (1.0, 100.0))
    assert pred.shape == (10, 2)
    assert len(best) == 3 and all(a in (1.0, 100.0) for a in best)


def test_ridge_fit_predict_shape_and_dtype():
    rng = np.random.default_rng(0)
    Xtr, Y, Xp = (rng.standard_normal((30, 5)),
                  rng.standard_normal((30, 3)),
                  rng.standard_normal((7, 5)))
    pred = ridge_fit_predict(Xtr, Y, Xp, alpha=1.0)
    assert pred.shape == (7, 3) and pred.dtype == np.float32
