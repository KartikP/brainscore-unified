"""Tests for brainscore.benchmarks.algonauts2025.scoring.

Two load-bearing regression guards:
- ``test_video_only_matches_current_benchmark_bitforbit`` reimplements the
  exact single-Ridge 5-fold path that ``_score_friends_train`` used to run
  inline and asserts the scorer reproduces it — so folding scoring out can't
  move the validated video-only number.
- ``test_banded_matches_reference_bitforbit`` does the same for the nested-CV
  banded protocol ported from the reproduction script.
"""
from itertools import product

import numpy as np
import pytest

from brainscore.benchmarks.algonauts2025.scoring import score_encoding_modes


def _synth(n_runs=10, per_run=20, n_feat=12, n_parcels=6, seed=0):
    rng = np.random.default_rng(seed)
    n = n_runs * per_run
    run_idx = np.repeat(np.arange(n_runs), per_run)
    X = rng.standard_normal((n, n_feat)).astype(np.float32)
    Wt = rng.standard_normal((n_feat, n_parcels))
    Y = (X @ Wt + 0.5 * rng.standard_normal((n, n_parcels))).astype(np.float32)
    return X, Y, run_idx


def _ref_single_ridge_pvr(X, Y, run_idx, alpha=1.0):
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import KFold
    unique_runs = np.unique(run_idx)
    kf = KFold(n_splits=5, shuffle=True, random_state=0)
    held = np.full_like(Y, np.nan, dtype=np.float32)
    for tr_pos, te_pos in kf.split(unique_runs):
        tr = np.isin(run_idx, unique_runs[tr_pos])
        te = np.isin(run_idx, unique_runs[te_pos])
        reg = Ridge(alpha=alpha).fit(X[tr], Y[tr])
        held[te] = reg.predict(X[te]).astype(np.float32)
    valid = ~np.isnan(held[:, 0])
    Yt, Yp = Y[valid], held[valid]
    Ytc = Yt - Yt.mean(0, keepdims=True)
    Ypc = Yp - Yp.mean(0, keepdims=True)
    num = (Ytc * Ypc).sum(0)
    den = np.sqrt((Ytc ** 2).sum(0) * (Ypc ** 2).sum(0))
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.where(den > 0, num / den, np.nan)


def _ref_banded_pvr(blocks, Y, run_idx, grid):
    from sklearn.model_selection import KFold
    widths = [b.shape[1] for b in blocks]
    offsets, cur = [], 0
    for w in widths:
        offsets.append((cur, cur + w)); cur += w
    X = np.concatenate(blocks, axis=1)
    tuples = list(product(grid, repeat=len(blocks)))
    unique_runs = np.unique(run_idx)
    kf = KFold(n_splits=5, shuffle=True, random_state=0)
    held = np.full_like(Y, np.nan, dtype=np.float32)
    for fold_i, (tr_pos, te_pos) in enumerate(kf.split(unique_runs)):
        tr_runs, te_runs = unique_runs[tr_pos], unique_runs[te_pos]
        tr = np.isin(run_idx, tr_runs); te = np.isin(run_idx, te_runs)
        ikf = KFold(n_splits=5, shuffle=True, random_state=fold_i)
        itp, ivp = next(iter(ikf.split(tr_runs)))
        itr = np.isin(run_idx, tr_runs[itp]); iva = np.isin(run_idx, tr_runs[ivp])
        XtXb = X[itr].T @ X[itr]; XtY = X[itr].T @ Y[itr]
        Yc = Y[iva] - Y[iva].mean(0); Yv = (Yc ** 2).sum(0)
        best, bt = -np.inf, None
        for atup in tuples:
            da = np.zeros(X.shape[1], dtype=np.float32)
            for (lo, hi), av in zip(offsets, atup):
                da[lo:hi] = av
            XtX = XtXb.copy(); XtX[np.diag_indices_from(XtX)] += da
            try:
                W = np.linalg.solve(XtX, XtY)
            except np.linalg.LinAlgError:
                continue
            pred = X[iva] @ W; pc = pred - pred.mean(0)
            num = (Yc * pc).sum(0); den = np.sqrt(Yv * (pc ** 2).sum(0))
            with np.errstate(divide='ignore', invalid='ignore'):
                vr = np.where(den > 0, num / den, 0.0)
            m = float(np.mean(vr))
            if m > best:
                best, bt = m, atup
        da = np.zeros(X.shape[1], dtype=np.float32)
        for (lo, hi), av in zip(offsets, bt):
            da[lo:hi] = av
        XtX = X[tr].T @ X[tr]; XtY = X[tr].T @ Y[tr]
        XtX[np.diag_indices_from(XtX)] += da
        W = np.linalg.solve(XtX, XtY)
        held[te] = (X[te] @ W).astype(np.float32)
    valid = ~np.isnan(held[:, 0])
    Yt, Yp = Y[valid], held[valid]
    Ytc = Yt - Yt.mean(0, keepdims=True); Ypc = Yp - Yp.mean(0, keepdims=True)
    num = (Ytc * Ypc).sum(0)
    den = np.sqrt((Ytc ** 2).sum(0) * (Ypc ** 2).sum(0))
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.where(den > 0, num / den, np.nan)


def test_video_only_matches_current_benchmark_bitforbit():
    X, Y, run_idx = _synth()
    ref = _ref_single_ridge_pvr(X, Y, run_idx, alpha=1.0)
    got, info = score_encoding_modes({'video': X}, Y, run_idx, 'video_only')
    assert info['bands'] == ['video']
    assert np.array_equal(got, ref, equal_nan=True)


def test_banded_matches_reference_bitforbit():
    Xv, Y, run_idx = _synth(n_feat=8, seed=1)
    Xa, _, _ = _synth(n_feat=5, seed=2)
    grid = (1.0, 100.0, 10000.0)
    ref = _ref_banded_pvr([Xv, Xa], Y, run_idx, grid)
    got, info = score_encoding_modes(
        {'video': Xv, 'audio': Xa}, Y, run_idx, 'banded',
        banded_alpha_grid=grid)
    assert info['bands'] == ['video', 'audio']
    assert len(info['chosen_alphas_per_fold']) == 5
    assert np.array_equal(got, ref, equal_nan=True)


def test_concat_and_per_modality_run():
    Xv, Y, run_idx = _synth(n_feat=8, seed=3)
    Xa, _, _ = _synth(n_feat=5, seed=4)
    blocks = {'video': Xv, 'audio': Xa}
    for mode in ('concat', 'per_modality'):
        pvr, info = score_encoding_modes(blocks, Y, run_idx, mode)
        assert pvr.shape == (Y.shape[1],)
        assert info['bands'] == ['video', 'audio']


def test_mode_requiring_absent_modality_raises():
    Xv, Y, run_idx = _synth()
    with pytest.raises(ValueError, match="needs 'audio'"):
        score_encoding_modes({'video': Xv}, Y, run_idx, 'audio_only')
