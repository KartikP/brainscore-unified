"""Per-parcel encoding scorers for the Algonauts 2025 benchmark.

Separated from ``benchmark.py`` (SoC: scoring vs data I/O) and made pure so
it is unit-testable without the heavy feature extraction or any fMRI data.

A "modality block" is a feature matrix that has ALREADY been
stimulus-window-stacked, HRF-shifted, and edge-excluded — one block per
modality, all sharing the same ``(n_obs,)`` row alignment and
``run_idx_kept``. ``score_encoding_modes`` combines the blocks according to
``mode`` and returns per-parcel Pearson r on run-held-out predictions:

- ``video_only`` / ``audio_only`` / ``language_only`` — single Ridge on that
  one block at a fixed α (default 1.0; reproduces the validated video-only
  number bit-for-bit).
- ``concat`` — single Ridge on the column-concatenation of all blocks.
- ``per_modality`` — one Ridge per block; predictions summed.
- ``banded`` — one α per block, selected by a nested inner-CV α-tuple sweep
  inside each outer fold (the Algonauts paper-baseline protocol). NOTE: this
  is a *different* CV protocol from ``tools.banded_ridge`` (single held-out
  split), so it is intentionally not shared with the Lahner benchmarks.

Banding is applied to the *stacked* columns: block ``g`` (width
``W * n_feat_g``) gets penalty α_g across all its columns.
"""
from itertools import product

import numpy as np

from brainscore.benchmarks._scoring_utils import (
    kfold_positions,
    masked_ridge_predictions,
    run_kfold_masks,
    valid_prediction_per_unit_pearson,
)

MODALITY_FOR_MODE = {
    'video_only': 'video',
    'audio_only': 'audio',
    'language_only': 'language',
}


def per_voxel_pearson(Y_true, Y_pred):
    """Per-column Pearson r between recorded and predicted BOLD, dropping the
    rows the model never predicted (NaN). The centered correlation itself is
    the canonical ``brainscore_core.metrics.per_unit_pearson``."""
    return valid_prediction_per_unit_pearson(Y_true, Y_pred)


def _run_kfold(run_idx_kept, n_splits, random_state):
    """Yield (train_mask, test_mask) over whole runs — no TR ever splits
    across train/test (avoids temporal leakage)."""
    yield from run_kfold_masks(run_idx_kept, n_splits, random_state)


def _cv_ridge_predict(X, Y, run_idx_kept, alpha, n_splits, random_state):
    """Run-held-out 5-fold Ridge; return held-out predictions (NaN where
    a row was never in a test fold — shouldn't happen with KFold)."""
    return masked_ridge_predictions(
        X, Y, _run_kfold(run_idx_kept, n_splits, random_state),
        alpha=alpha, dtype=np.float32)


def _banded_nested_cv(blocks, Y, run_idx_kept, banded_alpha_grid,
                      n_splits, random_state):
    """Banded ridge with a nested inner-CV α-tuple sweep per outer fold.

    Faithful port of the validated reproduction script: outer run-KFold;
    within each outer fold an inner KFold(random_state=fold_i) takes the
    first split as inner-train/inner-val; the best (α per block) tuple by
    mean inner-val Pearson is refit on the full outer-train and scored on
    the outer-test. XᵀX is hoisted out of the α-tuple loop.
    """
    widths = [b.shape[1] for b in blocks]
    offsets, cur = [], 0
    for w in widths:
        offsets.append((cur, cur + w))
        cur += w
    X = np.concatenate(blocks, axis=1)
    n_bands = len(blocks)
    tuples = list(product(banded_alpha_grid, repeat=n_bands))

    unique_runs = np.unique(run_idx_kept)
    held = np.full_like(Y, np.nan, dtype=np.float32)
    chosen = []
    for fold_i, (tr_pos, te_pos) in enumerate(kfold_positions(
            len(unique_runs), n_splits=n_splits,
            random_state=random_state)):
        tr_runs, te_runs = unique_runs[tr_pos], unique_runs[te_pos]
        tr = np.isin(run_idx_kept, tr_runs)
        te = np.isin(run_idx_kept, te_runs)

        i_tr_pos, i_va_pos = next(kfold_positions(
            len(tr_runs), n_splits=n_splits, random_state=fold_i))
        i_tr = np.isin(run_idx_kept, tr_runs[i_tr_pos])
        i_va = np.isin(run_idx_kept, tr_runs[i_va_pos])

        X_itr, Y_itr = X[i_tr], Y[i_tr]
        X_iva, Y_iva = X[i_va], Y[i_va]
        XtX_base = X_itr.T @ X_itr
        XtY = X_itr.T @ Y_itr
        Y_iva_c = Y_iva - Y_iva.mean(0)
        Y_iva_var = (Y_iva_c ** 2).sum(0)

        best_score, best_tuple = -np.inf, None
        for atup in tuples:
            diag_alpha = np.zeros(X.shape[1], dtype=np.float32)
            for (lo, hi), a_val in zip(offsets, atup):
                diag_alpha[lo:hi] = a_val
            XtX = XtX_base.copy()
            XtX[np.diag_indices_from(XtX)] += diag_alpha
            try:
                W_sol = np.linalg.solve(XtX, XtY)
            except np.linalg.LinAlgError:
                continue
            pred = X_iva @ W_sol
            pred_c = pred - pred.mean(0)
            num = (Y_iva_c * pred_c).sum(axis=0)
            den = np.sqrt(Y_iva_var * (pred_c ** 2).sum(0))
            with np.errstate(divide='ignore', invalid='ignore'):
                val_r = np.where(den > 0, num / den, 0.0)
            mean_r = float(np.mean(val_r))
            if mean_r > best_score:
                best_score, best_tuple = mean_r, atup
        chosen.append(best_tuple)

        diag_alpha = np.zeros(X.shape[1], dtype=np.float32)
        for (lo, hi), a_val in zip(offsets, best_tuple):
            diag_alpha[lo:hi] = a_val
        XtX = X[tr].T @ X[tr]
        XtY = X[tr].T @ Y[tr]
        XtX[np.diag_indices_from(XtX)] += diag_alpha
        W_sol = np.linalg.solve(XtX, XtY)
        held[te] = (X[te] @ W_sol).astype(np.float32)
    return held, chosen


def score_encoding_modes(per_modality_stacked, Y, run_idx_kept, mode, *,
                         ridge_alpha=1.0,
                         banded_alpha_grid=(1.0, 10.0, 100.0, 1000.0, 10000.0),
                         n_splits=5, random_state=0):
    """Score per-parcel encoding for the requested multimodal combination.

    Args:
        per_modality_stacked: ordered dict ``{modality: (n_obs, W*n_feat)}``
            of stacked+HRF-shifted+edge-excluded feature blocks.
        Y: ``(n_obs, n_parcels)`` recorded BOLD (already edge-excluded).
        run_idx_kept: ``(n_obs,)`` run id per observation.
        mode: one of video_only/audio_only/language_only/concat/per_modality/banded.

    Returns ``(per_voxel_r, info)``. Raises ValueError if ``mode`` needs a
    modality not present in ``per_modality_stacked``.
    """
    present = list(per_modality_stacked.keys())

    if mode in MODALITY_FOR_MODE:
        need = MODALITY_FOR_MODE[mode]
        if need not in per_modality_stacked:
            raise ValueError(
                f"mode={mode!r} needs '{need}' features but the candidate "
                f"only produced {present}. Use a candidate with a {need} "
                f"tower, or a mode that matches its modalities.")
        held = _cv_ridge_predict(per_modality_stacked[need], Y, run_idx_kept,
                                 ridge_alpha, n_splits, random_state)
        info = {'mode': mode, 'bands': [need], 'ridge_alpha': ridge_alpha}

    elif mode == 'concat':
        X = np.concatenate(list(per_modality_stacked.values()), axis=1)
        held = _cv_ridge_predict(X, Y, run_idx_kept, ridge_alpha,
                                 n_splits, random_state)
        info = {'mode': mode, 'bands': present, 'ridge_alpha': ridge_alpha}

    elif mode == 'per_modality':
        held = None
        for block in per_modality_stacked.values():
            pred = _cv_ridge_predict(block, Y, run_idx_kept, ridge_alpha,
                                     n_splits, random_state)
            held = pred if held is None else held + pred
        info = {'mode': mode, 'bands': present, 'ridge_alpha': ridge_alpha}

    elif mode == 'banded':
        blocks = list(per_modality_stacked.values())
        held, chosen = _banded_nested_cv(
            blocks, Y, run_idx_kept, list(banded_alpha_grid),
            n_splits, random_state)
        info = {'mode': mode, 'bands': present,
                'banded_alpha_grid': list(banded_alpha_grid),
                'chosen_alphas_per_fold': [
                    dict(zip(present, [float(x) for x in t])) for t in chosen]}

    else:
        raise ValueError(f"unknown mode {mode!r}")

    return per_voxel_pearson(Y, held), info
