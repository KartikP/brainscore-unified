"""Shared encoding-model scoring helpers for benchmark implementations."""

from typing import Iterable, Sequence

import numpy as np

from brainscore_core.metrics import per_unit_pearson
from brainscore.tools.banded_ridge import ridge_fit_predict


def run_kfold_masks(run_idx, n_splits: int = 5, random_state: int = 0):
    """Yield train/test masks over whole run ids."""
    unique_runs = np.unique(run_idx)
    for train_pos, test_pos in kfold_positions(
            len(unique_runs), n_splits=n_splits, random_state=random_state):
        yield (
            np.isin(run_idx, unique_runs[train_pos]),
            np.isin(run_idx, unique_runs[test_pos]),
        )


def kfold_positions(n_items: int, n_splits: int = 5, random_state: int = 0):
    """Yield KFold train/test integer positions for ``range(n_items)``."""
    from sklearn.model_selection import KFold

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    yield from kf.split(np.arange(n_items))


def kfold_ridge_predictions(
    X,
    Y,
    *,
    alpha: float = 1.0,
    n_splits: int = 5,
    random_state: int = 0,
    dtype=None,
):
    """Held-out predictions from shuffled KFold ridge over sample rows."""
    held = np.zeros_like(Y) if dtype is None else np.zeros(Y.shape, dtype=dtype)
    for train_idx, test_idx in kfold_positions(
            X.shape[0], n_splits=n_splits, random_state=random_state):
        held[test_idx] = ridge_fit_predict(
            X[train_idx], Y[train_idx], X[test_idx],
            alpha=alpha, dtype=dtype)
    return held


def masked_ridge_predictions(
    X,
    Y,
    fold_masks: Iterable[tuple[np.ndarray, np.ndarray]],
    *,
    alpha: float = 1.0,
    fill_value=np.nan,
    dtype=None,
):
    """Held-out ridge predictions for precomputed train/test row masks."""
    held = np.full(Y.shape, fill_value, dtype=(Y.dtype if dtype is None else dtype))
    for train_mask, test_mask in fold_masks:
        held[test_mask] = ridge_fit_predict(
            X[train_mask], Y[train_mask], X[test_mask],
            alpha=alpha, dtype=dtype)
    return held


def summed_group_kfold_ridge_predictions(
    feature_groups: Sequence[np.ndarray],
    Y,
    *,
    alpha: float = 1.0,
    n_splits: int = 5,
    random_state: int = 0,
    dtype=None,
):
    """KFold ridge per feature group, summed on the held-out folds."""
    held = np.zeros_like(Y) if dtype is None else np.zeros(Y.shape, dtype=dtype)
    n = feature_groups[0].shape[0]
    for train_idx, test_idx in kfold_positions(
            n, n_splits=n_splits, random_state=random_state):
        for group in feature_groups:
            held[test_idx] += ridge_fit_predict(
                group[train_idx], Y[train_idx], group[test_idx],
                alpha=alpha, dtype=dtype)
    return held


def callback_group_kfold_predictions(
    feature_groups: Sequence[np.ndarray],
    Y,
    predict_fold,
    *,
    n_splits: int = 5,
    random_state: int = 0,
    dtype=None,
):
    """KFold predictions for feature groups using a caller-provided fold fit."""
    held = np.zeros_like(Y) if dtype is None else np.zeros(Y.shape, dtype=dtype)
    fold_info = []
    n = feature_groups[0].shape[0]
    for train_idx, test_idx in kfold_positions(
            n, n_splits=n_splits, random_state=random_state):
        train_groups = [group[train_idx] for group in feature_groups]
        test_groups = [group[test_idx] for group in feature_groups]
        result = predict_fold(train_groups, test_groups, Y[train_idx])
        if isinstance(result, tuple):
            pred, info = result
            fold_info.append(info)
        else:
            pred = result
        held[test_idx] = pred if dtype is None else np.asarray(pred, dtype=dtype)
    return held, fold_info


def finite_per_unit_pearson(Y_true, Y_pred):
    """Per-unit Pearson r, dropping units with NaN correlation."""
    r = per_unit_pearson(Y_true, Y_pred)
    return r[~np.isnan(r)]


def valid_prediction_per_unit_pearson(Y_true, Y_pred):
    """Per-unit Pearson r over rows that received held-out predictions."""
    valid = ~np.isnan(Y_pred[:, 0])
    if not valid.any():
        return np.full(Y_true.shape[1], np.nan, dtype=np.float32)
    return per_unit_pearson(Y_true[valid], Y_pred[valid])


def pearson_summary(Y_true, Y_pred):
    """Return finite per-unit r plus median and mean summaries."""
    finite = finite_per_unit_pearson(Y_true, Y_pred)
    return finite, float(np.median(finite)), float(np.mean(finite))
