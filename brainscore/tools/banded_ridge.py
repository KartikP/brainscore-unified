"""Ridge encoders for feature→BOLD prediction — single-band and banded.

Two functions, one home. Both were previously copy-pasted across the
Lahner2024 multimodal benchmarks (and the now-archived sweep scripts):

- ``ridge_fit_predict`` — one penalty for all features (sklearn Ridge).
- ``banded_ridge_fit_predict`` — one penalty *per feature group* (per
  modality / per band), selected on a single held-out validation split,
  then refit on the full training fold. The closed form solves
  ``(XᵀX + diag(λ)) W = XᵀY`` with ``λ_j = α_{group(j)}``; XᵀX is hoisted
  out of the α-search loop (only ``diag(λ)`` changes per α-tuple), so the
  search is one matmul + ``len(alpha_grid) ** n_groups`` solves.

The banded function is generic over the number of groups; with two groups
it reproduces the prior hand-rolled nested-loop implementation bit-for-bit
(``itertools.product`` preserves the nested-loop order, so tie-breaking on
the strict ``>`` comparison is identical).
"""
from itertools import product

import numpy as np


def ridge_fit_predict(X_train, Y_train, X_pred, alpha=1.0,
                      dtype=np.float32):
    """Fit a single-penalty ridge encoder on (X_train → Y_train), predict X_pred.

    Returns predictions of shape ``(len(X_pred), Y_train.shape[1])``. By
    default predictions are float32, matching the historical helper contract;
    pass ``dtype=None`` to preserve sklearn's native prediction dtype.
    """
    from sklearn.linear_model import Ridge
    reg = Ridge(alpha=alpha).fit(X_train, Y_train)
    pred = reg.predict(X_pred)
    return pred if dtype is None else pred.astype(dtype)


def _mean_pearson(Y_true_centered, Y_true_var, Y_pred):
    """Mean across-target Pearson r between centered truth and raw preds."""
    yp = Y_pred - Y_pred.mean(axis=0)
    num = (Y_true_centered * yp).sum(axis=0)
    den = np.sqrt(Y_true_var * (yp ** 2).sum(axis=0))
    with np.errstate(divide='ignore', invalid='ignore'):
        r = np.where(den > 0, num / den, 0.0)
    return float(np.mean(r))


def banded_ridge_fit_predict(train_groups, test_groups, Y_train, alpha_grid,
                             val_frac=0.2, seed=0):
    """Banded ridge: one α per feature group, refit on the full training fold.

    Args:
        train_groups: list of 2-D arrays ``(n_train, n_features_g)``, one per
            band/modality. Columns are concatenated; group ``g`` gets α_g.
        test_groups: parallel list of ``(n_test, n_features_g)`` arrays.
        Y_train: ``(n_train, n_targets)``.
        alpha_grid: candidate α values searched as the full Cartesian product
            over groups (cost = ``len(alpha_grid) ** len(train_groups)`` solves).
        val_frac: fraction of the training fold held out to score α-tuples.
        seed: RNG seed for the validation split.

    Returns:
        ``(Y_test_pred, best_alpha_tuple)``.
    """
    n_groups = len(train_groups)
    rng = np.random.default_rng(seed)
    n_train = train_groups[0].shape[0]
    n_val = max(1, int(round(n_train * val_frac)))
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

    # Hoist the (n_obs × n_feat × n_feat) matmul out of the α-grid loop —
    # only diag(λ) changes per α-tuple. One matmul + n_grid solves.
    XtX_inner = X_inner.T @ X_inner
    XtY_inner = X_inner.T @ Y_inner
    Y_val_centered = Y_val - Y_val.mean(axis=0)
    Y_val_var = (Y_val_centered ** 2).sum(axis=0)

    def lam_for(alphas):
        return np.concatenate([np.full(sizes[g], alphas[g])
                               for g in range(n_groups)])

    best_score = -np.inf
    best_alpha = None
    for alphas in product(*([alpha_grid] * n_groups)):
        W = np.linalg.solve(XtX_inner + np.diag(lam_for(alphas)), XtY_inner)
        score = _mean_pearson(Y_val_centered, Y_val_var, X_val @ W)
        if score > best_score:
            best_score = score
            best_alpha = alphas

    # Refit on the full training fold with the chosen α-tuple.
    X_full_train = stack(train_groups, slice(None))
    X_full_test = stack(test_groups, slice(None))
    W = np.linalg.solve(X_full_train.T @ X_full_train + np.diag(lam_for(best_alpha)),
                        X_full_train.T @ Y_train)
    return X_full_test @ W, best_alpha
