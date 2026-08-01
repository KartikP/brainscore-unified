"""The chunked ridge must be numerically equivalent to sklearn's Ridge.

It exists only to bound memory: sklearn copies and upcasts the design matrix to
float64, which OOM-killed Algonauts scoring (63 GB on a 62 GB box). Swapping the
solver is only acceptable if it does not move the science, so equivalence is the
test that matters -- not speed, not memory.
"""
import numpy as np
import pytest

from brainscore.tools.banded_ridge import ridge_fit_predict, ridge_fit_predict_chunked


@pytest.mark.unit
@pytest.mark.parametrize('alpha', [1.0, 10.0, 1000.0])
def test_matches_sklearn_ridge(alpha):
    rng = np.random.default_rng(0)
    X = rng.standard_normal((600, 30)).astype(np.float32)
    Y = (X[:, :4] @ rng.standard_normal((4, 8))
         + 0.3 * rng.standard_normal((600, 8))).astype(np.float32)
    X_pred = rng.standard_normal((120, 30)).astype(np.float32)
    expected = ridge_fit_predict(X, Y, X_pred, alpha=alpha, dtype=None)
    actual = ridge_fit_predict_chunked(X, Y, X_pred, alpha=alpha, dtype=None)
    np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-4)


@pytest.mark.unit
def test_chunk_size_does_not_change_the_answer():
    """Row chunking is an implementation detail; it must not be observable."""
    rng = np.random.default_rng(1)
    X = rng.standard_normal((500, 20)).astype(np.float32)
    Y = rng.standard_normal((500, 6)).astype(np.float32)
    X_pred = rng.standard_normal((90, 20)).astype(np.float32)
    ref = ridge_fit_predict_chunked(X, Y, X_pred, alpha=5.0, dtype=None, row_chunk=500)
    for chunk in (7, 64, 499, 10_000):
        got = ridge_fit_predict_chunked(X, Y, X_pred, alpha=5.0, dtype=None,
                                        row_chunk=chunk)
        np.testing.assert_allclose(got, ref, rtol=1e-6, atol=1e-6)


@pytest.mark.unit
def test_intercept_is_fit_and_unpenalized():
    """A large constant offset in Y must be reproduced even at huge alpha --
    that only holds if the intercept is fit and left unpenalized."""
    rng = np.random.default_rng(2)
    X = rng.standard_normal((300, 5)).astype(np.float32)
    Y = (np.full((300, 2), 50.0) + 0.01 * rng.standard_normal((300, 2))).astype(np.float32)
    pred = ridge_fit_predict_chunked(X, Y, X[:10], alpha=1e6, dtype=None)
    np.testing.assert_allclose(pred, 50.0, atol=0.1)
