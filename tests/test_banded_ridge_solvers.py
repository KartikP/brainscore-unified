"""Tests for the ridge solvers in ``tools.banded_ridge``.

Both solvers exist to make wide, many-target problems tractable — the dual form
when features outnumber samples, the shape-aware projection when targets
outnumber held-out rows. Neither may change the estimator, only its cost, so
each is checked against the textbook solution as well as for taking the cheap
path.
"""

import numpy as np
import pytest

from brainscore.tools.banded_ridge import (
    dual_aware_ridge_predict, solve_and_project)

pytestmark = pytest.mark.unit


class TestRidgeSolveForms:
    """The dual form and the shape-aware projection exist so wide problems stay
    tractable; neither may change the estimator, only its cost."""

    def test_projection_is_grouping_independent(self):
        """`solve_and_project` must agree whichever side it solves against."""
        rng = np.random.default_rng(3)
        m, n_rows, n_targets = 60, 12, 400
        gram = rng.normal(size=(m, m)); gram = gram @ gram.T + m * np.eye(m)
        cross = rng.normal(size=(n_rows, m))
        target = rng.normal(size=(m, n_targets))
        explicit = cross @ np.linalg.solve(gram, target)
        assert np.allclose(solve_and_project(gram, cross, target), explicit, atol=1e-9)
        # and with the shapes reversed, so the other branch is taken
        cross2 = rng.normal(size=(n_targets, m))
        target2 = rng.normal(size=(m, n_rows))
        explicit2 = cross2 @ np.linalg.solve(gram, target2)
        assert np.allclose(solve_and_project(gram, cross2, target2), explicit2, atol=1e-9)

    @staticmethod
    def _explicit_primal(X, Y, X_test, alpha):
        return X_test @ np.linalg.solve(
            X.T @ X + alpha * np.eye(X.shape[1]), X.T @ Y)

    @pytest.mark.parametrize('n,p', [(200, 50), (50, 200), (80, 80)])
    def test_matches_the_textbook_solution_either_way(self, n, p):
        rng = np.random.default_rng(0)
        X = rng.normal(size=(n, p))
        X -= X.mean(axis=0)
        Y = rng.normal(size=(n, 4))
        X_test = rng.normal(size=(15, p))
        expected = self._explicit_primal(X, Y, X_test, 3.0)
        assert np.allclose(dual_aware_ridge_predict(X, Y, X_test, 3.0), expected, atol=1e-8)

    def test_wide_inputs_take_the_dual_path(self, monkeypatch):
        """Guard the cost, not just the answer: a (p, p) solve must not happen
        when p is much larger than n."""
        rng = np.random.default_rng(0)
        X = rng.normal(size=(40, 500))
        seen = []
        original = np.linalg.solve
        monkeypatch.setattr(
            np.linalg, 'solve',
            lambda a, b: (seen.append(a.shape), original(a, b))[1])
        dual_aware_ridge_predict(X, rng.normal(size=(40, 3)), rng.normal(size=(5, 500)), 1.0)
        assert seen == [(40, 40)]
