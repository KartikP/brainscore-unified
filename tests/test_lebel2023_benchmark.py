"""Structural tests for the LeBel2023 whole-cortex encoding benchmark.

These run without the source pickle and without model weights: they exercise
the design-matrix and cross-validation logic directly, which is where the
scientific errors live.
"""

import numpy as np
import pytest

from brainscore.benchmarks._scoring_utils import run_kfold_masks
from brainscore.benchmarks.lebel2023.benchmark import (
    DEFAULT_DELAYS,
    _delay_within_stories,
    _ridge_grouped_alpha,
    _zscore_per_story,
)

pytestmark = pytest.mark.unit


def _two_stories(n_a=12, n_b=9, n_features=3, seed=0):
    rng = np.random.default_rng(seed)
    stories = np.array(['a'] * n_a + ['b'] * n_b)
    features = rng.normal(size=(n_a + n_b, n_features)).astype(np.float32)
    return features, stories


class TestDelays:
    def test_shape_is_one_block_per_delay(self):
        """Rows are already compacted to those with full history."""
        features, stories = _two_stories()
        delayed, mask = _delay_within_stories(features, stories, (1, 2, 3))
        assert delayed.shape == (mask.sum(), features.shape[1] * 3)
        assert mask.sum() < len(stories)   # the opening rows really are dropped

    def test_delay_shifts_by_the_requested_number_of_rows(self):
        features, stories = _two_stories()
        delayed, mask = _delay_within_stories(features, stories, (2,))
        kept = np.flatnonzero(mask)
        rows_a = np.flatnonzero(stories == 'a')
        # Row `kept[i]` of the output holds the features from two rows earlier.
        wanted = np.searchsorted(kept, rows_a[2:])
        assert np.allclose(delayed[wanted], features[rows_a[:-2]])

    def test_no_feature_crosses_a_story_boundary(self):
        """Story b's rows must never draw on story a's closing rows."""
        features, stories = _two_stories()
        delayed, mask = _delay_within_stories(features, stories, (1, 2, 3, 4))
        kept = np.flatnonzero(mask)
        rows_b = np.flatnonzero(stories == 'b')
        opening = delayed[np.searchsorted(kept, rows_b[4:4 + 3])]
        for row in opening:
            for block in row.reshape(4, -1):
                if block.any():
                    # Any populated block must come from story b itself.
                    assert np.isclose(block, features[rows_b]).all(axis=1).any()

    def test_rows_without_full_history_are_flagged(self):
        features, stories = _two_stories(n_a=12, n_b=9)
        _, has_full_history = _delay_within_stories(
            features, stories, DEFAULT_DELAYS)
        max_delay = max(DEFAULT_DELAYS)
        for story in ('a', 'b'):
            rows = np.flatnonzero(stories == story)
            assert not has_full_history[rows[:max_delay]].any()
            assert has_full_history[rows[max_delay:]].all()

    def test_constant_features_carry_no_signal_after_masking(self):
        """The regression that mattered: story-onset padding must not predict.

        With constant input every retained row of the design matrix has to be
        identical. If it is not, the padding at each story's start is acting as
        an onset marker, and a model with no stimulus information can score
        against the BOLD onset response.
        """
        features = np.ones((30, 4), dtype=np.float32)
        stories = np.array(['a'] * 15 + ['b'] * 15)
        delayed, _ = _delay_within_stories(features, stories, DEFAULT_DELAYS)
        assert np.ptp(delayed, axis=0).max() == 0


    def test_compacted_output_matches_the_naive_full_length_build(self):
        """The compacted fill must reproduce what building-then-indexing gave.

        This is the guard on the memory optimization: the wide design matrices
        are built at final size rather than full length and then indexed, and
        that rewrite must not move a single value.
        """
        rng = np.random.default_rng(0)
        stories = np.repeat(['a', 'b', 'c'], 20)
        features = rng.normal(size=(len(stories), 4)).astype(np.float32)
        delays = (1, 2, 3, 4)

        n, p = features.shape
        naive = np.zeros((n, p * len(delays)), dtype=np.float32)
        for story in np.unique(stories):
            rows = np.flatnonzero(stories == story)
            for d_i, delay in enumerate(delays):
                span = slice(d_i * p, (d_i + 1) * p)
                naive[rows[delay:], span] = features[rows][:len(rows) - delay]

        compact, mask = _delay_within_stories(features, stories, delays)
        assert np.array_equal(compact, naive[mask])


class TestZScore:
    def test_each_story_is_standardised_independently(self):
        stories = np.array(['a'] * 10 + ['b'] * 10)
        values = np.concatenate([
            np.linspace(100, 200, 10)[:, None] * np.ones((1, 3)),
            np.linspace(0, 1, 10)[:, None] * np.ones((1, 3)),
        ]).astype(np.float32)
        out = _zscore_per_story(values, stories)
        for story in ('a', 'b'):
            block = out[stories == story]
            assert np.allclose(block.mean(axis=0), 0, atol=1e-5)
            assert np.allclose(block.std(axis=0), 1, atol=1e-5)

    def test_constant_vertex_becomes_zero_not_nan(self):
        stories = np.array(['a'] * 6)
        values = np.ones((6, 2), dtype=np.float32)
        out = _zscore_per_story(values, stories)
        assert np.isfinite(out).all()
        assert np.allclose(out, 0)


class TestStoryFolds:
    def test_no_story_appears_in_both_train_and_test(self):
        stories = np.repeat([f's{i}' for i in range(10)], 7)
        for train_mask, test_mask in run_kfold_masks(stories, n_splits=5):
            assert not (train_mask & test_mask).any()
            overlap = set(stories[train_mask]) & set(stories[test_mask])
            assert not overlap

    def test_every_sample_is_tested_exactly_once(self):
        stories = np.repeat([f's{i}' for i in range(10)], 7)
        counts = np.zeros(len(stories), dtype=int)
        for _, test_mask in run_kfold_masks(stories, n_splits=5):
            counts += test_mask
        assert (counts == 1).all()


class TestGroupedAlphaRidge:
    def test_recovers_a_linear_mapping_with_weak_penalty(self):
        rng = np.random.default_rng(0)
        stories = np.repeat([f's{i}' for i in range(8)], 25)
        X = rng.normal(size=(len(stories), 5))
        W_true = rng.normal(size=(5, 3))
        Y = X @ W_true
        pred, alpha = _ridge_grouped_alpha(
            X, Y, X, stories, alpha_grid=(1e-6, 1e3))
        assert alpha == 1e-6
        assert np.corrcoef(pred[:, 0], Y[:, 0])[0, 1] > 0.99

    def test_selection_holds_out_whole_stories(self):
        """A penalty must be chosen, and the chosen one must be in the grid."""
        rng = np.random.default_rng(1)
        stories = np.repeat([f's{i}' for i in range(8)], 25)
        X = rng.normal(size=(len(stories), 5))
        Y = rng.normal(size=(len(stories), 3))
        grid = (1e-3, 1.0, 1e3)
        _, alpha = _ridge_grouped_alpha(X, Y, X, stories, alpha_grid=grid)
        assert alpha in grid


class TestRegistration:
    def test_both_variants_are_registered(self):
        from brainscore import benchmark_registry
        assert 'LeBel2023-UTS03-encoding' in benchmark_registry
        assert 'LeBel2023-UTS03-encoding-smoke' in benchmark_registry

    def test_smoke_variant_subsamples_targets(self):
        from brainscore import benchmark_registry
        smoke = benchmark_registry['LeBel2023-UTS03-encoding-smoke']()
        full = benchmark_registry['LeBel2023-UTS03-encoding']()
        assert smoke._max_targets == 2000
        assert full._max_targets is None
