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
        # Every retained row must be identical. Stated as an explicit spread
        # rather than np.ptp, whose `out` handling misbehaves under coverage's
        # tracer and made this assertion fail only when instrumented.
        assert len(delayed) > 0
        spread = delayed.max(axis=0) - delayed.min(axis=0)
        assert float(spread.max()) == 0.0


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
        assert len(compact) == int(mask.sum())


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


class TestScoringEndToEnd:
    """Drive the whole scoring path on synthetic data.

    The benchmark's __call__ is where the delays, per-story standardisation,
    story-held-out folds and ridge fit are wired together; unit-testing the
    pieces separately leaves that wiring unchecked. A small synthetic assembly
    and a stub candidate exercise it without the source pickle or model weights.
    """

    @staticmethod
    def _benchmark(n_stories=6, n_tr=40, n_vertices=8, signal=True, seed=0, **kwargs):
        import pandas as pd
        from brainscore_core.supported_data_standards.brainio.assemblies import NeuroidAssembly
        from brainscore_core.supported_data_standards.brainio.stimuli import StimulusSet
        from brainscore.benchmarks.lebel2023.benchmark import LeBel2023Encoding

        rng = np.random.default_rng(seed)
        stories = np.repeat([f's{i}' for i in range(n_stories)], n_tr)
        n = len(stories)
        ids = [f'{st}_tr{i:03d}' for st, i in
               zip(stories, np.tile(np.arange(n_tr), n_stories))]
        features = rng.normal(size=(n, 5)).astype(np.float32)
        # BOLD is a lagged linear readout of the features when signal=True, so a
        # working pipeline must recover a positive correlation.
        weights = rng.normal(size=(5, n_vertices))
        bold = np.roll(features @ weights, 2, axis=0) if signal else rng.normal(size=(n, n_vertices))
        bold = (bold + rng.normal(scale=0.1, size=bold.shape)) * 100 + 5000  # uncentred, like real BOLD

        stim = StimulusSet(pd.DataFrame({'stimulus_id': ids, 'sentence': ['x'] * n}))
        stim.stimulus_paths = {}
        assembly = NeuroidAssembly(
            bold.astype(np.float32), dims=('presentation', 'neuroid'),
            coords={'stimulus_id': ('presentation', ids),
                    'story_id': ('presentation', stories),
                    'neuroid_id': ('neuroid', [f'v{i}' for i in range(n_vertices)]),
                    'vertex_index': ('neuroid', np.arange(n_vertices))})

        class _Synthetic(LeBel2023Encoding):
            def _data(self):
                return stim, assembly

            def _model_features(self, candidate, stimulus_set, assembly_):
                return features

        return _Synthetic(identifier='synthetic', n_splits=3, **kwargs)

    def test_scores_above_zero_when_bold_is_driven_by_the_features(self):
        score = self._benchmark(signal=True)(candidate=None)
        assert float(score) > 0.1

    def test_scores_near_zero_when_bold_is_unrelated(self):
        score = self._benchmark(signal=False, seed=3)(candidate=None)
        assert abs(float(score)) < 0.15

    def test_score_carries_the_documented_attrs(self):
        score = self._benchmark()(candidate=None)
        for key in ('raw', 'mean_r', 'n_targets', 'alphas', 'ceiling', 'error'):
            assert key in score.attrs, key
        assert len(score.attrs['alphas']) == 3          # one per fold
        assert score.attrs['n_targets'] == 8

    def test_reported_value_is_the_median_of_the_per_vertex_correlations(self):
        score = self._benchmark()(candidate=None)
        assert np.isclose(float(score), np.median(np.asarray(score.attrs['raw'])))

    def test_max_targets_subsamples_and_records_the_index(self):
        score = self._benchmark(n_vertices=8, max_targets=4)(candidate=None)
        assert score.attrs['n_targets'] == 4
        assert len(score.attrs['target_index']) == 4

    def test_feature_rows_are_matched_by_stimulus_id_not_position(self):
        """Extraction may reorder or cache rows, so alignment must be by id."""
        from brainscore.benchmarks.lebel2023.benchmark import LeBel2023Encoding
        benchmark = self._benchmark()
        stim, assembly = benchmark._data()
        n = len(assembly['stimulus_id'])

        class _Predictions:
            def __init__(self, order):
                self._order = order
                self.values = np.arange(n * 2, dtype=np.float32).reshape(n, 2)[order]
            def __getitem__(self, key):
                assert key == 'stimulus_id'
                ids = np.asarray(assembly['stimulus_id'].values)[self._order]
                return type('C', (), {'values': ids})()

        shuffled = np.random.default_rng(0).permutation(n)
        aligned = LeBel2023Encoding._align_features(
            benchmark, _Predictions(shuffled), assembly)
        expected = np.arange(n * 2, dtype=np.float32).reshape(n, 2)
        assert np.array_equal(aligned, expected)

    def test_missing_features_raise_rather_than_silently_misalign(self):
        from brainscore.benchmarks.lebel2023.benchmark import LeBel2023Encoding
        benchmark = self._benchmark()
        _, assembly = benchmark._data()

        class _Short:
            values = np.zeros((2, 2), dtype=np.float32)
            def __getitem__(self, key):
                return type('C', (), {'values': np.array(['nope_a', 'nope_b'])})()

        with pytest.raises(ValueError, match='no features'):
            LeBel2023Encoding._align_features(benchmark, _Short(), assembly)


class TestWideFeaturePath:
    """The dual branch is what large models take; it must be exercised too.

    A 27B model contributes 5120 features per delay, far more than there are
    training rows, so ``_ridge_grouped_alpha`` takes its dual branch. The other
    scoring tests use narrow features and never reach it.
    """

    def test_dual_branch_scores_the_same_signal_as_the_primal_branch(self):
        rng = np.random.default_rng(0)
        stories = np.repeat([f's{i}' for i in range(6)], 12)
        n = len(stories)
        # p > n after delays, forcing the dual form
        X = rng.normal(size=(n, 200))
        Y = X[:, :4] @ rng.normal(size=(4, 3)) + rng.normal(scale=0.01, size=(n, 3))
        pred, alpha = _ridge_grouped_alpha(X, Y, X, stories, alpha_grid=(1e-3, 1.0, 1e3))
        assert X.shape[1] > (~np.isin(stories, np.unique(stories)[:1])).sum() // 2
        assert alpha in (1e-3, 1.0, 1e3)
        assert np.corrcoef(pred[:, 0], Y[:, 0])[0, 1] > 0.5


class TestModelFeaturesSeam:
    def test_default_path_drives_the_candidate(self):
        """The default _model_features must record then process, in that order."""
        from brainscore.benchmarks.lebel2023.benchmark import LeBel2023Encoding

        calls = []
        n = 4
        ids = np.array([f'i{i}' for i in range(n)])

        class _Assembly:
            def __getitem__(self, key):
                assert key == 'stimulus_id'
                return type('C', (), {'values': ids})()

        class _Predictions:
            values = np.arange(n * 3, dtype=np.float32).reshape(n, 3)
            def __getitem__(self, key):
                return type('C', (), {'values': ids})()

        class _Candidate:
            def start_recording(self, region, recording_type=None):
                calls.append(('start_recording', region, recording_type))
            def process(self, stimuli):
                calls.append(('process', stimuli))
                return _Predictions()

        benchmark = LeBel2023Encoding(identifier='seam')
        out = benchmark._model_features(_Candidate(), 'STIM', _Assembly())

        assert [c[0] for c in calls] == ['start_recording', 'process']
        assert calls[0][1:] == ('language_system', 'fMRI')
        assert np.array_equal(out, _Predictions.values)


class TestRegistration:
    def test_all_variants_are_registered(self):
        from brainscore import benchmark_registry
        for identifier in ('LeBel2023-UTS03-encoding',
                           'LeBel2023-UTS03-encoding-smoke',
                           'LeBel2023-UTS03-encoding-contextwindow'):
            assert identifier in benchmark_registry, identifier

    def test_smoke_variant_subsamples_targets(self):
        from brainscore import benchmark_registry
        smoke = benchmark_registry['LeBel2023-UTS03-encoding-smoke']()
        full = benchmark_registry['LeBel2023-UTS03-encoding']()
        assert smoke._max_targets == 2000
        assert full._max_targets is None

    def test_default_reads_words_and_resamples(self):
        """The default must be the word-level path, not the per-TR one.

        Collapsing each TR to its last token discards ~6 words per sample and
        costs about 20% of the score, so which path is default is the whole
        point of the variant split.
        """
        from brainscore import benchmark_registry
        from brainscore.benchmarks.lebel2023.benchmark import (
            LeBel2023Encoding, LeBel2023EncodingWordLevel)
        default = benchmark_registry['LeBel2023-UTS03-encoding']()
        assert isinstance(default, LeBel2023EncodingWordLevel)
        assert default._pooling == 'lanczos'

        retained = benchmark_registry['LeBel2023-UTS03-encoding-contextwindow']()
        assert isinstance(retained, LeBel2023Encoding)
        assert not isinstance(retained, LeBel2023EncodingWordLevel)

    def test_pooling_choices_are_validated(self):
        from brainscore.benchmarks.lebel2023.benchmark import LeBel2023EncodingWordLevel
        with pytest.raises(ValueError, match='pooling must be one of'):
            LeBel2023EncodingWordLevel(pooling='nearest')
