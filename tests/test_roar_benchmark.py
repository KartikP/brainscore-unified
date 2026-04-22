"""Smoke tests for the ROAR (Yeatman2021) lexical decision benchmark.

Validates that the benchmark loads, the stimulus/assembly shapes are right,
the ceiling is computed, and scoring integrates end-to-end with
BrainScoreModel's behavioral readout.
"""

import numpy as np
import pytest


class TestRoarYeatman2021:
    @pytest.fixture(scope='class')
    def benchmark(self):
        import brainscore
        return brainscore.load_benchmark('Yeatman2021-lexical_decision')

    def test_registered(self):
        import brainscore
        assert 'Yeatman2021-lexical_decision' in brainscore.benchmark_registry

    def test_stimulus_set(self, benchmark):
        stim = benchmark._stimulus_set
        assert len(stim) == 500
        assert set(stim['image_label'].unique()) == {'real', 'pseudo'}
        # Balanced real vs pseudo
        counts = stim['image_label'].value_counts()
        assert abs(counts['real'] - counts['pseudo']) < 50

    def test_human_accuracy_range(self, benchmark):
        acc = benchmark._human_accuracy
        # Should be realistic human performance (well above chance, below ceiling)
        assert 0.7 < acc.mean() < 0.9
        assert acc.min() >= 0.0
        assert acc.max() <= 1.0
        # Spread should be non-trivial
        assert acc.std() > 0.05

    def test_ceiling(self, benchmark):
        ceiling = float(benchmark.ceiling)
        # Split-half reliability for 120 subjects should be high
        assert 0.85 < ceiling < 1.0

    def test_score_clip(self, benchmark):
        """End-to-end: fit behavioral readout on CLIP, score against humans."""
        import brainscore
        import brainscore_vision  # register CLIP
        model = brainscore.load_model('clip-vit-b-32')
        score = benchmark(model)

        # Sanity checks on the score
        raw = float(score.attrs['raw'])
        ceiled = float(score)
        ceiling = score.attrs['ceiling']
        p_value = score.attrs['p_value']

        assert -1.0 <= raw <= 1.0
        assert score.attrs['n_stimuli'] == 500
        assert 0.85 < ceiling < 1.0
        # CLIP should produce a significant positive correlation (features
        # capture some word-ness structure)
        assert raw > 0.1, f"expected positive correlation, got raw r={raw}"
        assert p_value < 0.01
        # ceiled == raw / ceiling
        assert np.isclose(ceiled, raw / ceiling)
