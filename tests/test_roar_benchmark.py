"""Tests for the ROAR (Yeatman2021) lexical decision benchmark.

Protocol replicates Honarmand et al. (2026 ICLR): 400 train / 100 test,
accuracy metric, 65% dyslexia threshold.
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

    def test_split_sizes(self, benchmark):
        assert len(benchmark._train_stimuli) == 400
        assert len(benchmark._test_stimuli) == 100
        train_counts = benchmark._train_stimuli['image_label'].value_counts().to_dict()
        test_counts = benchmark._test_stimuli['image_label'].value_counts().to_dict()
        assert train_counts['real'] == 200 and train_counts['pseudo'] == 200
        assert test_counts['real'] == 50 and test_counts['pseudo'] == 50

    def test_no_train_test_leakage(self, benchmark):
        train_ids = set(benchmark._train_stimuli['stimulus_id'])
        test_ids = set(benchmark._test_stimuli['stimulus_id'])
        assert not (train_ids & test_ids), "train and test overlap"

    def test_human_accuracy_on_test(self, benchmark):
        """Human accuracy on the 100-stim test set should be in the
        expected range (mean ~0.80 overall)."""
        acc = benchmark._human_accuracy
        assert 0.6 < acc < 0.95, f"unexpected human accuracy: {acc}"

    def test_dyslexia_threshold(self, benchmark):
        from brainscore.benchmarks.roar_yeatman2021.benchmark import DYSLEXIA_THRESHOLD
        assert DYSLEXIA_THRESHOLD == 0.65

    def test_score_clip(self, benchmark):
        """End-to-end: fit behavioral readout on CLIP train set, measure
        test accuracy, check metric structure."""
        import brainscore
        import brainscore_vision  # register CLIP
        model = brainscore.load_model('clip-vit-b-32')
        score = benchmark(model)

        raw = float(score.attrs['raw'])
        ceiled = float(score)
        ceiling = score.attrs['ceiling']

        # Accuracy is in [0, 1]
        assert 0.0 <= raw <= 1.0
        assert score.attrs['n_test_stimuli'] == 100
        # Human accuracy is ceiling
        assert 0.6 < ceiling < 0.95
        # Ceiled score is raw / ceiling
        assert np.isclose(ceiled, raw / ceiling)
        # CLIP should do better than chance
        assert raw > 0.55, f"CLIP raw accuracy too low: {raw}"
        # Per-class accuracies are reported
        assert 'accuracy_real' in score.attrs
        assert 'accuracy_pseudo' in score.attrs
        # Dyslexia flag present
        assert isinstance(score.attrs['dyslexic'], bool)
        assert score.attrs['dyslexia_threshold'] == 0.65
