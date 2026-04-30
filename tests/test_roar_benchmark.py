"""Tests for the ROAR (Yeatman2021) lexical decision benchmark.

Protocol replicates Honarmand et al. (2026 ICLR): 400 train / 100 test,
accuracy metric, 65% dyslexia threshold.

Per the April 30, 2026 design decision, ROAR is registered as two pinned
variants: ``Yeatman2021-lexical_decision-image`` and
``Yeatman2021-lexical_decision-text``. The legacy
``Yeatman2021-lexical_decision`` identifier is a deprecated alias that
defaults to the image variant.
"""

import warnings

import numpy as np
import pytest


class TestRoarYeatman2021:
    @pytest.fixture(scope='class')
    def benchmark(self):
        import brainscore
        return brainscore.load_benchmark('Yeatman2021-lexical_decision-image')

    def test_image_variant_registered(self):
        import brainscore
        assert 'Yeatman2021-lexical_decision-image' in brainscore.benchmark_registry

    def test_text_variant_registered(self):
        import brainscore
        assert 'Yeatman2021-lexical_decision-text' in brainscore.benchmark_registry

    def test_legacy_alias_registered(self):
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


class TestParameterizedModality:
    """Direct construction with the `modality` parameter (toolbox path).

    Each variant projects the stimulus set so model-side modality detection
    picks unambiguously, and the train/test partition is shared across
    variants so cross-variant scientific comparisons remain valid.
    """

    def test_default_modality_is_vision(self):
        from brainscore.benchmarks.roar_yeatman2021.benchmark import (
            Yeatman2021LexicalDecision,
        )
        bm = Yeatman2021LexicalDecision()
        assert bm.modality == 'vision'
        assert bm.required_modalities == {'vision'}

    def test_vision_variant_columns(self):
        from brainscore.benchmarks.roar_yeatman2021.benchmark import (
            Yeatman2021LexicalDecision,
        )
        bm = Yeatman2021LexicalDecision(modality='vision')
        assert bm.required_modalities == {'vision'}
        assert 'image_file_name' in bm._train_stimuli.columns
        assert 'sentence' not in bm._train_stimuli.columns
        assert 'image_file_name' in bm._test_stimuli.columns
        assert 'sentence' not in bm._test_stimuli.columns

    def test_text_variant_columns(self):
        from brainscore.benchmarks.roar_yeatman2021.benchmark import (
            Yeatman2021LexicalDecision,
        )
        bm = Yeatman2021LexicalDecision(modality='text')
        assert bm.required_modalities == {'text'}
        assert 'sentence' in bm._train_stimuli.columns
        assert 'image_file_name' not in bm._train_stimuli.columns
        assert 'sentence' in bm._test_stimuli.columns
        assert 'image_file_name' not in bm._test_stimuli.columns

    def test_invalid_modality_raises(self):
        from brainscore.benchmarks.roar_yeatman2021.benchmark import (
            Yeatman2021LexicalDecision,
        )
        with pytest.raises(AssertionError, match="modality must be one of"):
            Yeatman2021LexicalDecision(modality='audio')

    def test_variants_share_split(self):
        """Train/test split is consistent across modality variants — only
        the presentation differs, so cross-variant comparisons remain valid."""
        from brainscore.benchmarks.roar_yeatman2021.benchmark import (
            Yeatman2021LexicalDecision,
        )
        bm_vision = Yeatman2021LexicalDecision(modality='vision')
        bm_text = Yeatman2021LexicalDecision(modality='text')
        assert (sorted(bm_vision._train_stimuli['stimulus_id'].values)
                == sorted(bm_text._train_stimuli['stimulus_id'].values))
        assert (sorted(bm_vision._test_stimuli['stimulus_id'].values)
                == sorted(bm_text._test_stimuli['stimulus_id'].values))

    def test_variant_identifiers_distinct(self):
        from brainscore.benchmarks.roar_yeatman2021.benchmark import (
            Yeatman2021LexicalDecision,
        )
        bm_vision = Yeatman2021LexicalDecision(modality='vision')
        bm_text = Yeatman2021LexicalDecision(modality='text')
        assert bm_vision.identifier == 'Yeatman2021-lexical_decision-image'
        assert bm_text.identifier == 'Yeatman2021-lexical_decision-text'

    def test_variants_share_ceiling(self):
        """Both variants are scored against the same human accuracy on the
        same 100 test stimulus_ids, so the ceiling is identical."""
        from brainscore.benchmarks.roar_yeatman2021.benchmark import (
            Yeatman2021LexicalDecision,
        )
        bm_vision = Yeatman2021LexicalDecision(modality='vision')
        bm_text = Yeatman2021LexicalDecision(modality='text')
        assert np.isclose(bm_vision._human_accuracy, bm_text._human_accuracy)


class TestLegacyAlias:
    """Legacy `Yeatman2021-lexical_decision` identifier should still load,
    emit a DeprecationWarning, and default to the image variant."""

    def test_alias_emits_deprecation_warning(self):
        import brainscore
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            brainscore.load_benchmark('Yeatman2021-lexical_decision')
        assert any(
            issubclass(w.category, DeprecationWarning)
            and 'Yeatman2021-lexical_decision' in str(w.message)
            for w in caught
        )

    def test_alias_defaults_to_image_variant(self):
        import brainscore
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', DeprecationWarning)
            bm = brainscore.load_benchmark('Yeatman2021-lexical_decision')
        assert bm.modality == 'vision'
        assert bm.identifier == 'Yeatman2021-lexical_decision-image'
