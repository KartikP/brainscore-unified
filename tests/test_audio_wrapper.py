"""Unit tests for AudioWrapper — the fifth activations_model wrapper.

Covers the pieces that don't require downloading a real audio checkpoint:
- construction contract (identifier, backbone_id, sampling_rate resolution)
- ``_extract_paths`` stimulus-column detection
- ``_aggregate`` reduces (B, T, H) correctly under each mode
- ``_load_waveform`` truncates clips longer than max_duration_sec
- cache key propagation uses backbone_id (not identifier)
- column-to-modality map knows ``audio_path`` / ``audio_file_name``

A small real forward-pass test against a tiny HuggingFace model (e.g.
``facebook/wav2vec2-base``) is deferred to an EC2 smoke run, not CI.
"""

from unittest.mock import MagicMock

import numpy as np
import pytest
import pandas as pd


# ── Construction ────────────────────────────────────────────────────

class _FakeProcessor:
    """Mimics a HF audio processor: exposes sampling_rate, returns a dict
    of tensors when called."""

    def __init__(self, sampling_rate=16000):
        self.sampling_rate = sampling_rate

    def __call__(self, waveforms, sampling_rate, return_tensors='pt',
                 padding=True):
        import torch
        max_len = max(len(w) for w in waveforms)
        # shape (B, T) for Wav2Vec2-style models
        arr = np.zeros((len(waveforms), max_len), dtype=np.float32)
        for i, w in enumerate(waveforms):
            arr[i, :len(w)] = w
        attention_mask = np.zeros_like(arr, dtype=np.int64)
        for i, w in enumerate(waveforms):
            attention_mask[i, :len(w)] = 1
        return {
            'input_values': torch.from_numpy(arr),
            'attention_mask': torch.from_numpy(attention_mask),
        }


class _FakeModel:
    """nn.Module stand-in with a .to() returning self, a .eval() no-op,
    and a named-children hook target."""

    def __init__(self):
        import torch.nn as nn
        self._encoder = nn.Identity()
        self.encoder = self._encoder

    def to(self, device):
        return self

    def eval(self):
        pass

    def __call__(self, **kwargs):
        # Do nothing — the tests inject hook outputs manually via
        # _run_one_batch mocking when a real forward is needed.
        pass


@pytest.fixture
def wrapper():
    from brainscore.model_helpers.audio_wrapper import AudioWrapper
    import torch.nn as nn

    model = nn.Sequential(nn.Identity())  # real nn.Module for .to()
    processor = _FakeProcessor(sampling_rate=16000)
    return AudioWrapper(
        model=model, processor=processor,
        identifier='test-wrapper', backbone_id='test-backbone',
    )


class TestConstruction:

    def test_identifier_and_backbone_id(self, wrapper):
        assert wrapper.identifier == 'test-wrapper'
        assert wrapper.backbone_id == 'test-backbone'

    def test_default_backbone_id_equals_identifier(self):
        from brainscore.model_helpers.audio_wrapper import AudioWrapper
        import torch.nn as nn
        w = AudioWrapper(
            model=nn.Identity(), processor=_FakeProcessor(),
            identifier='audio-a',
        )
        assert w.backbone_id == 'audio-a'

    def test_sampling_rate_from_processor(self, wrapper):
        assert wrapper.target_sample_rate == 16000

    def test_sampling_rate_from_feature_extractor_attr(self):
        """Wav2Vec2-style processors nest sampling_rate under feature_extractor."""
        from brainscore.model_helpers.audio_wrapper import AudioWrapper
        import torch.nn as nn

        class _NestedProcessor:
            def __init__(self):
                self.feature_extractor = type(
                    'FE', (), {'sampling_rate': 24000})()

            def __call__(self, *a, **kw):
                pass

        w = AudioWrapper(
            model=nn.Identity(), processor=_NestedProcessor(),
            identifier='w',
        )
        assert w.target_sample_rate == 24000

    def test_missing_sampling_rate_raises(self):
        from brainscore.model_helpers.audio_wrapper import AudioWrapper
        import torch.nn as nn
        processor = type('P', (), {'__call__': lambda self, *a, **k: None})()
        with pytest.raises(ValueError, match="could not determine"):
            AudioWrapper(
                model=nn.Identity(), processor=processor, identifier='w',
            )

    def test_invalid_aggregation_raises(self):
        from brainscore.model_helpers.audio_wrapper import AudioWrapper
        import torch.nn as nn
        with pytest.raises(ValueError, match="layer_aggregation"):
            AudioWrapper(
                model=nn.Identity(), processor=_FakeProcessor(),
                identifier='w', layer_aggregation='invalid',
            )


# ── Stimulus column detection ───────────────────────────────────────

class TestExtractPaths:
    """_extract_paths should find audio paths under any of the known
    column names (audio_path, audio_file_name, audio_file, filename)."""

    def test_audio_path_column(self, wrapper):
        df = pd.DataFrame({
            'stimulus_id': ['a', 'b'],
            'audio_path': ['x.wav', 'y.wav'],
        })
        assert wrapper._extract_paths(df) == ['x.wav', 'y.wav']

    def test_audio_file_name_column(self, wrapper):
        df = pd.DataFrame({
            'stimulus_id': ['a'],
            'audio_file_name': ['x.wav'],
        })
        assert wrapper._extract_paths(df) == ['x.wav']

    def test_audio_file_column(self, wrapper):
        df = pd.DataFrame({
            'stimulus_id': ['a'],
            'audio_file': ['x.wav'],
        })
        assert wrapper._extract_paths(df) == ['x.wav']

    def test_no_audio_column_raises(self, wrapper):
        df = pd.DataFrame({
            'stimulus_id': ['a'],
            'sentence': ['nope'],
        })
        with pytest.raises(ValueError, match="No audio column"):
            wrapper._extract_paths(df)


# ── Aggregation semantics ───────────────────────────────────────────

class TestAggregate:

    def test_mean_time_reduces_temporal(self, wrapper):
        # (B=2, T=5, H=3)
        act = np.arange(2 * 5 * 3, dtype=np.float32).reshape(2, 5, 3)
        out = wrapper._aggregate({'l': act}, attention_mask=None)
        # unweighted mean over T
        assert out['l'].shape == (2, 3)
        np.testing.assert_allclose(out['l'], act.mean(axis=1))

    def test_mean_time_with_matching_attention_mask(self, wrapper):
        import torch
        act = np.ones((2, 4, 2), dtype=np.float32)
        act[0, 2:, :] = 100  # padded-region values we want masked out
        mask = torch.tensor([[1, 1, 0, 0], [1, 1, 1, 1]], dtype=torch.int64)
        out = wrapper._aggregate({'l': act}, attention_mask=mask)
        # Row 0: only first 2 steps count → mean = 1
        np.testing.assert_allclose(out['l'][0], [1.0, 1.0])
        # Row 1: all 4 steps — uniform ones
        np.testing.assert_allclose(out['l'][1], [1.0, 1.0])

    def test_mean_time_with_mismatched_mask_falls_back(self, wrapper):
        """If model output T differs from input mask T (conv downsampling),
        fall back to unweighted mean."""
        import torch
        act = np.ones((1, 3, 2), dtype=np.float32)
        # mask has 8 steps but activations only have 3 — conv stride
        mask = torch.ones((1, 8), dtype=torch.int64)
        out = wrapper._aggregate({'l': act}, attention_mask=mask)
        np.testing.assert_allclose(out['l'], act.mean(axis=1))

    def test_time_series_preserves_temporal(self):
        from brainscore.model_helpers.audio_wrapper import AudioWrapper
        import torch.nn as nn
        w = AudioWrapper(
            model=nn.Identity(), processor=_FakeProcessor(),
            identifier='w', layer_aggregation='time_series',
        )
        act = np.random.randn(3, 7, 4).astype(np.float32)
        out = w._aggregate({'l': act}, attention_mask=None)
        assert out['l'].shape == (3, 7, 4)
        np.testing.assert_array_equal(out['l'], act)

    def test_pooled_output_passes_through(self, wrapper):
        # Some layers emit (B, H) directly — no time axis to reduce.
        act = np.random.randn(2, 5).astype(np.float32)
        out = wrapper._aggregate({'l': act}, attention_mask=None)
        np.testing.assert_array_equal(out['l'], act)

    def test_bad_rank_raises(self, wrapper):
        act = np.random.randn(2, 3, 4, 5).astype(np.float32)
        with pytest.raises(ValueError, match="Unexpected activation rank"):
            wrapper._aggregate({'l': act}, attention_mask=None)


# ── Waveform loading + truncation ───────────────────────────────────

class TestWaveformLoading:

    def test_truncates_over_max_duration(self):
        from brainscore.model_helpers.audio_wrapper import AudioWrapper
        import torch.nn as nn
        long_wav = np.zeros(16000 * 70, dtype=np.float32)  # 70s @ 16k

        w = AudioWrapper(
            model=nn.Identity(), processor=_FakeProcessor(16000),
            identifier='w',
            max_duration_sec=60.0,
            audio_loader=lambda path, sr: long_wav,
        )
        with pytest.warns(UserWarning, match="truncating"):
            out = w._load_waveform('fake.wav')
        assert out.shape == (16000 * 60,)

    def test_does_not_truncate_when_disabled(self):
        from brainscore.model_helpers.audio_wrapper import AudioWrapper
        import torch.nn as nn
        long_wav = np.zeros(16000 * 70, dtype=np.float32)
        w = AudioWrapper(
            model=nn.Identity(), processor=_FakeProcessor(16000),
            identifier='w', max_duration_sec=None,
            audio_loader=lambda path, sr: long_wav,
        )
        out = w._load_waveform('fake.wav')
        assert out.shape == long_wav.shape


# ── Cache key propagation ───────────────────────────────────────────

class TestCacheKey:

    def test_from_paths_cached_uses_backbone_id(self, wrapper):
        captured = {}

        def fake_stored(identifier, stimuli_identifier, layers, paths):
            captured['identifier'] = identifier
            return 'sentinel'

        wrapper._from_paths_stored = fake_stored
        result = wrapper._from_paths_cached(
            paths=['x.wav'], layers=['enc'], stimuli_identifier='stim')
        assert result == 'sentinel'
        assert captured['identifier'] == 'test-backbone'

    def test_different_backbone_ids_dont_collide(self):
        from brainscore.model_helpers.audio_wrapper import AudioWrapper
        import torch.nn as nn
        a = AudioWrapper(model=nn.Identity(), processor=_FakeProcessor(),
                         identifier='a', backbone_id='wav2vec-bert')
        b = AudioWrapper(model=nn.Identity(), processor=_FakeProcessor(),
                         identifier='b', backbone_id='hubert')
        assert a.backbone_id != b.backbone_id


# ── Column-to-modality registry ─────────────────────────────────────

class TestColumnToModality:

    def test_audio_columns_registered(self):
        from brainscore_core.model_interface import BrainScoreModel
        mapping = BrainScoreModel.COLUMN_TO_MODALITY
        assert mapping.get('audio_path') == 'audio'
        assert mapping.get('audio_file_name') == 'audio'
        assert mapping.get('audio_file') == 'audio'
