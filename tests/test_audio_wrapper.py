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

    def test_load_returns_full_clip_no_truncation(self):
        """Loader returns the full waveform — chunking happens later."""
        from brainscore.model_helpers.audio_wrapper import AudioWrapper
        import torch.nn as nn
        long_wav = np.zeros(16000 * 70, dtype=np.float32)  # 70s @ 16k
        w = AudioWrapper(
            model=nn.Identity(), processor=_FakeProcessor(16000),
            identifier='w', max_duration_sec=60.0,
            audio_loader=lambda path, sr: long_wav,
        )
        out = w._load_waveform('fake.wav')
        assert out.shape == long_wav.shape


class TestWaveformChunking:
    """Long clips are split into chunks rather than truncated."""

    def test_short_clip_single_chunk(self):
        from brainscore.model_helpers.audio_wrapper import AudioWrapper
        import torch.nn as nn
        wav = np.zeros(16000 * 30, dtype=np.float32)  # 30s
        w = AudioWrapper(
            model=nn.Identity(), processor=_FakeProcessor(16000),
            identifier='w', max_duration_sec=60.0,
        )
        chunks = w._chunk_waveform(wav)
        assert len(chunks) == 1
        # No copy: chunks[0] is the original array (or a view)
        assert chunks[0].shape == wav.shape

    def test_long_clip_split_at_max_duration(self):
        from brainscore.model_helpers.audio_wrapper import AudioWrapper
        import torch.nn as nn
        wav = np.arange(16000 * 150, dtype=np.float32)  # 150s
        w = AudioWrapper(
            model=nn.Identity(), processor=_FakeProcessor(16000),
            identifier='w', max_duration_sec=60.0,
        )
        chunks = w._chunk_waveform(wav)
        # 150s / 60s/chunk = 2 full + 1 partial = 3 chunks
        assert len(chunks) == 3
        assert chunks[0].shape == (16000 * 60,)
        assert chunks[1].shape == (16000 * 60,)
        assert chunks[2].shape == (16000 * 30,)
        # Concatenating chunks must reproduce the original waveform
        np.testing.assert_array_equal(np.concatenate(chunks), wav)

    def test_chunking_disabled_when_max_duration_none(self):
        from brainscore.model_helpers.audio_wrapper import AudioWrapper
        import torch.nn as nn
        wav = np.zeros(16000 * 200, dtype=np.float32)  # 200s
        w = AudioWrapper(
            model=nn.Identity(), processor=_FakeProcessor(16000),
            identifier='w', max_duration_sec=None,
        )
        chunks = w._chunk_waveform(wav)
        assert len(chunks) == 1
        assert chunks[0].shape == wav.shape

    def test_recombine_2d_per_clip_mean_across_chunks(self):
        """For (n_chunks, H) layer outputs (mean_time / pooled), recombine
        averages across chunks belonging to the same clip."""
        from brainscore.model_helpers.audio_wrapper import AudioWrapper
        import torch.nn as nn
        w = AudioWrapper(
            model=nn.Identity(), processor=_FakeProcessor(16000),
            identifier='w', max_duration_sec=60.0,
        )
        # 4 chunks: clip 0 has chunks {0,1,2}, clip 1 has chunk {3}
        chunk_arr = np.array([[1, 1], [3, 3], [5, 5], [10, 20]],
                             dtype=np.float32)
        out = w._recombine_2d(chunk_arr,
                              clip_chunks=[[0, 1, 2], [3]], n_clips=2)
        assert out.shape == (2, 2)
        np.testing.assert_allclose(out[0], [3.0, 3.0])  # mean of 1,3,5
        np.testing.assert_allclose(out[1], [10.0, 20.0])

    def test_recombine_3d_per_clip_concat_along_time(self):
        """For (n_chunks, T, H) layer outputs (time_series), recombine
        concatenates along time and NaN-pads shorter clips."""
        from brainscore.model_helpers.audio_wrapper import AudioWrapper
        import torch.nn as nn
        w = AudioWrapper(
            model=nn.Identity(), processor=_FakeProcessor(16000),
            identifier='w', max_duration_sec=60.0,
            layer_aggregation='time_series',
        )
        # 3 chunks @ T=2, H=2. Clip 0 = chunks {0,1} → T=4. Clip 1 = {2} → T=2.
        chunk_arr = np.array([
            [[1, 1], [2, 2]],
            [[3, 3], [4, 4]],
            [[7, 8], [9, 10]],
        ], dtype=np.float32)
        out = w._recombine_3d(chunk_arr,
                              clip_chunks=[[0, 1], [2]], n_clips=2)
        assert out.shape == (2, 4, 2)  # T_max = 4
        np.testing.assert_allclose(out[0],
                                   [[1, 1], [2, 2], [3, 3], [4, 4]])
        # Clip 1 has T=2 valid + NaN padding for T=2,3
        np.testing.assert_allclose(out[1, :2], [[7, 8], [9, 10]])
        assert np.isnan(out[1, 2:]).all()


# ── Cache key propagation ───────────────────────────────────────────

class TestCacheKey:

    def test_from_paths_cached_uses_backbone_id(self, wrapper):
        captured = {}

        def fake_stored(identifier, stimuli_identifier, layers, paths, extraction_fingerprint):
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


# ── Temporal coords on time_series output ──────────────────────────


class TestTemporalCoords:
    """Verify time_bin_start_ms / time_bin_end_ms coords are derived
    from the model's per-step duration after the first forward pass."""

    def test_step_ms_unset_before_any_forward_pass(self, wrapper):
        assert wrapper._step_ms is None

    def test_pack_3d_omits_time_coords_when_step_ms_unset(self, wrapper):
        # Synthetic time_series output — _step_ms left as None
        act = np.zeros((2, 5, 4), dtype=np.float32)
        assembly = wrapper._pack_3d(
            act, layer_name='enc', paths=['a', 'b'], n_time=5, n_features=4)
        assert 'time_bin_start_ms' not in assembly.coords
        assert 'time_bin_end_ms' not in assembly.coords

    def test_pack_3d_attaches_time_coords_when_step_ms_set(self, wrapper):
        wrapper._step_ms = 20.0  # mimic a Wav2Vec2-style 50 Hz output
        act = np.zeros((1, 4, 3), dtype=np.float32)
        assembly = wrapper._pack_3d(
            act, layer_name='enc', paths=['a'], n_time=4, n_features=3)
        np.testing.assert_allclose(
            assembly['time_bin_start_ms'].values, [0.0, 20.0, 40.0, 60.0])
        np.testing.assert_allclose(
            assembly['time_bin_end_ms'].values, [20.0, 40.0, 60.0, 80.0])

    def test_run_one_batch_sets_step_ms_from_input_output_ratio(self):
        """First forward pass derives step_ms from longest input clip
        length and the hook output's time axis."""
        from brainscore.model_helpers.audio_wrapper import AudioWrapper
        import torch
        import torch.nn as nn

        class _StridedFakeModel(nn.Module):
            """nn.Module that emits a (B, T_out, H) hook output."""
            def __init__(self, t_out=50, hidden=4):
                super().__init__()
                self._t_out = t_out
                self._hidden = hidden
                self.encoder = nn.Identity()  # hook target

            def forward(self, **kwargs):
                B = kwargs['input_values'].shape[0]
                # Trigger the hook by passing through a tensor with the
                # downsampled time axis. Hook is on `self.encoder` which
                # is Identity — so whatever we pass through it appears
                # as the captured output.
                t = torch.zeros(B, self._t_out, self._hidden)
                self.encoder(t)
                return None

        model = _StridedFakeModel(t_out=50, hidden=4)
        # 1s @ 16kHz input → 50 output steps means step_ms = 1000/50 = 20
        w = AudioWrapper(
            model=model, processor=_FakeProcessor(16000),
            identifier='strided', layer_aggregation='time_series',
        )
        wav = np.zeros(16000, dtype=np.float32)  # 1s
        w._run_one_batch([wav], layers=['encoder'])
        assert w._step_ms == pytest.approx(20.0, rel=1e-6)

    def test_pack_3d_time_coords_independent_of_layer_name(self, wrapper):
        """time_bin_start_ms / time_bin_end_ms only depend on _step_ms
        and n_time, not the layer name — important for multi-layer
        outputs that get concatenated along neuroid."""
        wrapper._step_ms = 25.0
        act = np.zeros((1, 3, 2), dtype=np.float32)
        a1 = wrapper._pack_3d(
            act, layer_name='layer.1', paths=['a'], n_time=3, n_features=2)
        a2 = wrapper._pack_3d(
            act, layer_name='layer.2', paths=['a'], n_time=3, n_features=2)
        np.testing.assert_allclose(
            a1['time_bin_start_ms'].values, a2['time_bin_start_ms'].values)
        np.testing.assert_allclose(
            a1['time_bin_end_ms'].values, a2['time_bin_end_ms'].values)
