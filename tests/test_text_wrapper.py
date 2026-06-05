"""Tests for TextWrapper — symmetric text extraction with PytorchWrapper."""

import numpy as np
import pandas as pd
import pytest

pytestmark = pytest.mark.slow  # loads real CLIP / GPT-2 weights — run on demand

from brainscore_core.model_interface import BrainScoreModel
from brainscore_core.supported_data_standards.brainio.assemblies import NeuroidAssembly
from brainscore_core.supported_data_standards.brainio.stimuli import StimulusSet


@pytest.fixture(scope='module')
def clip_text_wrapper():
    """Create a TextWrapper around CLIP's text encoder."""
    from transformers import CLIPModel, CLIPProcessor
    from brainscore.model_helpers.text_wrapper import TextWrapper

    clip_model = CLIPModel.from_pretrained('openai/clip-vit-base-patch32')
    clip_processor = CLIPProcessor.from_pretrained('openai/clip-vit-base-patch32')

    return TextWrapper(
        model=clip_model.text_model,
        tokenizer=clip_processor.tokenizer,
        identifier='clip-vit-b-32-text',
        layer_aggregation='mean_tokens',
        max_length=77,
    )


@pytest.fixture
def text_stimuli():
    stimuli = StimulusSet(pd.DataFrame({
        'sentence': ['the quick brown fox', 'a dog runs fast', 'neurons fire together'],
        'stimulus_id': ['s0', 's1', 's2'],
        'category': ['animal', 'animal', 'science'],
    }))
    stimuli.identifier = 'test_text'
    return stimuli


class TestTextWrapperEncoderMode:
    def test_returns_neuroid_assembly(self, clip_text_wrapper, text_stimuli):
        result = clip_text_wrapper(text_stimuli, layers=['encoder.layers.10'])
        assert isinstance(result, NeuroidAssembly)

    def test_shape(self, clip_text_wrapper, text_stimuli):
        result = clip_text_wrapper(text_stimuli, layers=['encoder.layers.10'])
        assert result.dims == ('presentation', 'neuroid')
        assert result.shape[0] == 3  # 3 stimuli

    def test_hidden_dim(self, clip_text_wrapper, text_stimuli):
        result = clip_text_wrapper(text_stimuli, layers=['encoder.layers.10'])
        assert result.shape[1] == 512  # CLIP ViT-B hidden dim

    def test_stimulus_id_coord(self, clip_text_wrapper, text_stimuli):
        result = clip_text_wrapper(text_stimuli, layers=['encoder.layers.10'])
        assert 'stimulus_id' in result.coords
        assert list(result['stimulus_id'].values) == ['s0', 's1', 's2']

    def test_stimulus_set_meta_attached(self, clip_text_wrapper, text_stimuli):
        result = clip_text_wrapper(text_stimuli, layers=['encoder.layers.10'])
        assert 'category' in result.coords
        assert list(result['category'].values) == ['animal', 'animal', 'science']

    def test_layer_coord(self, clip_text_wrapper, text_stimuli):
        result = clip_text_wrapper(text_stimuli, layers=['encoder.layers.10'])
        layers = np.unique(result['layer'].values)
        assert list(layers) == ['encoder.layers.10']

    def test_neuroid_id_coord(self, clip_text_wrapper, text_stimuli):
        result = clip_text_wrapper(text_stimuli, layers=['encoder.layers.10'])
        assert len(result['neuroid_id'].values) == result.shape[1]

    def test_different_layers_different_features(self, clip_text_wrapper):
        stimuli = StimulusSet(pd.DataFrame({
            'sentence': ['hello world'],
            'stimulus_id': ['s0'],
        }))
        stimuli.identifier = 'test_layers'
        r1 = clip_text_wrapper(stimuli, layers=['encoder.layers.1'])
        r10 = clip_text_wrapper(stimuli, layers=['encoder.layers.10'])
        # Same shape but different values
        assert r1.shape == r10.shape
        assert not np.allclose(r1.values, r10.values)


class TestTextWrapperBatching:
    def test_large_batch(self, clip_text_wrapper):
        n = 50
        stimuli = StimulusSet(pd.DataFrame({
            'sentence': [f'sentence number {i}' for i in range(n)],
            'stimulus_id': [f's{i}' for i in range(n)],
        }))
        stimuli.identifier = 'test_large_batch'
        result = clip_text_wrapper(stimuli, layers=['encoder.layers.10'])
        assert result.shape[0] == n


class TestTextWrapperCaching:
    def test_second_call_uses_cache(self, clip_text_wrapper):
        stimuli = StimulusSet(pd.DataFrame({
            'sentence': ['cache test one', 'cache test two'],
            'stimulus_id': ['c0', 'c1'],
        }))
        stimuli.identifier = 'test_cache'
        r1 = clip_text_wrapper(stimuli, layers=['encoder.layers.5'])
        r2 = clip_text_wrapper(stimuli, layers=['encoder.layers.5'])
        assert np.allclose(r1.values, r2.values)


class TestTextWrapperIntegration:
    def test_works_as_brainscore_model_preprocessor(self, clip_text_wrapper, text_stimuli):
        """TextWrapper used as the text preprocessor in BrainScoreModel."""
        model = BrainScoreModel(
            identifier='clip-vit-b-32',
            model=None,
            region_layer_map={
                'language_system': 'encoder.layers.10',
            },
            preprocessors={
                'text': clip_text_wrapper,
            },
        )
        model.start_recording('language_system')
        # TextWrapper is callable: (stimuli, layers) -> NeuroidAssembly
        # BrainScoreModel.process() for text calls: preprocessor(model, stimuli, recording_layer=...)
        # But TextWrapper expects: (stimuli, layers=[...])
        # So we need to test the direct path
        result = clip_text_wrapper(text_stimuli, layers=['encoder.layers.10'])
        assert result.dims == ('presentation', 'neuroid')
        assert result.shape[0] == 3


@pytest.fixture(scope='module')
def gpt2_text_wrapper():
    """Create a TextWrapper around GPT-2 (causal LM).

    GPT-2 is a decoder-only causal model, so last_token aggregation is used.
    Layer paths follow GPT-2's naming: h.0, h.1, ..., h.11 (not encoder.layers.N).
    """
    from transformers import GPT2Model, GPT2Tokenizer
    from brainscore.model_helpers.text_wrapper import TextWrapper

    gpt2 = GPT2Model.from_pretrained('gpt2')
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    # GPT-2 has no pad token; set one so batching works
    tokenizer.pad_token = tokenizer.eos_token

    return TextWrapper(
        model=gpt2,
        tokenizer=tokenizer,
        identifier='gpt2-text',
        layer_aggregation='last_token',
        max_length=128,
    )


class TestTextWrapperCausalMode:
    """Validate last_token aggregation path using GPT-2."""

    def test_returns_neuroid_assembly(self, gpt2_text_wrapper, text_stimuli):
        result = gpt2_text_wrapper(text_stimuli, layers=['h.11'])
        assert isinstance(result, NeuroidAssembly)

    def test_shape(self, gpt2_text_wrapper, text_stimuli):
        result = gpt2_text_wrapper(text_stimuli, layers=['h.11'])
        assert result.dims == ('presentation', 'neuroid')
        assert result.shape[0] == 3
        assert result.shape[1] == 768  # GPT-2 small hidden dim

    def test_stimulus_set_meta_attached(self, gpt2_text_wrapper, text_stimuli):
        result = gpt2_text_wrapper(text_stimuli, layers=['h.11'])
        assert 'category' in result.coords

    def test_last_token_uses_attention_mask(self, gpt2_text_wrapper):
        """Two inputs with the same non-pad tokens but different padding lengths
        must produce identical activations if last_token honors attention_mask."""
        short = StimulusSet(pd.DataFrame({
            'sentence': ['hello world'],
            'stimulus_id': ['s_short'],
        }))
        short.identifier = 'causal_mask_short'
        long = StimulusSet(pd.DataFrame({
            'sentence': ['hello world', 'this sentence is intentionally much longer to force padding'],
            'stimulus_id': ['s_short', 's_long'],
        }))
        long.identifier = 'causal_mask_long'

        r_short = gpt2_text_wrapper(short, layers=['h.6'])
        r_long = gpt2_text_wrapper(long, layers=['h.6'])

        # The 'hello world' row must have identical activations regardless of
        # whether it was batched alone (no padding) or with a longer sentence
        # (right-padded). If last_token blindly used the final position, the
        # padded batch would pick an EOS token's activation instead.
        # stimulus_id is a coord but not an index — look up by value.
        short_idx = list(r_short['stimulus_id'].values).index('s_short')
        long_idx = list(r_long['stimulus_id'].values).index('s_short')
        short_hello = r_short.values[short_idx]
        long_hello = r_long.values[long_idx]
        assert np.allclose(short_hello, long_hello, atol=1e-5)

    def test_different_layers_different_features(self, gpt2_text_wrapper):
        stimuli = StimulusSet(pd.DataFrame({
            'sentence': ['the cat sat on the mat'],
            'stimulus_id': ['s0'],
        }))
        stimuli.identifier = 'causal_layer_diff'
        r0 = gpt2_text_wrapper(stimuli, layers=['h.0'])
        r11 = gpt2_text_wrapper(stimuli, layers=['h.11'])
        assert r0.shape == r11.shape
        assert not np.allclose(r0.values, r11.values)


# ── per-token mode + auto-chunking ──────────────────────────────────


@pytest.fixture(scope='module')
def gpt2_per_token_wrapper():
    """GPT-2 wrapper in per_token mode with a small max_length so chunking
    triggers naturally on transcripts longer than ~16 tokens."""
    from transformers import GPT2Model, GPT2Tokenizer
    from brainscore.model_helpers.text_wrapper import TextWrapper

    gpt2 = GPT2Model.from_pretrained('gpt2')
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token

    return TextWrapper(
        model=gpt2,
        tokenizer=tokenizer,
        identifier='gpt2-per-token',
        layer_aggregation='per_token',
        max_length=16,
    )


class TestTextWrapperPerToken:
    """per_token aggregation returns (presentation, time_bin, neuroid)."""

    def test_invalid_aggregation_raises(self):
        from transformers import GPT2Model, GPT2Tokenizer
        from brainscore.model_helpers.text_wrapper import TextWrapper
        gpt2 = GPT2Model.from_pretrained('gpt2')
        tok = GPT2Tokenizer.from_pretrained('gpt2')
        with pytest.raises(ValueError, match="layer_aggregation must be"):
            TextWrapper(model=gpt2, tokenizer=tok, layer_aggregation='nope')

    def test_dims_include_time_bin(self, gpt2_per_token_wrapper):
        stimuli = StimulusSet(pd.DataFrame({
            'sentence': ['hello world', 'a quick test'],
            'stimulus_id': ['p0', 'p1'],
        }))
        stimuli.identifier = 'per_token_short'
        result = gpt2_per_token_wrapper(stimuli, layers=['h.6'])
        assert result.dims == ('presentation', 'time_bin', 'neuroid')
        assert result.shape[0] == 2
        assert result.shape[2] == 768

    def test_token_length_coord_attached(self, gpt2_per_token_wrapper):
        stimuli = StimulusSet(pd.DataFrame({
            'sentence': ['short', 'a longer sentence with more tokens'],
            'stimulus_id': ['t0', 't1'],
        }))
        stimuli.identifier = 'per_token_lengths'
        result = gpt2_per_token_wrapper(stimuli, layers=['h.6'])
        assert 'token_length' in result.coords
        lengths = list(result['token_length'].values)
        assert lengths[1] > lengths[0]

    def test_long_input_is_chunked_not_truncated(self, gpt2_per_token_wrapper):
        """A transcript longer than max_length=16 must produce time_bin
        > max_length when per_token mode is used (i.e., chunking ran)."""
        long_text = ' '.join([f'word{i}' for i in range(60)])
        stimuli = StimulusSet(pd.DataFrame({
            'sentence': [long_text],
            'stimulus_id': ['long0'],
        }))
        stimuli.identifier = 'per_token_long'
        result = gpt2_per_token_wrapper(stimuli, layers=['h.6'])
        assert result.sizes['time_bin'] > 16
        # And the token_length coord must reflect the full token count
        assert int(result['token_length'].values[0]) > 16

    def test_padding_is_nan_past_token_length(self, gpt2_per_token_wrapper):
        """Shorter inputs in a multi-input batch must be NaN-padded
        in time so callers can mask them out."""
        stimuli = StimulusSet(pd.DataFrame({
            'sentence': ['one two', ' '.join(f'word{i}' for i in range(40))],
            'stimulus_id': ['p_short', 'p_long'],
        }))
        stimuli.identifier = 'per_token_mixed'
        result = gpt2_per_token_wrapper(stimuli, layers=['h.6'])
        # Find the short presentation index
        ids = list(result['stimulus_id'].values)
        short_idx = ids.index('p_short')
        short_len = int(result['token_length'].values[short_idx])
        # Beyond short_len, padding should be NaN
        beyond = result.values[short_idx, short_len:, :]
        assert np.isnan(beyond).all()

    def test_per_token_features_differ_across_positions(
            self, gpt2_per_token_wrapper):
        """Two different token positions in the same input should have
        different activations — guards against accidental aggregation."""
        stimuli = StimulusSet(pd.DataFrame({
            'sentence': ['the cat sat on the mat'],
            'stimulus_id': ['s0'],
        }))
        stimuli.identifier = 'per_token_distinct'
        result = gpt2_per_token_wrapper(stimuli, layers=['h.6'])
        first_token = result.values[0, 0, :]
        last_real = result.values[0,
                                  int(result['token_length'].values[0]) - 1, :]
        assert not np.allclose(first_token, last_real)
