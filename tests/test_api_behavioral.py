"""Unit tests for closed-weight API behavioral models — no network, no API key.

Uses a mock callable provider, so these exercise the full closure logic (modality
dispatch, response caching, label parsing) and the registration wiring without
ever hitting a live API.
"""
import pandas as pd
import pytest

from brainscore.model_helpers.api_behavioral import (
    build_api_generation_fn, _parse_label, _find_column, _build_instruction,
    IMAGE_COLUMNS, TEXT_COLUMNS,
)


class TestPureHelpers:
    def test_parse_exact_case_insensitive(self):
        assert _parse_label('REAL', ['real', 'pseudo']) == 'real'
        assert _parse_label('  pseudo ', ['real', 'pseudo']) == 'pseudo'

    def test_parse_word_boundary_substring(self):
        assert _parse_label('The answer is real.', ['real', 'pseudo']) == 'real'
        # word boundary: "unreal" must NOT match "real"
        assert _parse_label('unreal', ['real', 'pseudo']) is None

    def test_parse_unparseable_returns_none(self):
        assert _parse_label('maybe?', ['real', 'pseudo']) is None
        assert _parse_label(None, ['real', 'pseudo']) is None

    def test_build_instruction_has_forced_choice(self):
        out = _build_instruction('Is this a real word?', ['real', 'pseudo'])
        assert 'real, pseudo' in out
        assert 'exactly one' in out.lower()

    def test_find_column_prefers_image_then_text(self):
        row = pd.Series({'image_file_name': '/x.png', 'sentence': 'hi'})
        assert _find_column(row, IMAGE_COLUMNS) == 'image_file_name'
        text_only = pd.Series({'sentence': 'hi'})
        assert _find_column(text_only, IMAGE_COLUMNS) is None
        assert _find_column(text_only, TEXT_COLUMNS) == 'sentence'

    def test_find_column_skips_nan(self):
        row = pd.Series({'image_file_name': float('nan'), 'sentence': 'hi'})
        assert _find_column(row, IMAGE_COLUMNS) is None  # NaN is not a usable path


class TestGenerationClosure:
    def _mock(self, responses=None, counter=None):
        """A provider adapter that records calls and returns canned responses."""
        responses = responses or {}
        def call(model, system, user_text, image, max_tokens):
            if counter is not None:
                counter.append((model, user_text, image))
            # default: echo a label found in the prompt, else 'real'
            return responses.get('return', 'real')
        return call

    def test_text_row_returns_parsed_label(self):
        gen = build_api_generation_fn(self._mock({'return': 'pseudo'}), 'mock')
        row = pd.Series({'sentence': 'florp', 'stimulus_id': 's1'})
        out = gen(row, 'Is this a real word?', ['real', 'pseudo'])
        assert out == 'pseudo'

    def test_text_dispatch_includes_stimulus_text(self):
        calls = []
        gen = build_api_generation_fn(self._mock(counter=calls), 'mock')
        row = pd.Series({'sentence': 'florp', 'stimulus_id': 's1'})
        gen(row, 'Is this real?', ['real', 'pseudo'])
        _model, user_text, image = calls[0]
        assert 'florp' in user_text       # the stimulus text is in the prompt
        assert image is None              # text row -> no image payload

    def test_unparseable_returns_raw_for_caller_fallback(self):
        gen = build_api_generation_fn(self._mock({'return': 'I cannot say'}), 'mock')
        row = pd.Series({'sentence': 'x', 'stimulus_id': 's1'})
        out = gen(row, 'Is this real?', ['real', 'pseudo'])
        # raw string passed through; BrainScoreModel warns + defaults to label_set[0]
        assert out == 'I cannot say'

    def test_missing_modality_column_raises(self):
        gen = build_api_generation_fn(self._mock(), 'mock')
        row = pd.Series({'unrelated': 7, 'stimulus_id': 's1'})
        with pytest.raises(ValueError, match='no readable image column'):
            gen(row, 'Is this real?', ['real', 'pseudo'])

    def test_cache_avoids_resecond_call(self, tmp_path):
        calls = []
        gen = build_api_generation_fn(
            self._mock(counter=calls), 'mock', cache_dir=str(tmp_path))
        row = pd.Series({'sentence': 'florp', 'stimulus_id': 's1'})
        a = gen(row, 'Is this real?', ['real', 'pseudo'])
        b = gen(row, 'Is this real?', ['real', 'pseudo'])
        assert a == b == 'real'
        assert len(calls) == 1            # second call served from disk cache

    def test_cache_distinguishes_stimuli(self, tmp_path):
        calls = []
        gen = build_api_generation_fn(
            self._mock(counter=calls), 'mock', cache_dir=str(tmp_path))
        gen(pd.Series({'sentence': 'a', 'stimulus_id': 's1'}),
            'Is this real?', ['real', 'pseudo'])
        gen(pd.Series({'sentence': 'b', 'stimulus_id': 's2'}),
            'Is this real?', ['real', 'pseudo'])
        assert len(calls) == 2            # different stimuli -> separate calls

    def test_image_row_encodes_and_sends_image(self, tmp_path):
        # 1x1 PNG
        png = (b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00'
               b'\x01\x08\x06\x00\x00\x00\x1f\x15\xc4\x89\x00\x00\x00\nIDATx\x9cc'
               b'\x00\x01\x00\x00\x05\x00\x01\r\n-\xb4\x00\x00\x00\x00IEND\xaeB`\x82')
        img = tmp_path / 'stim.png'
        img.write_bytes(png)
        calls = []
        gen = build_api_generation_fn(self._mock(counter=calls), 'mock')
        row = pd.Series({'image_file_name': str(img), 'stimulus_id': 's1'})
        gen(row, 'Which object?', ['left', 'right'])
        _model, _text, image = calls[0]
        assert image is not None
        media, b64 = image
        assert media == 'image/png' and len(b64) > 0


class TestRegistration:
    def test_models_registered(self):
        import brainscore
        for ident in ('claude-opus-behavioral', 'claude-haiku-behavioral',
                      'gpt-4o-behavioral', 'deepseek-chat-behavioral',
                      'deepseek-r1-behavioral'):
            assert ident in brainscore.model_registry

    def test_deepseek_is_text_only(self):
        # DeepSeek's API has no vision; it must declare text only so a vision
        # benchmark doesn't route to it at the compatibility check.
        from brainscore.models.api_closed.model import get_model
        m = get_model('deepseek-r1-behavioral')
        assert m.supported_modalities == {'text'}
        assert callable(m._generation_fn)

    def test_deepseek_provider_is_openai_compatible(self):
        # The DeepSeek provider is the OpenAI-compatible adapter pointed at a
        # different base_url — same wire protocol.
        from brainscore.model_helpers.api_behavioral import PROVIDERS
        assert 'deepseek' in PROVIDERS
        assert PROVIDERS['deepseek'].__name__ == '_call_deepseek'

    def test_openrouter_provider_and_registration(self):
        # OpenRouter is the same OpenAI-compatible gateway, pointed at hundreds
        # of models. The provider exists and example models register.
        import brainscore
        from brainscore.model_helpers.api_behavioral import PROVIDERS
        from brainscore.models.api_closed.model import get_model
        assert PROVIDERS['openrouter'].__name__ == '_call_openrouter'
        assert 'llama-3.3-70b-behavioral' in brainscore.model_registry
        # an OpenRouter model that IS multimodal can declare vision
        m = get_model('or-claude-sonnet-behavioral')
        assert m.supported_modalities == {'vision', 'text'}
        # a text-only OpenRouter model declares text only
        assert get_model('llama-3.3-70b-behavioral').supported_modalities == {'text'}

    def test_get_model_wiring_no_api_call(self):
        # Building the model must NOT require an API key — the closure is lazy.
        from brainscore.models.api_closed.model import get_model
        m = get_model('claude-opus-behavioral')
        assert m.supported_modalities == {'vision', 'text'}
        assert callable(m._generation_fn)
        assert m._activations_model is None

    def test_neural_extraction_raises_clear_error(self):
        from brainscore.models.api_closed.model import get_model
        m = get_model('claude-opus-behavioral')
        # The nominal preprocessor exists only to declare modality support; if a
        # neural benchmark tries to extract activations it must fail clearly.
        with pytest.raises(NotImplementedError, match='closed-weight API model'):
            m._preprocessors['vision'](m._model, None, recording_layer=None)

    def test_unknown_identifier_raises(self):
        from brainscore.models.api_closed.model import get_model
        with pytest.raises(ValueError, match='unknown closed-API model'):
            get_model('claude-not-a-real-model')
