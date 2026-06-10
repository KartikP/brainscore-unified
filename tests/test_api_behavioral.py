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


class TestGameActionFn:
    """build_api_action_fn drives the embodied game via process(EnvironmentStep)."""

    def _env_step(self, n_actions=3, ascii_board=None):
        import numpy as np
        from brainscore_core.model_interface import EnvironmentStep
        obs = {
            'frame': np.zeros((8, 8, 3), dtype='uint8'),
            'instruction': 'reach the goal',
            'legal_actions': {i: f'move {i}' for i in range(n_actions)},
        }
        if ascii_board is not None:
            obs['ascii'] = ascii_board
        return EnvironmentStep(
            observation=obs, instruction='reach the goal',
            is_first=True, step_num=0)

    def _mock(self, response='Action: 2', counter=None):
        def call(model, system, user_text, image, max_tokens):
            if counter is not None:
                counter.append((user_text, image))
            return response
        return call

    def test_parses_action_index(self):
        from brainscore.model_helpers.api_behavioral import build_api_action_fn
        from brainscore_core.model_interface import EnvironmentResponse
        act = build_api_action_fn(self._mock('Let me think... Action: 2'), 'mock')
        resp = act(self._env_step(n_actions=3))
        assert isinstance(resp, EnvironmentResponse)
        assert int(resp.action) == 2

    def test_frame_sent_as_image(self):
        from brainscore.model_helpers.api_behavioral import build_api_action_fn
        calls = []
        act = build_api_action_fn(self._mock('Action: 1', counter=calls), 'mock')
        act(self._env_step())
        user_text, image = calls[0]
        assert image is not None and image[0] == 'image/png'   # frame -> PNG
        assert 'reach the goal' in user_text                   # mission in prompt

    def test_unparseable_falls_back_to_legal_random(self):
        from brainscore.model_helpers.api_behavioral import build_api_action_fn
        act = build_api_action_fn(self._mock('no idea'), 'mock', fallback_seed=0)
        resp = act(self._env_step(n_actions=3))
        assert 0 <= int(resp.action) < 3                       # random but legal

    def test_action_cache(self, tmp_path):
        from brainscore.model_helpers.api_behavioral import build_api_action_fn
        calls = []
        act = build_api_action_fn(self._mock('Action: 0', counter=calls), 'mock',
                                  cache_dir=str(tmp_path))
        s = self._env_step()
        a, b = act(s), act(s)
        assert int(a.action) == int(b.action) == 0
        assert len(calls) == 1                                 # identical frame -> cached

    def test_missing_frame_raises(self):
        from brainscore.model_helpers.api_behavioral import build_api_action_fn
        from brainscore_core.model_interface import EnvironmentStep
        act = build_api_action_fn(self._mock(), 'mock')
        with pytest.raises(ValueError, match="no 'frame'"):
            act(EnvironmentStep(observation={'instruction': 'x'}, step_num=0))

    def test_ascii_mode_sends_board_no_image(self):
        from brainscore.model_helpers.api_behavioral import build_api_action_fn
        calls = []
        act = build_api_action_fn(self._mock('Action: 1', counter=calls), 'mock',
                                  obs_mode='ascii')
        resp = act(self._env_step(n_actions=3, ascii_board='P . .\n. . G'))
        user_text, image = calls[0]
        assert image is None                  # ascii mode -> no image payload
        assert 'P . .' in user_text           # the text board is in the prompt
        assert int(resp.action) == 1

    def test_ascii_mode_missing_board_raises(self):
        from brainscore.model_helpers.api_behavioral import build_api_action_fn
        act = build_api_action_fn(self._mock(), 'mock', obs_mode='ascii')
        with pytest.raises(ValueError, match="no 'ascii'"):
            act(self._env_step(n_actions=3))   # frame-only step, no ascii

    def test_trace_captures_reasoning(self):
        from brainscore.model_helpers.api_behavioral import build_api_action_fn
        act = build_api_action_fn(
            self._mock('The goal is below me. Action: 1'), 'mock')
        assert act.trace == []                          # empty before any tick
        act(self._env_step(n_actions=3))
        assert len(act.trace) == 1
        rec = act.trace[0]
        assert rec['action'] == 1 and rec['fallback'] is False
        assert 'The goal is below me' in rec['response']  # reasoning preserved
        assert rec['step'] == 0

    def test_trace_flags_fallback_moves(self):
        from brainscore.model_helpers.api_behavioral import build_api_action_fn
        act = build_api_action_fn(self._mock('no idea'), 'mock', fallback_seed=0)
        act(self._env_step(n_actions=3))
        assert act.trace[0]['fallback'] is True          # unparseable -> random
        assert 0 <= act.trace[0]['action'] < 3

    def test_history_window_zero_omits_history(self):
        from brainscore.model_helpers.api_behavioral import build_api_action_fn
        calls = []
        act = build_api_action_fn(self._mock('Action: 2', counter=calls), 'mock')
        act(self._env_step(n_actions=3)); act(self._env_step(n_actions=3))
        assert 'recent moves' not in calls[1][0].lower()   # memoryless by default

    def test_history_window_includes_recent_moves(self):
        from brainscore.model_helpers.api_behavioral import build_api_action_fn
        calls = []
        act = build_api_action_fn(self._mock('Action: 2', counter=calls), 'mock',
                                  history_window=8)
        act(self._env_step(n_actions=3))                   # tick 0: no history yet
        assert 'recent moves' not in calls[0][0].lower()
        act(self._env_step(n_actions=3))                   # tick 1: history present
        assert 'recent moves' in calls[1][0].lower()
        assert 'move 2' in calls[1][0]                     # the prior action's label

    def test_history_flags_noop_when_view_unchanged(self):
        from brainscore.model_helpers.api_behavioral import build_api_action_fn
        calls = []
        act = build_api_action_fn(self._mock('Action: 2', counter=calls), 'mock',
                                  obs_mode='ascii', history_window=8)
        board = 'P . .\n. . G'
        act(self._env_step(n_actions=3, ascii_board=board))
        act(self._env_step(n_actions=3, ascii_board=board))   # identical board
        assert 'did NOT change' in calls[1][0]                # blocked/no-op flagged

    def test_history_flags_repeated_action(self):
        from brainscore.model_helpers.api_behavioral import build_api_action_fn
        calls = []
        act = build_api_action_fn(self._mock('Action: 2', counter=calls), 'mock',
                                  obs_mode='ascii', history_window=8)
        # three identical ticks -> 4th prompt warns about the repeated action
        for board in ('a', 'b', 'c', 'd'):
            act(self._env_step(n_actions=3, ascii_board=board))
        assert 'times in a row' in calls[3][0]


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

    def test_text_only_model_sent_image_gives_actionable_error(self, monkeypatch):
        # A text-only OpenAI-compatible model (DeepSeek-V3, Llama) sent an image
        # returns a provider 404; the adapter must re-raise with the real fix.
        from brainscore.model_helpers import api_behavioral as ab

        class _FakeNotFound(Exception):
            pass

        class _FakeClient:
            def __init__(self, *a, **k):
                self.chat = self
                self.completions = self

            def create(self, **kwargs):
                raise _FakeNotFound(
                    "Error code: 404 - No endpoints found that support image input")

        import sys, types
        fake_openai = types.SimpleNamespace(OpenAI=_FakeClient)
        monkeypatch.setitem(sys.modules, 'openai', fake_openai)
        call = ab._make_openai_compatible(
            base_url='https://api.deepseek.com', label='deepseek')
        with pytest.raises(RuntimeError, match="has no image-input endpoint"):
            call('deepseek-chat', None, 'hi', ('image/png', 'Zm9v'), 16)

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
