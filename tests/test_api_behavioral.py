"""Unit tests for closed-weight API behavioral models — no network, no API key.

Uses a mock callable provider, so these exercise the full closure logic (modality
dispatch, response caching, label parsing) and the registration wiring without
ever hitting a live API.
"""
import pandas as pd
import pytest

from brainscore_core.compatibility import ExecutionNotFeasibleError
from brainscore.model_helpers.api_behavioral import (
    api_answer, _find_column, _answer_text, _row_parts, _CACHE_FORM,
)


class TestPureHelpers:
    def test_the_provider_text_is_returned_exactly(self):
        """The answer part does not parse: BrainScoreModel reads the text with Choice.parse."""
        for text in ('REAL', '  Pseudo, not REAL  ', 'The answer is real.'):
            gen = api_answer(lambda model, system, messages, settings: text, 'mock')
            assert gen(pd.Series({'sentence': 'florp', 'stimulus_id': 's1'}), 'Real?',
                       ['real', 'pseudo']) == text

    def test_prompt_is_the_instruction_then_core_choice_directive(self):
        assert _answer_text('Is this a real word?', ['real', 'pseudo']) == (
            'Is this a real word?\n\nAnswer with exactly one of: real, pseudo.')
        assert _answer_text(None, ['real', 'pseudo']) == 'Answer with exactly one of: real, pseudo.'

    def test_find_column_reads_the_catalog_columns_in_order(self):
        row = pd.Series({'image_file_name': '/x.png', 'sentence': 'hi'})
        assert _find_column(row, 'image') == 'image_file_name'
        assert _find_column(pd.Series({'image_path': '/x.png'}), 'image') == 'image_path'
        text_only = pd.Series({'sentence': 'hi'})
        assert _find_column(text_only, 'image') is None
        assert _find_column(text_only, 'text') == 'sentence'
        # a name outside the catalog is not read (stated change: the old lists had it)
        assert _find_column(pd.Series({'filepath': '/x.png', 'word': 'hi'}), 'image') is None

    def test_find_column_skips_nan(self):
        row = pd.Series({'image_file_name': float('nan'), 'sentence': 'hi'})
        assert _find_column(row, 'image') is None  # NaN is not a usable path

    def test_row_parts_names_the_accepted_columns_when_the_row_has_neither(self):
        assert _row_parts(pd.Series({'image_path': '/x.png', 'text': 'hi'}), 'part') == ('image_path', 'text')
        with pytest.raises(ValueError, match=r"part: the stimulus row has no image column and no text "
                                             r"column; accepted columns, in order: image \['image_file_name', "
                                             r"'image_path', 'filename', 'image', 'vision'\], text \['sentence', 'text'\]"):
            _row_parts(pd.Series({'notes': 'x'}), 'part')


def _turn(messages):
    """The text and the image (or None) of the one user turn an answer or policy part sends."""
    (turn,) = messages
    text = next(part['text'] for part in turn['content'] if part['type'] == 'text')
    images = [(part['media_type'], part['data']) for part in turn['content'] if part['type'] == 'image']
    return text, images[0] if images else None


class TestGenerationClosure:
    def _mock(self, responses=None, counter=None):
        """A provider client that records calls and returns canned responses."""
        responses = responses or {}
        def call(model, system, messages, settings):
            user_text, image = _turn(messages)
            if counter is not None:
                counter.append((model, user_text, image))
            # default: echo a label found in the prompt, else 'real'
            return responses.get('return', 'real')
        return call

    def test_text_row_returns_the_provider_text(self):
        gen = api_answer(self._mock({'return': 'pseudo'}), 'mock')
        row = pd.Series({'sentence': 'florp', 'stimulus_id': 's1'})
        out = gen(row, 'Is this a real word?', ['real', 'pseudo'])
        assert out == 'pseudo'

    def test_text_dispatch_includes_stimulus_text(self):
        calls = []
        gen = api_answer(self._mock(counter=calls), 'mock')
        row = pd.Series({'sentence': 'florp', 'stimulus_id': 's1'})
        gen(row, 'Is this real?', ['real', 'pseudo'])
        _model, user_text, image = calls[0]
        assert 'florp' in user_text       # the stimulus text is in the prompt
        assert image is None              # text row -> no image payload

    def test_unparseable_returns_raw_for_caller_fallback(self):
        gen = api_answer(self._mock({'return': 'I cannot say'}), 'mock')
        row = pd.Series({'sentence': 'x', 'stimulus_id': 's1'})
        out = gen(row, 'Is this real?', ['real', 'pseudo'])
        # raw string passed through; BrainScoreModel records it as an invalid answer
        assert out == 'I cannot say'

    def test_missing_modality_column_raises(self):
        gen = api_answer(self._mock(), 'mock')
        row = pd.Series({'unrelated': 7, 'stimulus_id': 's1'})
        with pytest.raises(ValueError, match='no image column and no text column'):
            gen(row, 'Is this real?', ['real', 'pseudo'])

    def test_cache_avoids_resecond_call(self, tmp_path):
        calls = []
        gen = api_answer(
            self._mock(counter=calls), 'mock', cache_dir=str(tmp_path))
        row = pd.Series({'sentence': 'florp', 'stimulus_id': 's1'})
        a = gen(row, 'Is this real?', ['real', 'pseudo'])
        b = gen(row, 'Is this real?', ['real', 'pseudo'])
        assert a == b == 'real'
        assert len(calls) == 1            # second call served from disk cache

    def test_the_cache_key_holds_the_prompt_so_an_entry_of_the_old_prompt_is_not_used(self, tmp_path):
        """Stated change: the prompt text changed, so a cached reply of the old prompt misses once."""
        import hashlib, json
        calls = []
        provider = self._mock(counter=calls)
        payload = hashlib.sha256(b'florp').hexdigest()[:16]

        def key(prompt, *form):      # the old key had no form part
            raw = json.dumps([provider.__name__, 'mock', prompt, ['real', 'pseudo'], 's1', payload, None, 16,
                              *form])
            return hashlib.sha256(raw.encode('utf-8')).hexdigest()

        old = ('Is this real?\n\nAnswer with exactly one of: real, pseudo. '
               'Reply with only that word, nothing else.')
        (tmp_path / f'{key(old)}.json').write_text(json.dumps({'response': 'pseudo'}))
        gen = api_answer(provider, 'mock', cache_dir=str(tmp_path))
        assert gen(pd.Series({'sentence': 'florp', 'stimulus_id': 's1'}), 'Is this real?',
                   ['real', 'pseudo']) == 'real'
        assert len(calls) == 1
        new = 'Is this real?\n\nAnswer with exactly one of: real, pseudo.'
        assert (tmp_path / f'{key(new, _CACHE_FORM)}.json').exists()

    def test_cache_distinguishes_stimuli(self, tmp_path):
        calls = []
        gen = api_answer(
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
        gen = api_answer(self._mock(counter=calls), 'mock')
        row = pd.Series({'image_file_name': str(img), 'stimulus_id': 's1'})
        gen(row, 'Which object?', ['left', 'right'])
        _model, _text, image = calls[0]
        assert image is not None
        media, b64 = image
        assert media == 'image/png' and len(b64) > 0


class TestGameActionFn:
    """api_policy drives the embodied game via process(row)."""

    def _env_step(self, n_actions=3, ascii_board=None):
        import numpy as np
        obs = {
            'frame': np.zeros((8, 8, 3), dtype='uint8'),
            'instruction': 'reach the goal',
            'legal_actions': {i: f'move {i}' for i in range(n_actions)},
        }
        if ascii_board is not None:
            obs['ascii'] = ascii_board
        return {'observation': obs, 'instruction': 'reach the goal'}

    def _mock(self, response='Action: 2', counter=None):
        def call(model, system, messages, settings):
            user_text, image = _turn(messages)
            if counter is not None:
                counter.append((user_text, image))
            return response
        return call

    def test_parses_action_index(self):
        from brainscore.model_helpers.api_behavioral import api_policy
        act = api_policy(self._mock('Let me think... Action: 2'), 'mock')
        resp = act(self._env_step(n_actions=3))['motor']
        assert resp.channel == 'motor' and resp.meta['valid']
        assert int(resp.payload) == 2

    def test_frame_sent_as_image(self):
        from brainscore.model_helpers.api_behavioral import api_policy
        calls = []
        act = api_policy(self._mock('Action: 1', counter=calls), 'mock')
        act(self._env_step())
        user_text, image = calls[0]
        assert image is not None and image[0] == 'image/png'   # frame -> PNG
        assert 'reach the goal' in user_text                   # mission in prompt

    def test_unparseable_falls_back_to_legal_random(self):
        from brainscore.model_helpers.api_behavioral import api_policy
        act = api_policy(self._mock('no idea'), 'mock', fallback_seed=0)
        resp = act(self._env_step(n_actions=3))['motor']
        assert 0 <= int(resp.payload) < 3                       # random but legal

    def test_action_cache(self, tmp_path):
        from brainscore.model_helpers.api_behavioral import api_policy
        calls = []
        act = api_policy(self._mock('Action: 0', counter=calls), 'mock',
                                  cache_dir=str(tmp_path))
        s = self._env_step()
        a = act(s)['motor']
        act.reset()                                            # a new episode begins
        b = act(s)['motor']
        assert int(a.payload) == int(b.payload) == 0
        assert len(calls) == 1                                 # identical frame, same step -> cached

    def test_action_cache_keyed_by_step(self, tmp_path):
        from brainscore.model_helpers.api_behavioral import api_policy
        calls = []
        act = api_policy(self._mock('Action: 0', counter=calls), 'mock',
                                  cache_dir=str(tmp_path))
        s = self._env_step()
        act(s), act(s)
        assert len(calls) == 2                                 # same view, two steps
        act.reset()
        act(s), act(s)
        assert len(calls) == 2                                 # re-run replays from cache

    def test_missing_frame_raises(self):
        from brainscore.model_helpers.api_behavioral import api_policy
        act = api_policy(self._mock(), 'mock')
        with pytest.raises(ValueError, match="no 'frame'"):
            act({'observation': {'instruction': 'x'}})

    def test_ascii_mode_sends_board_no_image(self):
        from brainscore.model_helpers.api_behavioral import api_policy
        calls = []
        act = api_policy(self._mock('Action: 1', counter=calls), 'mock',
                                  obs_mode='ascii')
        resp = act(self._env_step(n_actions=3, ascii_board='P . .\n. . G'))['motor']
        user_text, image = calls[0]
        assert image is None                  # ascii mode -> no image payload
        assert 'P . .' in user_text           # the text board is in the prompt
        assert int(resp.payload) == 1

    def test_ascii_mode_missing_board_raises(self):
        from brainscore.model_helpers.api_behavioral import api_policy
        act = api_policy(self._mock(), 'mock', obs_mode='ascii')
        with pytest.raises(ValueError, match="no 'ascii'"):
            act(self._env_step(n_actions=3))   # frame-only step, no ascii

    def test_trace_captures_reasoning(self):
        from brainscore.model_helpers.api_behavioral import api_policy
        act = api_policy(
            self._mock('The goal is below me. Action: 1'), 'mock')
        assert act.trace == []                          # empty before any tick
        act(self._env_step(n_actions=3))
        assert len(act.trace) == 1
        rec = act.trace[0]
        assert rec['action'] == 1 and rec['fallback'] is False
        assert 'The goal is below me' in rec['response']  # reasoning preserved
        assert rec['step'] == 0

    def test_trace_flags_fallback_moves(self):
        from brainscore.model_helpers.api_behavioral import api_policy
        act = api_policy(self._mock('no idea'), 'mock', fallback_seed=0)
        act(self._env_step(n_actions=3))
        assert act.trace[0]['fallback'] is True          # unparseable -> random
        assert 0 <= act.trace[0]['action'] < 3

    def test_history_window_zero_omits_history(self):
        from brainscore.model_helpers.api_behavioral import api_policy
        calls = []
        act = api_policy(self._mock('Action: 2', counter=calls), 'mock')
        act(self._env_step(n_actions=3)); act(self._env_step(n_actions=3))
        assert 'recent moves' not in calls[1][0].lower()   # memoryless by default

    def test_history_window_includes_recent_moves(self):
        from brainscore.model_helpers.api_behavioral import api_policy
        calls = []
        act = api_policy(self._mock('Action: 2', counter=calls), 'mock',
                                  history_window=8)
        act(self._env_step(n_actions=3))                   # tick 0: no history yet
        assert 'recent moves' not in calls[0][0].lower()
        act(self._env_step(n_actions=3))                   # tick 1: history present
        assert 'recent moves' in calls[1][0].lower()
        assert 'move 2' in calls[1][0]                     # the prior action's label

    def test_history_flags_noop_when_view_unchanged(self):
        from brainscore.model_helpers.api_behavioral import api_policy
        calls = []
        act = api_policy(self._mock('Action: 2', counter=calls), 'mock',
                                  obs_mode='ascii', history_window=8)
        board = 'P . .\n. . G'
        act(self._env_step(n_actions=3, ascii_board=board))
        act(self._env_step(n_actions=3, ascii_board=board))   # identical board
        assert 'did NOT change' in calls[1][0]                # blocked/no-op flagged

    def test_history_flags_repeated_action(self):
        from brainscore.model_helpers.api_behavioral import api_policy
        calls = []
        act = api_policy(self._mock('Action: 2', counter=calls), 'mock',
                                  obs_mode='ascii', history_window=8)
        # three identical ticks -> 4th prompt warns about the repeated action
        for board in ('a', 'b', 'c', 'd'):
            act(self._env_step(n_actions=3, ascii_board=board))
        assert 'times in a row' in calls[3][0]


class TestRegistration:
    def test_deepseek_is_text_only(self):
        # DeepSeek's API has no vision; it must declare text only so a vision
        # benchmark doesn't route to it at the compatibility check.
        from brainscore.models.api_closed.model import get_model
        m = get_model('deepseek-r1-behavioral')
        assert 'text' in m.in_channels and 'vision:image' not in m.in_channels
        assert callable(m._answer)

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
            call('deepseek-chat', None, [{'role': 'user', 'content': [
                {'type': 'text', 'text': 'hi'},
                {'type': 'image', 'media_type': 'image/png', 'data': 'Zm9v'}]}], {'max_tokens': 16})

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
        assert {'vision:image', 'text'} <= m.in_channels
        # a text-only OpenRouter model declares text only
        assert 'vision:image' not in get_model('llama-3.3-70b-behavioral').in_channels

    def test_get_model_wiring_no_api_call(self):
        # Building the model must NOT require an API key — the closure is lazy.
        from brainscore.models.api_closed.model import get_model
        m = get_model('claude-opus-behavioral')
        assert {'vision:image', 'text'} <= m.in_channels
        assert callable(m._answer)
        assert dict(m.extractor) == {}

    def test_neural_extraction_raises_clear_error(self):
        from brainscore.models.api_closed.model import get_model
        m = get_model('claude-opus-behavioral')
        from brainscore_core.supported_data_standards.brainio.stimuli import StimulusSet
        # A behavioral-only model declares its inputs without pretending to extract activity.
        with pytest.raises(NotImplementedError, match='no activation extractor'):
            m.process(StimulusSet({'image_path': ['unused.png'], 'stimulus_id': ['s0']}))

    def test_unknown_identifier_raises(self):
        from brainscore.models.api_closed.model import get_model
        with pytest.raises(ValueError, match='unknown closed-API model'):
            get_model('claude-not-a-real-model')


# ── reply part: written answers under the task's settings ──────────────

PIXEL = 'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAAAAAA6fptVAAAACklEQVR4nGNgAAAAAgABSK+kcQAAAABJRU5ErkJggg=='


def _recording_provider(answer='42'):
    calls = []

    def provider(model, system, messages, settings):
        calls.append(dict(model=model, system=system, messages=messages, settings=settings))
        return answer
    return provider, calls


class TestApiReply:
    def test_parts_in_order_instruction_as_system_and_settings(self, tmp_path):
        import base64
        from brainscore.model_helpers.api_behavioral import build_api_reply
        from brainscore_core.responses import Prompt, Text
        image = tmp_path / 'p.png'
        image.write_bytes(base64.b64decode(PIXEL))
        provider, calls = _recording_provider()
        reply = build_api_reply(provider, 'm-1')
        prompt = Prompt((('text', 'Rate it.'), ('text', 'first'), ('image', str(image)), ('text', 'second')),
                        names=('instruction', 'text', 'image_path', 'text_2'),
                        response=Text(max_tokens=512, temperature=1.0, seed=3))
        assert reply(prompt) == '42'
        call = calls[0]
        assert call['system'] == 'Rate it.'
        assert call['messages'] == [{'role': 'user', 'content': [
            {'type': 'text', 'text': 'first'},
            {'type': 'image', 'media_type': 'image/png', 'data': PIXEL},
            {'type': 'text', 'text': 'second'}]}]
        assert call['settings'] == {'max_tokens': 512, 'temperature': 1.0, 'seed': 3}

    def test_no_settings_uses_its_own_length_only(self):
        from brainscore.model_helpers.api_behavioral import build_api_reply
        from brainscore_core.responses import Choice, Prompt
        provider, calls = _recording_provider()
        reply = build_api_reply(provider, 'm-1', max_tokens=64)
        reply(Prompt((('text', 'x'),), response=Choice(['a', 'b'])))
        assert calls[0]['settings'] == {'max_tokens': 64}

    def test_history_becomes_earlier_turns(self):
        from brainscore.model_helpers.api_behavioral import build_api_reply
        from brainscore_core.responses import Prompt
        provider, calls = _recording_provider()
        reply = build_api_reply(provider, 'm-1')
        earlier = Prompt((('text', 'one'),))
        reply(Prompt((('text', 'two'),), history=((earlier, 'yes'),)))
        assert [m['role'] for m in calls[0]['messages']] == ['user', 'assistant', 'user']

    def test_cache_is_keyed_on_the_settings(self, tmp_path):
        from brainscore.model_helpers.api_behavioral import build_api_reply
        from brainscore_core.responses import Prompt, Text
        provider, calls = _recording_provider()
        reply = build_api_reply(provider, 'm-1', cache_dir=tmp_path)
        shown = (('text', 'x'),)
        for response in (Text(temperature=1.0, seed=1), Text(temperature=1.0, seed=1),
                         Text(temperature=0.5, seed=1), Text(temperature=1.0, seed=1, max_tokens=8)):
            reply(Prompt(shown, response=response))
        # The repeated request is served from the cache; each new setting is a new request.
        assert len(calls) == 3

    def test_a_sampled_call_without_a_seed_is_not_cached(self, tmp_path):
        from brainscore.model_helpers.api_behavioral import build_api_reply
        from brainscore_core.responses import Prompt, Text
        calls = []

        def provider(model, system, messages, settings):
            calls.append(1)
            return f'answer {len(calls)}'
        reply = build_api_reply(provider, 'm-1', cache_dir=tmp_path)
        shown = (('text', 'x'),)
        sampled = [reply(Prompt(shown, response=Text(temperature=1.0))) for _ in range(5)]
        assert len(calls) == 5 and len(set(sampled)) == 5
        for response in (Text(temperature=1.0, seed=2), Text(temperature=0.0), Text()):
            first = reply(Prompt(shown, response=response))
            assert reply(Prompt(shown, response=response)) == first
        assert len(calls) == 8

    @pytest.mark.parametrize('settings', [{'temperature': 0.5}, {'temperature': 1.0, 'top_p': 0.9},
                                          {'seed': 1}])
    def test_anthropic_refuses_unsupported_settings_before_any_request(self, settings, monkeypatch):
        import sys
        from brainscore.model_helpers.api_behavioral import build_api_reply
        from brainscore_core.responses import Prompt, Text
        monkeypatch.setitem(sys.modules, 'anthropic', None)   # a request would fail differently
        reply = build_api_reply('anthropic', 'claude-x')
        with pytest.raises(ExecutionNotFeasibleError, match='Anthropic API refuses'):
            reply(Prompt((('text', 'x'),), response=Text(**settings)))

    def test_anthropic_default_temperature_is_not_sent(self, monkeypatch):
        import sys
        import types
        from brainscore.model_helpers.api_behavioral import build_api_reply
        from brainscore_core.responses import Prompt, Text
        sent = {}

        class Messages:
            def create(self, **kwargs):
                sent.update(kwargs)
                return types.SimpleNamespace(content=[types.SimpleNamespace(type='text', text=' ok ')])
        fake = types.SimpleNamespace(Anthropic=lambda: types.SimpleNamespace(messages=Messages()))
        monkeypatch.setitem(sys.modules, 'anthropic', fake)
        reply = build_api_reply('anthropic', 'claude-x')
        assert reply(Prompt((('text', 'Do it.'), ('text', 'x')), names=('instruction', 'text'),
                            response=Text(max_tokens=512, temperature=1.0))) == ' ok '   # the provider's text exactly
        assert sent == {'model': 'claude-x', 'max_tokens': 512, 'system': 'Do it.',
                        'messages': [{'role': 'user', 'content': [{'type': 'text', 'text': 'x'}]}]}

    def test_anthropic_reply_returns_reasoning_and_token_usage_as_extra_data(self, monkeypatch, tmp_path):
        import sys
        import types
        from brainscore.model_helpers.api_behavioral import build_api_reply
        from brainscore_core.responses import Prompt, Text
        from brainscore_core.streaming import StreamEvent
        requests = []

        class Messages:
            def create(self, **kwargs):
                requests.append(kwargs)
                return types.SimpleNamespace(
                    content=[types.SimpleNamespace(type='thinking', thinking='Add them.'),
                             types.SimpleNamespace(type='text', text=' 4 ')],
                    usage=types.SimpleNamespace(input_tokens=7, output_tokens=2))
        fake = types.SimpleNamespace(Anthropic=lambda: types.SimpleNamespace(messages=Messages()))
        monkeypatch.setitem(sys.modules, 'anthropic', fake)
        reply = build_api_reply('anthropic', 'claude-x', cache_dir=tmp_path)
        prompt = Prompt((('text', '2 + 2?'),), response=Text())
        for _ in range(2):   # the second reply comes from the cache, with its extra data
            answer = reply(prompt)
            assert isinstance(answer, StreamEvent) and answer.payload == ' 4 '   # exactly as given
            assert answer.meta == {'reasoning': 'Add them.',
                                   'token_usage': {'input_tokens': 7, 'output_tokens': 2}}
        assert len(requests) == 1

    def test_openai_reply_returns_reasoning_and_token_usage_as_extra_data(self, monkeypatch):
        import sys
        import types
        from brainscore.model_helpers.api_behavioral import _make_openai_compatible
        from brainscore_core.responses import Prompt, Text
        message = types.SimpleNamespace(content=' 4 ', reasoning_content='Add them.')
        response = types.SimpleNamespace(
            choices=[types.SimpleNamespace(message=message)],
            usage=types.SimpleNamespace(prompt_tokens=7, completion_tokens=2))
        client = types.SimpleNamespace(chat=types.SimpleNamespace(
            completions=types.SimpleNamespace(create=lambda **kwargs: response)))
        monkeypatch.setitem(sys.modules, 'openai', types.SimpleNamespace(OpenAI=lambda **kwargs: client))
        answer = _make_openai_compatible()('m', None, [{'role': 'user', 'content': []}], {'max_tokens': 8})
        assert answer.payload == ' 4 '   # exactly as given
        assert answer.meta == {'reasoning': 'Add them.',
                               'token_usage': {'input_tokens': 7, 'output_tokens': 2}}

    def test_video_parts_are_refused(self):
        from brainscore.model_helpers.api_behavioral import build_api_reply
        from brainscore_core.responses import Prompt
        provider, calls = _recording_provider()
        with pytest.raises(ExecutionNotFeasibleError, match='video'):
            build_api_reply(provider, 'm-1')(Prompt((('video', '/clip.mp4'),)))
        assert calls == []

    def test_each_api_model_has_one_identifier_with_both_parts(self):
        import brainscore
        from brainscore.models.api_closed.model import REGISTRY, get_model
        assert set(REGISTRY) <= set(brainscore.model_registry)
        assert not any(identifier.endswith('-reply') for identifier in brainscore.model_registry)
        model = get_model('claude-opus-behavioral')
        assert callable(model._answer) and callable(model._reply)
        assert 'behavior:text' in model.out_channels
        assert {'vision:image', 'text'} <= model.in_channels

    def test_an_api_model_answers_text_through_its_reply_and_a_plain_choice_through_its_answer(
            self, tmp_path, monkeypatch):
        """No network: a fake adapter stands in for the provider."""
        from brainscore.model_helpers import api_behavioral
        from brainscore.models.api_closed import model as api_closed
        from brainscore_core.contract import TaskContext
        from brainscore_core.responses import Choice, Text
        from brainscore_core.streaming_helpers import stimulus_session
        from brainscore_core.supported_data_standards.brainio.stimuli import StimulusSet
        provider, calls = _recording_provider('long')

        monkeypatch.setitem(api_behavioral.PROVIDERS, 'openai', provider)
        monkeypatch.setattr(api_closed, '_CACHE_ROOT', str(tmp_path))
        model = api_closed.get_model('gpt-4o-behavioral')
        stimuli = StimulusSet({'stimulus_id': ['a'], 'sentence': ['xyz']})
        stimuli.identifier = 'api-one-identifier'
        for response, channel in ((Text(max_tokens=8), 'behavior:text'),
                                  (Choice(['short', 'long']), 'behavior:choice')):
            session = stimulus_session(stimuli, [channel], TaskContext(response=response))
            model.participate(session)
            answer = session.collect(channel)
            assert list(answer['raw_answer'].values if channel == 'behavior:choice'
                        else answer.values) == ['long']
        assert [call['model'] for call in calls] == ['gpt-4o-2024-08-06'] * 2
        assert calls[0]['settings'] == {'max_tokens': 8}
        # A plain Choice goes to the answer part (point r), which phrases its own prompt.
        assert calls[1]['messages'][0]['content'][-1] == {
            'type': 'text',
            'text': 'xyz\n\nAnswer with exactly one of: short, long.'}

    def test_an_api_reply_model_runs_coggym(self, tmp_path, monkeypatch):
        """API models run CogGym: the settings reach the provider call."""
        from brainscore import BrainScoreModel
        from brainscore.harnesses import coggym
        from brainscore.model_helpers.api_behavioral import build_api_reply
        from brainscore.experiments import Experiment
        from types import SimpleNamespace
        import json
        (tmp_path / 'evaluation').mkdir()
        (tmp_path / 'evaluation/public_manifest.json').write_text(json.dumps(
            {'experiments': [{'path': 'Hu2023Fine/exp1'}]}))
        trials = [SimpleNamespace(id='trial-0', queries=[])]

        def execute(experiment, selected, provider, model, temperature, max_tokens):
            raw = provider.complete_with_metadata('system', [{'role': 'user', 'content': 'q'}],
                                                  model, temperature, max_tokens)
            return [{'trial_id': 'trial-0', 'response': raw['text'], 'scores': {'a': {'scorable': True}}}]
        cli = SimpleNamespace(load_experiment=lambda path: None, _load_selection_map=lambda path: {},
                              _selected_trials=lambda *a: trials,
                              build_messages=lambda trial, exp: ('system', [{'role': 'user', 'content': 'q'}]),
                              _run_experiment=execute, aggregate_experiment=lambda rows: {})
        monkeypatch.setattr(coggym, '_load_reference', lambda checkout: (cli, SimpleNamespace(
            analyze=lambda *a, **k: ([], [], []))))
        monkeypatch.setattr(coggym, '_check_checkout', lambda checkout: None)
        provider, calls = _recording_provider('{"answer": 3}')
        model = BrainScoreModel('api-fixture', in_channels={'text'},
                                reply=build_api_reply(provider, 'm-1'))
        runner = coggym.CogGymRunner(tmp_path, model='m-1')
        Experiment(subject=model, protocol=runner.protocol(reset=lambda repetition: None),
                   output_dir=tmp_path / 'run').run()
        assert calls[0]['system'] == 'system'
        assert calls[0]['settings'] == {'max_tokens': 512, 'temperature': 1.0}
        saved = json.loads((tmp_path / 'run/coggym/run-001.json').read_text())
        assert saved['trials'][0]['response'] == '{"answer": 3}'
