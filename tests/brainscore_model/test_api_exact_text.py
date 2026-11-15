"""The API clients return the provider's text exactly, and the cache never gives back a
text that the old code stored stripped (S8-01-R1).

Each test runs the real client function of a provider (``PROVIDERS``) with the SDK
transport replaced by a local object; every session goes through ``participate``.
"""
import hashlib
import json
import sys
import types

import numpy as np
import pytest

from brainscore import BrainScoreModel
from brainscore_core.contract import TaskContext
from brainscore_core.responses import Choice, Scale, Text
from brainscore_core.streaming_helpers import stimulus_session
from brainscore_core.supported_data_standards.brainio.stimuli import StimulusSet

pytestmark = pytest.mark.unit

PROVIDERS = ('anthropic', 'openai', 'deepseek', 'openrouter')
CHOICE = TaskContext(instruction='Is this a real word or a pseudo word?', response=Choice(['real', 'pseudo']))
VALID, INVALID = '  REAL.  ', '  Pseudo, not REAL  '


@pytest.fixture
def transport(monkeypatch):
    """Both SDKs replaced by local objects that give back ``transport.text``; ``calls`` counts requests."""
    state = types.SimpleNamespace(text=VALID, calls=0)

    class Messages:
        def create(self, **kwargs):
            state.calls += 1
            return types.SimpleNamespace(content=[types.SimpleNamespace(type='text', text=state.text)],
                                         usage=None)

    class Completions:
        def create(self, **kwargs):
            state.calls += 1
            message = types.SimpleNamespace(content=state.text)
            return types.SimpleNamespace(choices=[types.SimpleNamespace(message=message)], usage=None)

    monkeypatch.setitem(sys.modules, 'anthropic', types.SimpleNamespace(
        Anthropic=lambda **kwargs: types.SimpleNamespace(messages=Messages())))
    monkeypatch.setitem(sys.modules, 'openai', types.SimpleNamespace(
        OpenAI=lambda **kwargs: types.SimpleNamespace(chat=types.SimpleNamespace(completions=Completions()))))
    return state


def _words():
    table = StimulusSet({'stimulus_id': ['w1'], 'sentence': ['florp']})
    table.identifier = 'api-exact-text'
    return table


def _answer_model(provider, **kwargs):
    from brainscore.model_helpers.api_behavioral import api_answer
    return BrainScoreModel(f'api-{provider}', in_channels={'text'},
                           answer=api_answer(provider, 'm', **kwargs))


def _choice(model):
    session = stimulus_session(_words(), ['behavior:choice'], CHOICE)
    model.participate(session)
    return session.collect('behavior:choice')


@pytest.mark.parametrize('provider', PROVIDERS)
def test_a_valid_reply_is_stored_exactly_and_parsed(transport, provider):
    choice = _choice(_answer_model(provider))
    assert list(choice['raw_answer'].values) == [VALID]
    assert list(choice['response_state'].values) == ['valid']
    np.testing.assert_array_equal(choice.values, [[1.0, 0.0]])       # Choice.parse ignores outer spaces


@pytest.mark.parametrize('provider', PROVIDERS)
def test_an_invalid_reply_is_stored_exactly(transport, provider):
    transport.text = INVALID
    choice = _choice(_answer_model(provider))
    assert list(choice['raw_answer'].values) == [INVALID]
    assert list(choice['response_state'].values) == ['invalid']
    assert list(choice['parse_error'].values) == ["several labels in the reply: ['real', 'pseudo']"]


@pytest.mark.parametrize('provider', PROVIDERS)
def test_the_answer_cache_gives_the_exact_text_and_no_old_stripped_entry(transport, provider, tmp_path):
    from brainscore.model_helpers.api_behavioral import _answer_text
    payload = hashlib.sha256(b'florp').hexdigest()[:16]
    # An entry of the old form: the key without the form part, the text stripped.
    old_key = hashlib.sha256(json.dumps([provider, 'm', _answer_text(CHOICE.instruction, ['real', 'pseudo']),
                                         ['real', 'pseudo'], 'w1', payload, None, 16])
                             .encode('utf-8')).hexdigest()
    (tmp_path / f'{old_key}.json').write_text(json.dumps({'response': 'pseudo'}))
    model = _answer_model(provider, cache_dir=str(tmp_path))
    first = _choice(model)
    assert transport.calls == 1                                       # the old entry is not read
    transport.text = 'something else'
    second = _choice(model)
    assert transport.calls == 1                                       # served by the cache
    assert list(first['raw_answer'].values) == list(second['raw_answer'].values) == [VALID]


@pytest.mark.parametrize('provider', PROVIDERS)
@pytest.mark.parametrize('response, channel, expected', [
    (Text(max_tokens=8), 'behavior:text', VALID),
    (Scale(1, 7, 1), 'behavior:rating', 4.0),
])
def test_the_reply_part_stores_the_exact_text_and_reads_no_old_entry(transport, provider, tmp_path,
                                                                   response, channel, expected):
    """Text and rating tasks: the reply cache key also holds the form of the stored text."""
    from brainscore.model_helpers.api_behavioral import _reply_messages, build_api_reply
    from brainscore_core.responses import trial_prompt
    transport.text = '  4  ' if channel == 'behavior:rating' else VALID
    task = TaskContext(instruction='Answer.', response=response)
    system, messages = _reply_messages(trial_prompt(task, _words().iloc[0]))
    settings = {'max_tokens': 8 if isinstance(response, Text) else 512}   # the reply's settings
    old_key = hashlib.sha256(json.dumps([provider, 'm', system, messages, settings], sort_keys=True)
                             .encode('utf-8')).hexdigest()
    (tmp_path / f'{old_key}.json').write_text(json.dumps({'response': 'stale'}))
    model = BrainScoreModel(f'api-reply-{provider}', in_channels={'text'},
                            reply=build_api_reply(provider, 'm', cache_dir=tmp_path))
    session = stimulus_session(_words(), [channel], task)
    model.participate(session)
    answer = session.collect(channel)
    assert transport.calls == 1
    assert list(answer['raw_answer'].values) == [transport.text]
    assert answer.values.tolist() == [expected]
