"""Contract test: which part answers a behavioral task, what it receives, and what comes back.

One chooser picks the part (generation, likelihood or readout) and one entry
answers. These tests pin, through ``participate``, the exact arguments each part
receives and the exact packaging of its answer, so a change to the machinery
cannot change an answer or a score without failing here.
"""
import sys
import types

import numpy as np
import pytest

from brainscore import BrainScoreModel
from brainscore_core.contract import TaskContext
from brainscore_core.responses import Choice, Text
from brainscore_core.streaming import StreamEvent
from brainscore_core.streaming_helpers import stimulus_session
from brainscore_core.supported_data_standards.brainio.assemblies import NeuroidAssembly
from brainscore_core.supported_data_standards.brainio.stimuli import StimulusSet

pytestmark = pytest.mark.unit

STIMULI = StimulusSet({'stimulus_id': ['a', 'b', 'c'], 'sentence': ['real x', 'fake y', 'real z'],
                       'label': ['cat', 'dog', 'cat']})
STIMULI.identifier = 'behavior-paths'
FITTING = StimulusSet({'stimulus_id': ['f0', 'f1', 'f2', 'f3'], 'sentence': ['real 0', 'fake 1', 'real 2', 'fake 3'],
                       'label': ['cat', 'dog', 'cat', 'dog']})
FITTING.identifier = 'behavior-paths-fit'


def _features(stimuli, layers, **kwargs):
    ids = list(stimuli['stimulus_id'])
    values = np.array([[2.0, 0.0] if 'real' in text else [0.0, 2.0] for text in stimuli['sentence']])
    return NeuroidAssembly(values, dims=['presentation', 'neuroid'], coords={
        'stimulus_id': ('presentation', ids), 'sentence': ('presentation', list(stimuli['sentence'])),
        'neuroid_id': ('neuroid', ['n0', 'n1']), 'layer': ('neuroid', [layers[0]] * 2)})


def _model(calls, **parts):
    def answer(stimulus_row, instruction, label_set):
        calls.append(('answer', stimulus_row['stimulus_id'], instruction, list(label_set)))
        return {'a': 'cat', 'b': 'dog', 'c': 'neither'}[stimulus_row['stimulus_id']]

    def reply(prompt):
        calls.append(('reply', prompt.stimulus_id, prompt.text()))
        return 'Cat' if prompt.stimulus_id != 'b' else 'dog'

    def likelihood(prompt, candidates):
        calls.append(('likelihood', prompt.stimulus_id, list(candidates)))
        return {'cat': -1.0, 'dog': -2.0} if prompt.stimulus_id != 'b' else {'cat': -3.0, 'dog': -0.5}
    chosen = {'answer': answer, 'reply': reply, 'likelihood': likelihood}
    return BrainScoreModel('paths', extractor={'text': _features}, behavioral_readout_layer='late',
                           **{name: chosen[name] for name in parts})


def _run(model, outputs, task):
    session = stimulus_session(STIMULI, outputs, task)
    model.participate(session)
    return session


LEGACY = dict(task_type='probabilities', instruction='Which animal?', label_set=['cat', 'dog'])


def test_a_legacy_task_with_an_answer_part_calls_it_once_per_stimulus_and_keeps_its_packaging():
    calls = []
    model = _model(calls, answer=True, reply=True, likelihood=True)
    with pytest.warns(UserWarning, match="answer function returned 'neither'"):
        session = _run(model, ['behavior'], TaskContext(**LEGACY))
    assert calls == [('answer', sid, 'Which animal?', ['cat', 'dog']) for sid in 'abc']
    assert model._behavioral.response_path is None   # the session gave the task back
    (event,) = session.emitted
    assert (event.channel, event.t_ms, dict(event.meta)) == (
        'behavior', 0.0, {'driver': 'interact', 'task_type': 'probabilities'})
    answer = event.payload
    np.testing.assert_array_equal(answer.values[:2], [[1, 0], [0, 1]])
    assert np.isnan(answer.values[2]).all() and answer.dtype == np.float32
    assert list(answer.indexes['presentation'].names) == [
        'stimulus_id', 'sentence', 'label', 'raw_answer', 'answer_valid']
    assert list(answer['raw_answer'].values) == ['cat', 'dog', 'neither']
    assert answer.attrs['measure'] == {'kind': 'reported', 'source': 'answer'}


def test_a_legacy_task_prefers_the_answer_and_warns_when_a_readout_could_fit():
    calls = []
    model = _model(calls, answer=True)
    with pytest.warns(UserWarning, match='Both generation and fitted readout'):
        session = _run(model, ['behavior'], TaskContext(**LEGACY, fitting_stimuli=FITTING))
    assert [call[0] for call in calls] == ['answer'] * 3
    assert session.collect('behavior').attrs['measure'] == {'kind': 'reported', 'source': 'answer'}


def test_a_legacy_task_on_a_model_built_without_an_answer_part_is_read_out():
    calls = []
    model = _model(calls)
    answer = _run(model, ['behavior'], TaskContext(**LEGACY, fitting_stimuli=FITTING)).collect('behavior')
    assert calls == []
    assert answer.attrs['measure'] == {'kind': 'implied', 'source': 'readout'}
    assert answer.values.argmax(1).tolist() == [0, 1, 0]


def test_a_legacy_task_streamed_gives_one_event_per_trial():
    calls = []
    model = _model(calls, answer=True)
    session = stimulus_session(STIMULI, ['behavior'], TaskContext(**LEGACY), streaming=True)
    with pytest.warns(UserWarning):
        model.participate(session)
    assert [dict(e.meta) for e in session.emitted] == [
        {'driver': 'interact_streaming', 'task_type': 'probabilities', 'trial_index': i, 'streaming': True}
        for i in range(3)]


def test_a_legacy_task_with_a_region_records_the_answer_pass():
    calls = []
    model = _model(calls, answer=True)
    model.region_layer_map = {'IT': 'late'}
    from brainscore_core.compatibility import CompatibilityError
    with pytest.raises(CompatibilityError, match='record'):
        # The plain extractor has no record(layers) and no model= to hook: N/A.
        _run(model, ['behavior', 'neural:IT'], TaskContext(**LEGACY))
    assert calls == []                               # before any answer


def test_a_choice_goes_to_the_reply_part_with_the_full_prompt():
    calls = []
    model = _model(calls, answer=True, reply=True, likelihood=True)
    choice = _run(model, ['behavior:choice'],
                  TaskContext(instruction='Which animal?', response=Choice(['cat', 'dog']))
                  ).collect('behavior:choice')
    assert calls[0] == ('reply', 'a', 'Which animal?\n\nreal x\n\nAnswer with exactly one of: cat, dog.')
    assert [c[0] for c in calls] == ['reply'] * 3
    assert choice.attrs['measure'] == {'kind': 'reported', 'source': 'reply'}


def test_a_plain_choice_without_a_reply_part_goes_to_the_answer_part():
    calls = []
    model = _model(calls, answer=True)
    choice = _run(model, ['behavior:choice'],
                  TaskContext(instruction='Which animal?', response=Choice(['cat', 'dog']))
                  ).collect('behavior:choice')
    assert calls == [('answer', sid, 'Which animal?', ['cat', 'dog']) for sid in 'abc']
    assert choice.attrs['measure'] == {'kind': 'reported', 'source': 'answer'}
    assert list(choice['answer_valid'].values) == [True, True, False]


def test_probabilities_next_to_a_reply_come_from_the_likelihood_part_first():
    calls = []
    model = _model(calls, reply=True, likelihood=True)
    session = _run(model, ['behavior:choice', 'behavior:probabilities'],
                   TaskContext(instruction='Which?', response=Choice(['cat', 'dog'], probabilities=True)))
    assert [c[0] for c in calls] == ['likelihood', 'reply'] * 3
    assert session.collect('behavior:probabilities').attrs['measure'] == {
        'kind': 'implied', 'source': 'likelihood'}


def test_text_settings_reach_the_reply_part_and_extra_data_reaches_the_event():
    seen = []

    def reply(prompt):
        seen.append(prompt.response)
        return StreamEvent('behavior:text', 'written', None, {'reasoning': 'r'})
    model = BrainScoreModel('text', in_channels={'text'}, reply=reply)
    session = _run(model, ['behavior:text'], TaskContext(instruction='Say.', response=Text(max_tokens=9, seed=1)))
    assert seen == [Text(max_tokens=9, seed=1)] * 3
    (event,) = session.emitted
    assert event.meta['reasoning'] == ['r', 'r', 'r']


def test_the_one_chooser_names_every_path_it_tried():
    model = BrainScoreModel('none', in_channels={'text'})
    with pytest.raises(ValueError, match='Paths tried: generation needs answer=.*readout needs'):
        model.start_task(TaskContext(**LEGACY))
    with pytest.raises(ValueError, match='cannot give a Choice answer. Paths tried'):
        model.start_task(TaskContext(response=Choice(['cat', 'dog'])))


@pytest.mark.parametrize('provider, image_first, settings', [
    ('anthropic', True, {'max_tokens': 16}),
    ('openai', False, {'max_tokens': 16, 'temperature': 0}),
])
def test_an_api_answer_part_sends_the_request_it_always_sent(provider, image_first, settings,
                                                              monkeypatch, tmp_path):
    """One client per provider; the answer part's request is unchanged."""
    from brainscore.model_helpers.api_behavioral import build_api_generation_fn
    requests = []

    class Messages:
        def create(self, **kwargs):
            requests.append(kwargs)
            return types.SimpleNamespace(content=[types.SimpleNamespace(type='text', text='real')],
                                         usage=types.SimpleNamespace(input_tokens=1, output_tokens=1))

    class Completions:
        def create(self, **kwargs):
            requests.append(kwargs)
            return types.SimpleNamespace(
                choices=[types.SimpleNamespace(message=types.SimpleNamespace(content='real'))],
                usage=types.SimpleNamespace(prompt_tokens=1, completion_tokens=1))
    monkeypatch.setitem(sys.modules, 'anthropic', types.SimpleNamespace(
        Anthropic=lambda: types.SimpleNamespace(messages=Messages())))
    monkeypatch.setitem(sys.modules, 'openai', types.SimpleNamespace(
        OpenAI=lambda **kwargs: types.SimpleNamespace(chat=types.SimpleNamespace(completions=Completions()))))
    image = tmp_path / 'w.png'
    image.write_bytes(b'png')
    answer = build_api_generation_fn(provider, 'm', max_tokens=16)
    assert answer({'stimulus_id': 's', 'image_file_name': str(image)}, 'Real?', ['real', 'pseudo']) == 'real'
    (request,) = requests
    content = request['messages'][-1]['content']
    kinds = [part['type'] for part in content]
    assert kinds == (['image', 'text'] if image_first else ['text', 'image_url'])
    text = next(part['text'] for part in content if part['type'] == 'text')
    assert text == 'Real?\n\nAnswer with exactly one of: real, pseudo. Reply with only that word, nothing else.'
    assert {k: v for k, v in request.items() if k in ('max_tokens', 'temperature')} == settings
