"""TaskContext(response=...): choices, ratings and written answers.

Answers come from a reply (parsed by the response), a likelihood part or a
readout. Each answer's response record says which.
"""
import numpy as np
import pytest

from brainscore_core.response_formats import response_record
from brainscore_core.compatibility import ExecutionNotFeasibleError

from brainscore.brainscore_model import BrainScoreModel
from brainscore_core.contract import TaskContext
from brainscore_core.responses import Choice, Scale, Text
from brainscore_core.streaming_helpers import stimulus_session
from brainscore_core.supported_data_standards.brainio.assemblies import NeuroidAssembly
from brainscore_core.supported_data_standards.brainio.stimuli import StimulusSet

pytestmark = pytest.mark.unit

STIMULI = StimulusSet({'stimulus_id': ['a', 'b', 'c', 'd'], 'sentence': ['w', 'x', 'y', 'z'],
                       'label': ['cat', 'dog', 'cat', 'dog']})
STIMULI.identifier = 'responses-toy'


def _features(stimuli, layers, **kwargs):
    ids = list(stimuli['stimulus_id'])
    values = np.array([[1.0, 0.0] if i in 'ac' else [0.0, 1.0] for i in ids])
    return NeuroidAssembly(values, dims=['presentation', 'neuroid'], coords={
        'stimulus_id': ('presentation', ids), 'neuroid_id': ('neuroid', ['f0', 'f1']),
        'layer': ('neuroid', [layers[0]] * 2)})


def _model(replies=None, scores=None, readout=False):
    prompts = []

    def reply(prompt):
        prompts.append(prompt.text())
        return replies[prompt.stimulus_id]

    def likelihood(prompt, candidates):
        return {c: scores[prompt.stimulus_id][c] for c in candidates}

    # A derived format is declared by a statement (rule 4.11): the likelihood's
    # probabilities and, without a reply, its choice; a readout's choice.
    stated = {}
    if scores:
        stated['behavior:probabilities'] = ('likelihood', 'softmax')
        if not replies:
            stated['behavior:choice'] = ('likelihood', 'softmax_argmax')
    elif readout and not replies:
        stated['behavior:choice'] = ('readout', 'argmax')
    model = BrainScoreModel(identifier='toy', extractor={'text': _features},
                            reply=reply if replies else None,
                            likelihood=likelihood if scores else None,
                            behavioral_readout_layer='late' if readout else None,
                            capability_config={'behavior-sessions': stated})
    return model, prompts


def _ask(model, task, channels):
    session = stimulus_session(STIMULI, channels, task=task)
    model.participate(session)
    return session


def test_choice_reply_is_parsed_and_invalid_replies_are_kept_as_invalid():
    model, prompts = _model(replies={'a': 'Cat', 'b': 'a dog.', 'c': 'cat or dog', 'd': 'bird'})
    task = TaskContext(instruction='Which animal?', response=Choice(['cat', 'dog']))
    choice = _ask(model, task, ['behavior:choice']).collect('behavior:choice')
    np.testing.assert_array_equal(choice.values[:2], [[1, 0], [0, 1]])
    assert np.isnan(choice.values[2:]).all()
    assert list(choice['answer_valid'].values) == [True, True, False, False]
    assert list(choice['raw_answer'].values) == ['Cat', 'a dog.', 'cat or dog', 'bird']
    assert choice.attrs['response'] == response_record('behavior:choice', 'generation', parsed_by='model')
    assert prompts[0] == 'Which animal?\n\nw\n\nAnswer with exactly one of: cat, dog.'


def test_rating_from_a_reply_and_from_a_readout():
    model, _ = _model(replies={'a': '6', 'b': 'about 2', 'c': '7.5', 'd': 'none'})
    rating = _ask(model, TaskContext(instruction='How cute?', response=Scale(1, 7, 1)),
                  ['behavior:rating']).collect('behavior:rating')
    assert list(rating.values[:2]) == [6, 2] and np.isnan(rating.values[2:]).all()

    fitting = STIMULI.assign(label=[6.0, 2.0, 6.0, 2.0])
    fitting.identifier = 'responses-ratings'
    readout, _ = _model(readout=True)
    task = TaskContext(response=Scale(1, 7), fitting_stimuli=fitting)
    predicted = _ask(readout, task, ['behavior:rating']).collect('behavior:rating')
    np.testing.assert_allclose(predicted.values, [6, 2, 6, 2], atol=0.5)
    assert predicted.attrs['response'] == response_record('behavior:rating', 'readout')


def test_written_answers_are_kept_as_written():
    model, prompts = _model(replies={s: f'A story about {s}.' for s in 'abcd'})
    text = _ask(model, TaskContext(instruction='Describe it.', response=Text()),
                ['behavior:text']).collect('behavior:text')
    assert list(text.values) == [f'A story about {s}.' for s in 'abcd']
    assert prompts[0] == 'Describe it.\n\nw'


def test_likelihood_gives_a_choice_and_its_implied_probabilities():
    scores = {s: {'cat': -1.0, 'dog': -3.0} if s in 'ac' else {'cat': -2.0, 'dog': -0.5}
              for s in 'abcd'}
    model, _ = _model(scores=scores)
    task = TaskContext(instruction='Which?', response=Choice(['cat', 'dog']))
    session = _ask(model, task, ['behavior:choice', 'behavior:probabilities'])
    probabilities = session.collect('behavior:probabilities')
    np.testing.assert_allclose(probabilities.values[0], [1 / (1 + np.exp(-2)), 1 - 1 / (1 + np.exp(-2))],
                               rtol=1e-6)
    assert probabilities.attrs['response'] == response_record(
        'behavior:probabilities', 'likelihood', conversion='softmax',
        source_format='behavior:log_probabilities')
    np.testing.assert_array_equal(session.collect('behavior:choice').values.argmax(1), [0, 1, 0, 1])


def test_readout_choice_with_probabilities():
    model, _ = _model(readout=True)
    task = TaskContext(response=Choice(['cat', 'dog']), fitting_stimuli=STIMULI)
    session = _ask(model, task, ['behavior:choice', 'behavior:probabilities'])
    np.testing.assert_array_equal(session.collect('behavior:choice').values.argmax(1), [0, 1, 0, 1])
    assert session.collect('behavior:probabilities').values.shape == (4, 2)


@pytest.mark.parametrize('labels', [['cat', 'dog'], ['dog', 'cat']])
def test_readout_probabilities_follow_the_task_label_order(labels):
    model, _ = _model(readout=True)
    task = TaskContext(response=Choice(labels), fitting_stimuli=STIMULI)
    session = _ask(model, task, ['behavior:choice', 'behavior:probabilities'])
    choice, probabilities = session.collect('behavior:choice'), session.collect('behavior:probabilities')
    assert list(probabilities['choice'].values) == list(choice['choice'].values) == labels
    np.testing.assert_array_equal(probabilities.values.argmax(1), choice.values.argmax(1))


@pytest.mark.parametrize('response, channel', [(Text(), 'behavior:text')])
def test_reported_answers_from_a_readout_only_model_are_refused(response, channel):
    model, _ = _model(readout=True)
    assert channel not in model.out_channels
    with pytest.raises(ExecutionNotFeasibleError, match=f"does not produce \\['{channel}'\\]"):
        _ask(model, TaskContext(response=response, fitting_stimuli=STIMULI), [channel])


def test_plain_behavior_request_is_refused_for_a_reply_only_subject():
    """Bare behavior is given only by an answer part or a readout with an extractor."""
    model, prompts = _model(replies={s: 'dog' for s in 'abcd'})
    assert 'behavior' not in model.out_channels
    task = TaskContext(instruction='Which?', response=Choice(['cat', 'dog']))
    with pytest.raises(ExecutionNotFeasibleError, match=r"does not produce \['behavior'\]"):
        _ask(model, task, ['behavior'])
    assert prompts == []


def test_plain_behavior_request_returns_the_primary_answer():
    """With an answer part bare behavior is declared; a response task's choice is copied to it.
    A plain Choice is answered by answer= when it is wired (point r)."""
    model = BrainScoreModel(identifier='toy', extractor={'text': _features},
                            answer=lambda stimulus_row, instruction, label_set: 'cat',
                            reply=lambda prompt: 'dog')
    task = TaskContext(instruction='Which?', response=Choice(['cat', 'dog']))
    behavior = _ask(model, task, ['behavior']).collect('behavior')
    np.testing.assert_array_equal(behavior.values.argmax(1), [0, 0, 0, 0])


def test_task_type_and_response_must_agree():
    with pytest.raises(ValueError, match='contradicts'):
        TaskContext(task_type='label', response=Scale(1, 7))
    with pytest.raises(ValueError, match='passive'):
        TaskContext()
    assert TaskContext(response=Choice(['a', 'b'])).label_set == ['a', 'b']


def test_a_trial_is_shown_in_the_declared_order():
    from brainscore_core.responses import trial_prompt
    row = {'stimulus_id': 's', 'sentence': 'Watch the ball.', 'video_path': 'clip.mp4'}
    task = TaskContext(instruction='Which way?', response=Choice(['left', 'right']),
                       present=('text', 'video'))
    assert trial_prompt(task, row).parts == (
        ('text', 'Which way?'), ('text', 'Watch the ball.'), ('video', 'clip.mp4'),
        ('text', 'Answer with exactly one of: left, right.'))
    with pytest.raises(ValueError, match='present'):
        trial_prompt(TaskContext(instruction='Which way?', response=Choice(['left', 'right'])), row)


def test_an_unordered_multi_part_stimulus_fails_before_any_reply():
    calls = []
    model = BrainScoreModel(identifier='m', in_channels={'text', 'vision:video'},
                            reply=lambda prompt: calls.append(prompt) or 'left')
    both = StimulusSet({'stimulus_id': ['a'], 'sentence': ['x'], 'video_path': ['v.mp4']})
    both.identifier = 'both-parts'
    session = stimulus_session(both, ['behavior:choice'], task=TaskContext(
        instruction='Which way?', response=Choice(['left', 'right'])))
    with pytest.raises(ValueError, match='present'):
        model.participate(session)
    assert calls == []


def test_named_parts_show_a_sequence_in_order():
    from brainscore_core.responses import trial_prompt
    row = {'stimulus_id': 't1', 'instruction': 'rules.png', 'word_1': 'w1.png', 'word_2': 'w2.png'}
    task = TaskContext(response=Choice(['yes', 'no']),
                       present=('image:instruction', 'image:word_1', 'image:word_2'))
    prompt = trial_prompt(task, row)
    assert prompt.parts[:3] == (('image', 'rules.png'), ('image', 'w1.png'), ('image', 'w2.png'))
    assert prompt.names == ('instruction', 'word_1', 'word_2', 'directive')
    with pytest.raises(ValueError, match='word_3'):
        trial_prompt(TaskContext(response=Choice(['yes', 'no']), present=('image:word_3',)), row)
    with pytest.raises(ValueError, match='kind'):
        TaskContext(response=Choice(['yes', 'no']), present=('picture:word_1',))


@pytest.mark.parametrize('link', ['evidence_margin', 'surprisal'])
def test_implied_reaction_times_rise_for_close_calls(link):
    scores = {'a': {'cat': -0.1, 'dog': -5.0}, 'b': {'cat': -1.0, 'dog': -1.2},
              'c': {'cat': -3.0, 'dog': -0.05}, 'd': {'cat': -0.7, 'dog': -0.69}}
    model, _ = _model(scores=scores)
    task = TaskContext(instruction='Which?', response=Choice(['cat', 'dog'], reaction_time=link))
    rt = _ask(model, task, ['behavior:rt']).collect('behavior:rt')
    assert rt.attrs['response'] == response_record(
        'behavior:rt', 'link', source_format='behavior:probabilities',
        parameters={'link': link, 'source_method': 'likelihood', 'units': 'monotone'})
    fast, slow = rt.values[0], rt.values[3]                   # a clear call, a near tie
    assert slow > fast


def test_a_reply_only_model_cannot_give_reaction_times():
    model, _ = _model(replies={s: 'cat' for s in 'abcd'})
    assert 'behavior:rt' not in model.out_channels
