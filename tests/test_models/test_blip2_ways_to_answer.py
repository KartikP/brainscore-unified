"""blip2-opt-2.7b answers each kind of answer with one part: writing, likelihood or readout.

The registry loader builds the registration with stand-in weights, processor, ``generate`` and
logits, and stand-in extractors; the sessions run through ``participate``. Every kind of answer
it gave before keeps its part (the plain choice: the answer part; probabilities and the rating:
the readout); written answers come from the reply and log probabilities from the likelihood.
Each test checks the part that the stand-in network records as the caller and the collected
answer with its response record.
"""
import types

import numpy as np
import pytest
import torch
from PIL import Image

import brainscore
from brainscore import model_registry
from tests.test_harnesses.test_coggym import reference  # noqa: F401  (a fixture)
from brainscore.benchmarks.coggym.test import scored_reference  # noqa: F401  (a fixture)
from brainscore_core.contract import TaskContext
from brainscore_core.responses import Choice, Scale, Text
from brainscore_core.streaming_helpers import stimulus_session
from brainscore_core.supported_data_standards.brainio.assemblies import NeuroidAssembly
from brainscore_core.supported_data_standards.brainio.stimuli import StimulusSet

pytestmark = pytest.mark.unit

IDENTIFIER = 'blip2-opt-2.7b'
LABELS = ['real', 'pseudo']
IMAGE, START = 1, 2                     # the stand-in tokenizer's picture and start tokens
QUERY_TOKENS = 4                        # the picture's query tokens in the stand-in processor
TABLE = torch.linspace(-3.0, 3.0, 256)  # the stand-in logits: a fixed function of the last token


def _log_probabilities(previous: int) -> torch.Tensor:
    """The stand-in model's log probability of each next token after ``previous``."""
    return torch.roll(TABLE, previous).log_softmax(-1)


def expected_score(text: str, option: str) -> float:
    """The summed log probability of the option's tokens in f"{text} {option}" (one token per character)."""
    full = f'{text} {option}'
    return float(sum(_log_probabilities(ord(full[k - 1]))[ord(full[k])] for k in range(len(text), len(full))))


class _Tokenizer:
    """One token per character; the picture and start tokens are special."""
    pad_token_id, eos_token_id = 0, START

    def __call__(self, text, return_tensors=None, add_special_tokens=True):
        ids = ([START] if add_special_tokens else []) + [ord(c) for c in text]
        if return_tensors == 'pt':
            return {'input_ids': torch.tensor([ids]), 'attention_mask': torch.ones(1, len(ids), dtype=torch.long)}
        return {'input_ids': ids, 'attention_mask': [1] * len(ids)}

    def decode(self, ids, skip_special_tokens=False):
        return ''.join(chr(i) for i in ids.tolist() if not (skip_special_tokens and i in (IMAGE, START)))


class _Processor:
    """BLIP-2's processor: the picture's query tokens, then the text's tokens; and pixel values."""

    def __init__(self, calls):
        self.tokenizer, self.calls = _Tokenizer(), calls

    def __call__(self, text=None, images=None, return_tensors='pt', add_special_tokens=True):
        self.calls.append(('processor', text))
        encoded = self.tokenizer(text, return_tensors='pt', add_special_tokens=add_special_tokens)
        ids = torch.cat([torch.full((1, QUERY_TOKENS), IMAGE), encoded['input_ids']], dim=1)
        return {'input_ids': ids, 'attention_mask': torch.ones_like(ids),
                'pixel_values': torch.zeros(1, 3, 2, 2)}


class _Network(torch.nn.Module):
    """The stand-in network: ``generate`` writes the scripted reply, a forward pass gives logits."""

    def __init__(self, calls, name, replies):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(1))
        self.calls, self.name, self.replies = calls, name, replies

    def generate(self, input_ids, attention_mask, **settings):
        self.calls.append(('generate', self.name, tuple(sorted(
            (key, value) for key, value in settings.items() if not torch.is_tensor(value)))))
        reply = torch.tensor([[ord(c) for c in self.replies[0]]])
        return torch.cat([input_ids, reply], dim=1)     # the prompt, then the reply

    def forward(self, input_ids, attention_mask, pixel_values=None):
        self.calls.append(('logits', self.name))
        return types.SimpleNamespace(logits=torch.stack([torch.stack([torch.roll(TABLE, int(token))
                                                                      for token in row]) for row in input_ids]))


class _Blip2(_Network):
    def __init__(self, calls, replies):
        super().__init__(calls, 'blip-2', replies)
        self.vision_model = 'vision model'
        self.language_model = _Network(calls, 'language model', replies)
        self.language_model.model = types.SimpleNamespace(decoder='decoder')


class _Extractor:
    """A wrapper stand-in: features that separate the two labels, and a record of each call."""

    def __init__(self, calls, identifier):
        self.calls, self.identifier = calls, identifier

    def layer_names(self):
        if self.identifier.endswith('-text'):
            return [f'layers.{i}' for i in range(32)]
        return [f'encoder.layers.{i}' for i in range(39)]

    def __call__(self, stimuli, layers, **kwargs):
        ids = list(stimuli['stimulus_id'])
        self.calls.append(('extract', self.identifier, tuple(ids)))
        values = np.array([[2.0, 0.0] if index % 2 == 0 else [0.0, 2.0] for index in range(len(ids))])
        return NeuroidAssembly(values, dims=['presentation', 'neuroid'], coords={
            'stimulus_id': ('presentation', ids), 'neuroid_id': ('neuroid', ['n0', 'n1']),
            'layer': ('neuroid', [layers[0]] * 2)})


@pytest.fixture
def registered(monkeypatch):
    """``build()``: the registration from its registry loader, every heavy part a stand-in.

    It returns the model, the record of calls and ``replies``: ``replies[0]`` is the text the
    stand-in network writes.
    """
    import brainscore.model_helpers._device as device
    import brainscore.models._downloads as downloads
    from brainscore.models.blip2_opt_2_7b import model as blip2

    calls, replies = [], ['real']
    monkeypatch.setattr(downloads, 'hf_preflight', lambda *args, **kwargs: {})
    monkeypatch.setattr(device, 'select_device', lambda: 'cpu')
    monkeypatch.setattr(blip2, 'Blip2ForConditionalGeneration', types.SimpleNamespace(
        from_pretrained=lambda *args, **kwargs: _Blip2(calls, replies)))
    monkeypatch.setattr(blip2, 'AutoProcessor', types.SimpleNamespace(
        from_pretrained=lambda *args, **kwargs: _Processor(calls)))
    monkeypatch.setattr(blip2, 'pin_image_processor', lambda processor, *args, **kwargs: processor)
    monkeypatch.setattr(blip2, 'picture_extractor',
                        lambda model, processor, identifier: _Extractor(calls, identifier))
    monkeypatch.setattr(blip2, 'TextWrapper', lambda **kwargs: _Extractor(calls, kwargs['identifier']))

    def build():
        model = model_registry[IDENTIFIER]()
        calls.clear()
        return model, calls, replies
    return build


def _images(tmp_path, name, n, labels=LABELS):
    rows = []
    for index in range(n):
        path = tmp_path / f'{name}{index}.png'
        Image.new('RGB', (8, 8), color=(index * 30, 0, 0)).save(path)
        rows.append({'stimulus_id': f'{name}{index}', 'image_file_name': str(path),
                     'image_label': labels[index % len(labels)]})
    stimuli = StimulusSet(rows)
    stimuli.identifier = f'blip2-ways-{name}'
    stimuli.stimulus_paths = dict(zip(stimuli['stimulus_id'], stimuli['image_file_name']))
    return stimuli


def _texts(name, texts, **columns):
    stimuli = StimulusSet({'stimulus_id': [f'{name}{i}' for i in range(len(texts))], 'text': texts, **columns})
    stimuli.identifier = f'blip2-ways-{name}'
    return stimuli


def _run(model, stimuli, outputs, task):
    session = stimulus_session(stimuli, outputs, task)
    model.participate(session)
    return {output: session.collect(output) for output in outputs}


def _settings(call):
    return dict(call[2])


def test_the_registration_declares_each_way_to_answer(registered):
    model, _, _ = registered()
    assert {'instruction', 'text', 'vision:image'} <= set(model.in_channels)
    assert {'behavior', 'behavior:choice', 'behavior:probabilities', 'behavior:rt', 'behavior:rating',
            'behavior:text', 'behavior:log_probabilities'} <= set(model.out_channels)


def test_a_plain_choice_is_answered_by_the_answer_part(registered, tmp_path):
    model, calls, _ = registered()
    stimuli = _images(tmp_path, 's', 2)
    task = TaskContext(instruction='Is this a real word or a pseudo word?', response=Choice(LABELS),
                       fitting_stimuli=_images(tmp_path, 'f', 4))
    choice = _run(model, stimuli, ['behavior:choice'], task)['behavior:choice']
    assert choice.attrs['response']['method'] == 'generation'
    # The stand-in writes "real": a valid choice of "real" in every row of the session's table.
    assert list(choice['response_state'].values) == ['valid'] * len(stimuli)
    assert [choice['choice'].values[row] for row in np.argmax(choice.values, axis=1)] == ['real'] * len(stimuli)
    # The answer part is the caller: its own prompt names the options, and it writes at most
    # eight tokens greedily; no extractor runs.
    written = [call for call in calls if call[0] == 'generate']
    assert len(written) == len(stimuli)
    assert all(_settings(call)['max_new_tokens'] == 8 and _settings(call)['do_sample'] is False
               for call in written)
    shown = [call[1] for call in calls if call[0] == 'processor']
    assert all('Choose from: real, pseudo. Answer:' in text for text in shown)
    assert not any(call[0] == 'extract' for call in calls)


def test_probabilities_come_from_the_readout(registered, tmp_path):
    model, calls, _ = registered()
    stimuli = _images(tmp_path, 's', 2)
    task = TaskContext(response=Choice(LABELS), fitting_stimuli=_images(tmp_path, 'f', 4))
    probabilities = _run(model, stimuli, ['behavior:probabilities'], task)['behavior:probabilities']
    assert probabilities.attrs['response']['method'] == 'readout'
    assert probabilities.sizes['presentation'] == len(stimuli)
    np.testing.assert_allclose(probabilities.sum('choice').values, 1.0)
    assert {call[0] for call in calls} == {'extract'}


def test_a_rating_comes_from_the_readout(registered, tmp_path):
    model, calls, _ = registered()
    stimuli = _images(tmp_path, 's', 2)
    task = TaskContext(instruction='How much?', response=Scale(1, 7),
                       fitting_stimuli=_images(tmp_path, 'f', 4, labels=[6.0, 2.0]))
    rating = _run(model, stimuli, ['behavior:rating'], task)['behavior:rating']
    assert rating.attrs['response']['method'] == 'readout'
    assert rating.sizes['presentation'] == len(stimuli)
    assert {call[0] for call in calls} == {'extract'}


@pytest.mark.parametrize('picture', [False, True])
def test_a_written_answer_comes_from_writing_under_the_task_settings(registered, tmp_path, picture):
    model, calls, replies = registered()
    replies[0] = ' A plain written answer. '
    stimuli = (_images(tmp_path, 's', 1) if picture
               else _texts('t', ['The cat sat on the mat.']))
    task = TaskContext(instruction='Describe what you see.',
                       response=Text(max_tokens=5, temperature=0.5, top_p=0.9, seed=3))
    text = _run(model, stimuli, ['behavior:text'], task)['behavior:text']
    assert text.attrs['response']['method'] == 'generation'
    assert list(text.values.ravel()) == ['A plain written answer.']
    written = [call for call in calls if call[0] == 'generate']
    assert written[0][1] == ('blip-2' if picture else 'language model')
    assert {('max_new_tokens', 5), ('do_sample', True), ('temperature', 0.5), ('top_p', 0.9)} <= set(written[0][2])
    if picture:     # BLIP-2's own prompt form: the question, then the answer cue
        shown = [call[1] for call in calls if call[0] == 'processor'][0]
        assert shown.startswith('Question: Describe what you see.') and shown.endswith('Answer:')


@pytest.mark.parametrize('picture', [False, True])
def test_verbatim_log_probabilities_come_from_the_likelihood(registered, tmp_path, picture):
    model, calls, _ = registered()
    texts = ['The bag is full of balls.', 'The bag is full of blocks.']
    options = [('I drew a ball.', 'I drew a block.')] * 2
    if picture:
        stimuli = _images(tmp_path, 's', 2)
        stimuli['text'] = texts
        stimuli['options'] = options
        present = ('image', 'text')
    else:
        stimuli, present = _texts('t', texts, options=options), None
    task = TaskContext(response=Choice(labels='options', verbatim=True), present=present)
    values = _run(model, stimuli, ['behavior:log_probabilities'], task)['behavior:log_probabilities']
    assert values.attrs['response']['method'] == 'likelihood'
    expected = [[expected_score(text, option) for option in pair] for text, pair in zip(texts, options)]
    np.testing.assert_allclose(values.transpose('presentation', 'choice').values, expected, rtol=1e-5)
    assert {call[:2] for call in calls if call[0] in ('generate', 'logits')} == {
        ('logits', 'blip-2' if picture else 'language model')}


def test_two_pictures_are_refused(registered, tmp_path):
    from brainscore_core.compatibility import ExecutionNotFeasibleError
    model, _, _ = registered()
    stimuli = _images(tmp_path, 's', 1)
    stimuli['second'] = stimuli['image_file_name']
    task = TaskContext(response=Text(), present=('image', 'image:second'))
    with pytest.raises(ExecutionNotFeasibleError):
        _run(model, stimuli, ['behavior:text'], task)


@pytest.fixture
def ewok(tmp_path):
    """A prepared EWoK data set of two made-up items (no EWoK sentence)."""
    import pandas as pd
    from brainscore.data import package_data
    rows = [dict(Domain='synthetic-domain', MetaTemplateID='1', TemplateID=str(i), Version='0',
                 Context1=f'fixture context A {i}', Context2=f'fixture context B {i}',
                 Target1=f'fixture target A {i}', Target2=f'fixture target B {i}') for i in range(2)]
    pd.DataFrame(rows).to_csv(tmp_path / 'native.csv', index=False)
    return package_data('ewok', source=tmp_path / 'native.csv', output=tmp_path / 'build')


def test_ewok_log_probabilities_become_applicable_and_run(registered, ewok, tmp_path):
    benchmark = brainscore.load_benchmark('EWoK-core-1.0-logprobs', root=ewok, output_dir=tmp_path / 'run')
    model, calls, _ = registered()
    score = brainscore.score(model, benchmark, check_mem=False)
    assert score.attrs.get('status') != 'N/A' and np.isfinite(float(score))
    assert score.attrs['n_trials'] == 2 * score.attrs['items']      # each item: two targets
    assert {call[0] for call in calls} == {'logits'}                # the likelihood alone


def test_the_ewok_choice_runs_by_writing(registered, ewok, tmp_path):
    benchmark = brainscore.load_benchmark('EWoK-core-1.0-choice', root=ewok, output_dir=tmp_path / 'run')
    model, calls, replies = registered()
    replies[0] = '1'
    score = brainscore.score(model, benchmark, check_mem=False)
    assert float(score) == 0.5 and score.attrs['n_invalid'] == 0      # always "1": one of two right
    # EWoK's prompt as given: no question and no answer cue added.
    assert [call for call in calls if call[0] == 'generate'] and not any(
        call[1].endswith('Answer:') for call in calls if call[0] == 'processor')


def test_a_coggym_experiment_becomes_applicable_and_runs(registered, scored_reference, tmp_path):
    root, _ = scored_reference
    benchmark = brainscore.load_benchmark('CogGym.Hu2023Fine.exp1', folder=root, reset=lambda repetition: None,
                                          repetitions=1, max_tokens=7, output_dir=tmp_path / 'run')
    model, calls, replies = registered()
    replies[0] = 'A written answer.'
    score = brainscore.score(model, benchmark, check_mem=False)
    assert score.attrs.get('status') != 'N/A'
    assert score.attrs['n_valid'] == score.attrs['n_trials'] > 0
    assert all(('max_new_tokens', 7) in call[2] for call in calls if call[0] == 'generate')


def test_a_segments_task_is_answered_by_the_language_model_with_the_shared_scorer(registered, monkeypatch):
    from brainscore.models.blip2_opt_2_7b import model as blip2
    from brainscore_core.responses import Segments
    from brainscore.model_helpers.replies import _SEGMENT_JOIN as SEGMENT_JOIN
    model, _, _ = registered()
    scored = []
    parameters = {'layout': 'segment', 'context': 'prefix', 'aggregation': 'sum', 'units': 'bits',
                  'tokenization': {'join': SEGMENT_JOIN, 'special_tokens': 'tokenizer_default',
                                   'template': 'none', 'truncation': 2048}}

    def scorer(network, tokenizer, segments, sliding_window=False, **unused):
        scored.append((network, tokenizer, segments, sliding_window))
        return [float('nan'), -1.5, -2.5], parameters

    monkeypatch.setattr(blip2, "_segment_log_probabilities", scorer)
    stimuli = _texts('seg', ['the dog ran'], segments=[('the', 'dog', 'ran')])
    answer = _run(model, stimuli, ['behavior:log_probabilities'],
                  TaskContext(response=Segments('segments')))['behavior:log_probabilities']
    (network, tokenizer, segments, sliding_window), = scored
    assert network.name == 'language model' and segments == ('the', 'dog', 'ran')
    assert sliding_window      # OPT's learned positions: the registration states the slide
    assert answer.attrs['response']['parameters']['layout'] == 'segment'
    np.testing.assert_array_equal(answer.values[0, 1:], [-1.5, -2.5])
