"""Each Qwen registration answers behavior:probabilities with the part it states (rule 4.11).

The registry loaders build the registrations with stand-in weights, extractors, answer, reply
and likelihood parts; the sessions run through ``participate``. qwen2.5-vl-3b, -7b, -3b-bf16 and
qwen3.6-27b state the readout (fitted on the task's fitting stimuli); the variants with the
word-form region (-vwfa) state the likelihood with softmax. The default registration also keeps
its text route (generation) and its log-probability route (likelihood).
"""
import sys
import types

import numpy as np
import pytest
from PIL import Image

from brainscore import model_registry
from brainscore_core.compatibility import ExecutionNotFeasibleError
from brainscore_core.contract import TaskContext
from brainscore_core.responses import Choice, Text
from brainscore_core.streaming_helpers import stimulus_session
from brainscore_core.supported_data_standards.brainio.assemblies import NeuroidAssembly
from brainscore_core.supported_data_standards.brainio.stimuli import StimulusSet

pytestmark = pytest.mark.unit

READOUT = ['qwen2.5-vl-3b', 'qwen2.5-vl-7b', 'qwen2.5-vl-3b-bf16', 'qwen3.6-27b']
LIKELIHOOD = ['qwen2.5-vl-3b-vwfa', 'qwen2.5-vl-7b-vwfa']
LABELS = ['real', 'pseudo']
SCORES = {'real': -0.5, 'pseudo': -1.5}       # the stand-in likelihood of each option


class _Extractor:
    """A wrapper stand-in: features that separate the labels, and a record of each call."""

    def __init__(self, calls, identifier):
        self.calls, self.identifier = calls, identifier

    def layer_names(self):
        """The layers of the tower it wraps: the decoder for text, else the picture tower."""
        if self.identifier.endswith('-text'):
            return [f'layers.{i}' for i in range(64)]
        return [f'blocks.{i}' for i in range(32)]

    def __call__(self, stimuli, layers, **kwargs):
        ids = list(stimuli['stimulus_id'])
        self.calls.append(('extract', self.identifier, tuple(ids)))
        values = np.array([[2.0, 0.0] if index % 2 == 0 else [0.0, 2.0] for index in range(len(ids))])
        return NeuroidAssembly(values, dims=['presentation', 'neuroid'], coords={
            'stimulus_id': ('presentation', ids), 'neuroid_id': ('neuroid', ['n0', 'n1']),
            'layer': ('neuroid', [layers[0]] * 2)})


def _parts(calls):
    def answer(stimulus_row, instruction, label_set):
        calls.append(('answer', stimulus_row['stimulus_id']))
        return label_set[0]

    def reply(prompt):
        calls.append(('reply', prompt.stimulus_id))
        return LABELS[0]

    def likelihood(prompt, candidates):
        calls.append(('likelihood', prompt.stimulus_id))
        return {candidate: SCORES[candidate] for candidate in candidates}
    return answer, reply, likelihood


@pytest.fixture
def registered(monkeypatch):
    """Build a registration through its registry loader, with every heavy part a stand-in."""
    import brainscore.model_helpers.intervention as intervention
    import brainscore.model_helpers.replies as replies
    import brainscore.model_helpers.text_wrapper as text_wrapper
    import brainscore.model_helpers.vlm_video_wrapper as vlm_video_wrapper
    import brainscore.model_helpers.vlm_vision_wrapper as vlm_vision_wrapper
    import brainscore.models._downloads as downloads
    import brainscore_core.hf_compat as hf_compat
    from brainscore.models.qwen25_vl_3b import model as qwen25
    from brainscore.models.qwen36_27b import model as qwen36
    from tests.test_models.test_qwen36_registration import FakeProcessor, FakeVLM

    calls = []
    answer, reply, likelihood = _parts(calls)
    monkeypatch.setattr(replies, 'chat_parts', lambda *args, **kwargs: (reply, likelihood))
    monkeypatch.setattr(text_wrapper, 'TextWrapper', lambda **kwargs: _Extractor(calls, kwargs['identifier']))
    monkeypatch.setattr(vlm_vision_wrapper, 'VLMVisionWrapper',
                        lambda **kwargs: _Extractor(calls, kwargs['identifier']))
    monkeypatch.setattr(vlm_video_wrapper, 'VLMVideoWrapper',
                        lambda model, processor, identifier: _Extractor(calls, identifier))
    monkeypatch.setattr(downloads, 'hf_preflight', lambda *args, **kwargs: {})
    monkeypatch.setattr(hf_compat, 'pin_image_processor', lambda processor, *args, **kwargs: processor)
    # Qwen2.5-VL: the checkpoint loader, the picture tower and the answer part.
    stand_in = types.SimpleNamespace(model=types.SimpleNamespace(visual='visual', language_model='decoder'))
    processor = types.SimpleNamespace(tokenizer='tokenizer')
    monkeypatch.setattr(qwen25, 'load_qwen', lambda identifier, size: (stand_in, processor, {}))
    monkeypatch.setattr(qwen25, 'picture_extractor',
                        lambda model, processor, identifier: _Extractor(calls, f'{identifier}-vision'))
    monkeypatch.setattr(qwen25, '_make_answer', lambda *args, **kwargs: answer)
    monkeypatch.setattr(qwen25, '_bf16_provenance', lambda *args: {'checkpoint': 'stand-in'})
    monkeypatch.setattr(intervention, 'pytorch_intervention', lambda module: (lambda change: None))
    # Qwen3.6: the checkpoint classes and the answer part.
    monkeypatch.setitem(sys.modules, 'transformers', types.SimpleNamespace(
        Qwen3_5ForConditionalGeneration=FakeVLM,
        AutoProcessor=types.SimpleNamespace(from_pretrained=lambda *a, **k: FakeProcessor())))
    monkeypatch.setattr(qwen36, '_make_answer', lambda *args, **kwargs: answer)

    def build(identifier):
        model = model_registry[identifier]()
        calls.clear()
        return model, calls
    return build


def _images(tmp_path, name, n):
    rows = []
    for index in range(n):
        path = tmp_path / f'{name}{index}.png'
        Image.new('RGB', (8, 8), color=(index * 30, 0, 0)).save(path)
        rows.append({'stimulus_id': f'{name}{index}', 'image_file_name': str(path),
                     'image_label': LABELS[index % 2]})
    stimuli = StimulusSet(rows)
    stimuli.identifier = f'qwen-statements-{name}'
    stimuli.stimulus_paths = dict(zip(stimuli['stimulus_id'], stimuli['image_file_name']))
    return stimuli


def _probabilities(model, stimuli, fitting):
    task = TaskContext(response=Choice(LABELS), fitting_stimuli=fitting)
    session = stimulus_session(stimuli, ['behavior:probabilities'], task)
    model.participate(session)
    return session.collect('behavior:probabilities')


@pytest.mark.parametrize('identifier', READOUT)
def test_a_readout_registration_answers_probabilities_from_its_fitted_readout(registered, identifier, tmp_path):
    model, calls = registered(identifier)
    answer = _probabilities(model, _images(tmp_path, 's', 2), _images(tmp_path, 'f', 4))
    assert answer.attrs['response']['method'] == 'readout'
    assert {call[0] for call in calls} == {'extract'}             # no answer, reply or likelihood
    assert ('extract', calls[0][1], ('f0', 'f1', 'f2', 'f3')) in calls   # the fit


@pytest.mark.parametrize('identifier', READOUT)
def test_a_readout_registration_refuses_probabilities_without_fitting_stimuli_before_compute(
        registered, identifier, tmp_path):
    model, calls = registered(identifier)
    with pytest.raises(ExecutionNotFeasibleError):
        _probabilities(model, _images(tmp_path, 's', 2), None)
    assert calls == []


@pytest.mark.parametrize('identifier', LIKELIHOOD)
@pytest.mark.parametrize('fitted', [False, True])
def test_a_vwfa_registration_answers_probabilities_from_its_likelihood(registered, identifier, fitted, tmp_path):
    model, calls = registered(identifier)
    answer = _probabilities(model, _images(tmp_path, 's', 2), _images(tmp_path, 'f', 4) if fitted else None)
    record = answer.attrs['response']
    assert (record['method'], record['conversion']) == ('likelihood', 'softmax')
    assert calls == [('likelihood', 's0'), ('likelihood', 's1')]
    implied = np.exp([SCORES[label] for label in LABELS])
    np.testing.assert_allclose(answer.values, [implied / implied.sum()] * 2, rtol=1e-6)


def _sentences(options=False):
    rows = [{'stimulus_id': 't0', 'sentence': 'a word'}, {'stimulus_id': 't1', 'sentence': 'another'}]
    if options:
        for row in rows:
            row['options'] = LABELS
    stimuli = StimulusSet(rows)
    stimuli.identifier = f'qwen-statements-sentences-{options}'
    return stimuli


def test_the_default_registration_writes_text_with_its_reply(registered):
    """qwen2.5-vl-3b keeps its text route: free text comes from the generation part."""
    model, calls = registered('qwen2.5-vl-3b')
    session = stimulus_session(_sentences(), ['behavior:text'], TaskContext(instruction='Say it.', response=Text()))
    model.participate(session)
    assert session.collect('behavior:text').attrs['response']['method'] == 'generation'
    assert ('reply', 't0') in calls


def test_the_default_registration_gives_log_probabilities_with_its_likelihood(registered):
    """qwen2.5-vl-3b keeps its log-probability route: the likelihood part scores each option."""
    model, calls = registered('qwen2.5-vl-3b')
    session = stimulus_session(_sentences(options=True), ['behavior:log_probabilities'],
                               TaskContext(response=Choice('options', verbatim=True)))
    model.participate(session)
    answer = session.collect('behavior:log_probabilities')
    assert answer.attrs['response']['method'] == 'likelihood'
    assert ('likelihood', 't0') in calls
