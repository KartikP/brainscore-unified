"""Wiring of the Qwen3.6-27B vision-language registration, without weights.

Transformers, the processor and the wrappers are stand-ins; the towers are tiny
real torch modules so interventions can be observed. This checks wiring only,
not checkpoint behavior (that is a GPU run).
"""
import sys
import types

import pytest
import torch
from PIL import Image

from brainscore_core import Perturbation, Selection, StateChange, TaskContext
from brainscore_core.streaming_helpers import stimulus_session
from brainscore_core.supported_data_standards.brainio.stimuli import StimulusSet
from brainscore_core.streaming_helpers import apply_state_change

import brainscore.models._downloads as downloads
import brainscore.model_helpers.text_wrapper as text_wrapper
import brainscore.model_helpers.vision_wrapper as vision_wrapper
import brainscore_core.hf_compat as hf_compat
from brainscore.models.qwen36_27b import model as registration

pytestmark = pytest.mark.unit


class FakeWrapper:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.identifier = kwargs['identifier']

    def __call__(self, stimuli, layers, stimuli_identifier=None, **kwargs):
        raise AssertionError('not called in wiring tests')


class FakeVLM(torch.nn.Module):
    """Qwen3_5ForConditionalGeneration shape: model.visual.blocks, model.language_model.layers."""

    def __init__(self):
        super().__init__()
        visual = torch.nn.Module()
        visual.blocks = torch.nn.ModuleList([torch.nn.Linear(2, 2) for _ in range(27)])
        language_model = torch.nn.Module()
        language_model.layers = torch.nn.ModuleList([torch.nn.Linear(2, 2) for _ in range(64)])
        self.model = torch.nn.Module()
        self.model.visual, self.model.language_model = visual, language_model
        self.generate_kwargs = None
        self.device = torch.device('cpu')

    @classmethod
    def from_pretrained(cls, name, **kwargs):
        assert name == 'Qwen/Qwen3.6-27B'
        return cls()

    def generate(self, **kwargs):
        self.generate_kwargs = kwargs
        return torch.tensor([[1, 2, 3, 9]])      # prompt is 3 tokens, reply is 1


class FakeInputs(dict):
    def to(self, device):
        return self


class FakeProcessor:
    def __init__(self):
        self.tokenizer = types.SimpleNamespace(
            padding_side='right', pad_token=None, eos_token='<eos>',
            decode=lambda ids, skip_special_tokens: ' Pseudo.')
        self.template_kwargs = self.images = None

    def apply_chat_template(self, messages, **kwargs):
        self.template_kwargs = kwargs
        return 'prompt'

    def __call__(self, text, images, return_tensors, padding):
        self.images = images
        return FakeInputs(input_ids=torch.tensor([[1, 2, 3]]))


@pytest.fixture
def built(monkeypatch):
    processor = FakeProcessor()
    monkeypatch.setitem(sys.modules, 'transformers', types.SimpleNamespace(
        Qwen3_5ForConditionalGeneration=FakeVLM,
        AutoProcessor=types.SimpleNamespace(from_pretrained=lambda *a, **k: processor)))
    monkeypatch.setattr(hf_compat, 'pin_image_processor', lambda p, *a, **k: p)
    monkeypatch.setattr(downloads, 'hf_preflight', lambda *a, **k: {})
    monkeypatch.setattr(vision_wrapper, 'VisionWrapper', FakeWrapper)
    monkeypatch.setattr(text_wrapper, 'TextWrapper', FakeWrapper)
    import brainscore.model_helpers.vlm_video_wrapper as vlm_video_wrapper
    monkeypatch.setattr(vlm_video_wrapper, 'VLMVideoWrapper',
                        lambda model, processor, identifier: FakeWrapper(model=model, identifier=identifier))
    return registration.get_model('qwen3.6-27b'), processor


def test_both_towers_are_wired(built):
    model, _ = built
    vision, text = model.extractor['vision'], model.extractor['text']
    assert vision.kwargs['kind'] == 'vlm'
    assert vision.kwargs['forward_kwargs_map'] == {'grid_thw': 'image_grid_thw'}
    assert isinstance(vision.kwargs['model'].blocks, torch.nn.ModuleList)
    assert len(text.kwargs['model'].layers) == 64
    # Same identifier and cache key as the earlier text-only registration.
    assert (text.identifier, text.kwargs['backbone_id']) == ('qwen3.6-27b-text', 'qwen3.6-27b')
    assert {'vision', 'text'} <= set(model.available_modalities)
    # Stills as images, clips as video, from the same vision tower.
    assert {'vision:image', 'vision:video'} <= model.in_channels
    assert model.extractor['video'].kwargs['model'] is vision.kwargs['model']
    assert {'behavior:text', 'behavior:probabilities'} <= model.out_channels


def test_regions_route_to_their_tower(built):
    model, _ = built
    selectors = model.region_layer_selectors
    assert (selectors['language_system'].name, selectors['language_system'].modality) == ('layers.40', 'text')
    for region in ('V1', 'V2', 'V4', 'IT'):
        assert selectors[region].modality == 'vision'
        assert int(selectors[region].name.split('.')[1]) < 27


@pytest.mark.parametrize('layer, tower, index', [('blocks.23', 'visual', 23),
                                                  ('layers.40', 'language_model', 40)])
def test_intervention_reaches_the_named_tower(built, layer, tower, index):
    model, _ = built
    module = getattr(model.extractor['vision' if tower == 'visual' else 'text'].kwargs['model'],
                     'blocks' if tower == 'visual' else 'layers')[index]
    x = torch.ones(1, 2)
    assert module(x).abs().sum() > 0
    apply_state_change(model, StateChange(kind='ablation', target=Selection(layer=layer),
                                          perturbation=Perturbation(kind='zero')))
    assert module(x).abs().sum() == 0
    model.reset()
    assert module(x).abs().sum() > 0


def test_answer_sends_the_image_without_thinking(built, tmp_path):
    model, processor = built
    path = tmp_path / 'word.png'
    Image.new('RGB', (8, 8)).save(path)
    stimuli = StimulusSet([{'stimulus_id': 'w0', 'image_file_name': str(path)}])
    stimuli.identifier = 'qwen36-answer-wiring'
    task = TaskContext(task_type='label', label_set=['real', 'pseudo'],
                       instruction='Is this a real word?', prefer_path='generation')
    session = stimulus_session(stimuli, ['behavior'], task_context=task)
    model.participate(session)
    assert list(session.collect('behavior').values.ravel()) == [0.0, 1.0]   # one-hot over (real, pseudo)
    assert processor.template_kwargs['enable_thinking'] is False
    assert len(processor.images) == 1
