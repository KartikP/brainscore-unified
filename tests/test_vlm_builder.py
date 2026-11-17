"""build_vlm_subject on a tiny stand-in: the depth table, the declarations, sessions
through participate, and the provenance entry on a score.

No weights: the checkpoint class is a stand-in set on ``transformers``, the processor,
the download check and the processor pin are replaced, and every real
``from_pretrained`` raises, so no test can download or load a checkpoint.
"""
import sys
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip('torch')
transformers = pytest.importorskip('transformers')

import brainscore
from brainscore import model_registry
from brainscore.model_helpers import vlm
from brainscore_core.benchmarks import BenchmarkBase
from brainscore_core.contract import TaskContext
from brainscore_core.responses import Choice
from brainscore_core.streaming_helpers import stimulus_session
from brainscore_core.supported_data_standards.brainio.stimuli import StimulusSet

pytestmark = pytest.mark.unit

WORDS = ['[PAD]', '[UNK]', 'system', 'user', 'assistant', ':', '<image>', 'cat', 'dog', 'the', 'sat']
VISION_BLOCKS, LANGUAGE_LAYERS = 12, 4


def _tokenizer():
    from tokenizers import Tokenizer, models, pre_tokenizers
    raw = Tokenizer(models.WordLevel(vocab={w: i for i, w in enumerate(WORDS)}, unk_token='[UNK]'))
    raw.pre_tokenizer = pre_tokenizers.Whitespace()
    return transformers.PreTrainedTokenizerFast(tokenizer_object=raw, unk_token='[UNK]', pad_token='[PAD]',
                                                eos_token='[PAD]', model_max_length=64)


class _Vision(torch.nn.Module):
    """A vision part in the flattened-patch layout: one input row per picture."""

    def __init__(self):
        super().__init__()
        self.blocks = torch.nn.ModuleList(torch.nn.Linear(4, 4) for _ in range(VISION_BLOCKS))

    def forward(self, pixel_values, grid_thw=None):
        for block in self.blocks:
            pixel_values = torch.tanh(block(pixel_values))
        return pixel_values


class TinyVLM(torch.nn.Module):
    """Qwen-VL layout: model.visual.blocks, model.language_model.layers, a head; it always writes 'dog'."""
    loads = []

    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        self.model = torch.nn.Module()
        self.model.visual = _Vision()
        self.model.language_model = transformers.Qwen2Model(transformers.Qwen2Config(
            vocab_size=len(WORDS), hidden_size=16, intermediate_size=24, num_hidden_layers=LANGUAGE_LAYERS,
            num_attention_heads=2, num_key_value_heads=1, max_position_embeddings=64))
        self.lm_head = torch.nn.Linear(16, len(WORDS))

    @classmethod
    def from_pretrained(cls, hf_id, **settings):
        cls.loads.append((hf_id, settings))
        return cls().to(settings['dtype'])

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, input_ids, attention_mask=None, pixel_values=None, **_):
        hidden = self.model.language_model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        return SimpleNamespace(logits=self.lm_head(hidden))

    def generate(self, input_ids, **_):
        return torch.cat([input_ids, torch.tensor([[WORDS.index('dog')]])], dim=1)


class FakeProcessor:
    """A chat processor: word tokens, and four numbers per picture (its mean colour and 1)."""

    def __init__(self):
        self.tokenizer = _tokenizer()
        self.image_processor = SimpleNamespace()
        self.video_processor = SimpleNamespace(temporal_patch_size=2, fps=2)

    def apply_chat_template(self, messages, add_generation_prompt=False, **_):
        def words(content):
            return content if isinstance(content, str) else ' '.join(
                '<image>' if part['type'] == 'image' else part.get('text', '') for part in content)
        return ' '.join(f"{m['role']} : {words(m['content'])}" for m in messages) + (
            ' assistant :' if add_generation_prompt else '')

    def __call__(self, text, images=None, return_tensors='pt', padding=True, **_):
        inputs = dict(self.tokenizer(text, return_tensors='pt', padding=True, add_special_tokens=False))
        if images:
            inputs['pixel_values'] = torch.tensor(
                [[*(np.asarray(image, np.float32).mean((0, 1)) / 255), 1.0] for image in images])
        return transformers.BatchFeature(inputs)


@pytest.fixture(autouse=True)
def no_checkpoint(monkeypatch):
    """Every real load refuses; the stand-in class and processor take their place."""
    import brainscore.models._downloads as downloads
    import brainscore_core.hf_compat as hf_compat

    def refuse(*args, **kwargs):
        raise AssertionError('a test must not load a checkpoint')
    monkeypatch.setattr(transformers.PreTrainedModel, 'from_pretrained', classmethod(refuse))
    monkeypatch.setattr(transformers.AutoProcessor, 'from_pretrained', lambda *a, **k: FakeProcessor())
    # The first lazy loads above can put another module object under the name; the
    # builder imports the one there now.
    monkeypatch.setattr(sys.modules['transformers'], 'TinyVLMForConditionalGeneration', TinyVLM, raising=False)
    monkeypatch.setattr(downloads, 'hf_preflight', lambda *a, revision=None, **k: {'local_files_only': True})
    monkeypatch.setattr(hf_compat, 'pin_image_processor', lambda processor, *a, **k: processor)
    monkeypatch.setattr('brainscore.model_helpers._device.select_device', lambda device=None: 'cpu')
    monkeypatch.setenv('RESULTCACHING_DISABLE', '1')
    TinyVLM.loads.clear()


def _build(**settings):
    return vlm.build_vlm_subject('tiny-vlm', 'Org/Tiny-VLM', 'TinyVLMForConditionalGeneration',
                                 vision=('model.visual', 'blocks'), language=('model.language_model', 'layers'),
                                 dtype='float32', download_gb=0.1, **settings)


def _pictures(tmp_path, n=2):
    from PIL import Image
    paths = []
    for index in range(n):
        paths.append(str(tmp_path / f'picture{index}.png'))
        Image.new('RGB', (8, 8), (40 * index, 90, 200)).save(paths[-1])
    stimuli = StimulusSet({'stimulus_id': [f'p{i}' for i in range(n)], 'image_file_name': paths})
    stimuli.identifier = 'vlm-builder-pictures'
    return stimuli


def test_regions_are_the_stated_depths_of_each_part():
    subject = _build()
    layers = subject.region_layer_map
    # 12 vision blocks: index f * 11 rounded half up; 4 language layers: 0.67 * 3 -> 2.
    assert {region: layers[region] for region in ('V1', 'V2', 'V4', 'IT', 'VWFA')} == {
        'V1': 'blocks.1', 'V2': 'blocks.3', 'V4': 'blocks.6', 'IT': 'blocks.10', 'VWFA': 'blocks.10'}
    assert layers['language_system'].name == 'layers.2'
    assert layers['language_system'].modality == ('text', 'vision')


def test_declarations_follow_from_the_parts():
    subject = _build()
    assert {'vision:image', 'text'} <= subject.in_channels and 'vision:video' not in subject.in_channels
    assert {'neural:V1', 'neural:IT', 'neural:language_system', 'behavior:choice', 'behavior:text',
            'behavior:rating', 'behavior:probabilities'} <= subject.out_channels
    assert 'neural:A1' not in subject.out_channels
    assert 'vision:video' in _build(video=True).in_channels


def test_the_registration_states_the_part_for_probabilities():
    inner = _build(probabilities=('likelihood', 'softmax'))._subject
    assert inner.capability_config['behavior-sessions']['behavior:probabilities'] == ('likelihood', 'softmax')
    assert _build()._subject.capability_config['behavior-sessions']['behavior:probabilities'] == 'readout'


def test_pixels_record_the_vision_part_with_pca():
    from brainscore_vision.model_helpers.activations.pca import LayerPCA
    from brainscore_vision.model_helpers.activations.pytorch import PytorchWrapper
    image = _build(pixels={'do_image_splitting': False})._subject.extractor['vision']
    assert isinstance(image, PytorchWrapper) and LayerPCA.is_hooked(image)
    with pytest.raises(ValueError, match='video'):
        _build(pixels={}, video=True)


def test_precision_and_device_map_reach_the_load():
    vlm.build_vlm_subject('tiny-vlm', 'Org/Tiny-VLM', 'TinyVLMForConditionalGeneration',
                          vision=('model.visual', 'blocks'), language=('model.language_model', 'layers'),
                          dtype='bfloat16', download_gb=0.1, device_map='auto')
    hf_id, settings = TinyVLM.loads[-1]
    assert (hf_id, settings['dtype'], settings['device_map']) == ('Org/Tiny-VLM', torch.bfloat16, 'auto')
    assert 'max_memory' in settings and settings['local_files_only'] is True


def test_a_missing_class_is_named_before_the_download_check(monkeypatch):
    import brainscore.models._downloads as downloads

    def download_needed(*args, **kwargs):
        raise RuntimeError('download the checkpoint first')
    monkeypatch.setattr(downloads, 'hf_preflight', download_needed)
    with pytest.raises(ImportError, match='NoSuchVLMForConditionalGeneration'):
        vlm.build_vlm_subject('tiny-vlm', 'Org/Tiny-VLM', 'NoSuchVLMForConditionalGeneration',
                              vision=('model.visual', 'blocks'), language=('model.language_model', 'layers'),
                              dtype='float32', download_gb=0.1)


def test_a_neural_session_records_the_vision_part(tmp_path):
    subject = _build()
    session = stimulus_session(_pictures(tmp_path), ['neural:IT', 'neural:V1'], TaskContext(task_type='passive'))
    subject.participate(session)
    responses = {event.channel: event.payload for event in session.emitted}
    assert responses['neural:IT'].shape == (2, 4) and responses['neural:V1'].shape == (2, 4)
    assert responses['neural:IT'].attrs['provenance']['region_layer_map_status'].startswith('depth hypothesis v1')


def test_a_choice_session_is_answered_by_the_reply(tmp_path):
    subject = _build()
    session = stimulus_session(_pictures(tmp_path), ['behavior:choice'], TaskContext(response=Choice(['cat', 'dog'])))
    subject.participate(session)
    choices = session.collect('behavior:choice')
    assert list(choices.coords['choice'].values[choices.values.argmax(1)]) == ['dog', 'dog']


def test_a_lesion_reaches_the_named_part_and_is_undone(tmp_path):
    from brainscore_core.events import Perturbation, Selection, StateChange
    from brainscore_core.streaming_helpers import apply_state_change
    subject = _build()
    assert {'lesion', 'stimulation', 'pharmacological'} <= subject.in_channels
    assert 'perturbation' in subject.out_channels
    block = subject._subject._model.model.visual.blocks[10]
    x = torch.ones(1, 4)
    before = block(x).abs().sum()
    record = apply_state_change(subject, StateChange(kind='ablation', target=Selection(layer='blocks.10'),
                                                     perturbation=Perturbation(kind='zero')))
    assert block(x).abs().sum() == 0
    apply_state_change(subject, StateChange(kind='reset', handle_id=record.handle_id))
    assert block(x).abs().sum() == before


def test_the_induced_dyslexia_benchmarks_are_compatible():
    from brainscore.benchmarks.induced_dyslexia.benchmark import InducedDyslexia
    from brainscore_core.compatibility import check_compatibility
    benchmark = InducedDyslexia(reading_benchmark=object(), localizer_stimuli=None,
                                identifier='induced-dyslexia-stand-in')
    assert check_compatibility(_build(), benchmark).kind == 'full'
    assert 'perturbation' not in _build(intervention=False).out_channels


class _MeanIT(BenchmarkBase):
    """The mean IT response to two pictures."""

    def __init__(self, stimuli):
        self.stimulus_set = stimuli
        self.required_input_channels = frozenset({'vision:image'})
        self.requested_output_channels = frozenset({'neural:IT'})
        super().__init__(identifier='vlm-builder-mean-it', version=1, parent='neural', ceiling=None)

    def __call__(self, subject):
        from brainscore.benchmarks import benchmark_score, not_estimated
        session = stimulus_session(self.stimulus_set, ['neural:IT'], TaskContext(task_type='passive'))
        subject.participate(session)
        self._ceiling = not_estimated()
        return benchmark_score(self, raw=float(session.collect('neural:IT').values.mean()), metric='mean')


def test_the_depth_hypothesis_reaches_the_score(tmp_path):
    score = brainscore.score(_build(), _MeanIT(_pictures(tmp_path)), check_mem=False)
    registration = [entry['registration'] for entry in score.attrs['provenance'] if 'registration' in entry]
    assert registration and registration[0]['region_layer_map_status'] == (
        'depth hypothesis v1: V1 0.10, V2 0.25, V4 0.50, IT 0.90, VWFA 0.90, language_system 0.67; '
        'no layer search')
    assert (registration[0]['checkpoint'], registration[0]['dtype']) == ('Org/Tiny-VLM', 'float32')


# --- every registration made with the builder -------------------------------------

class _AnyLayout(torch.nn.Module):
    """Has every module path a registration names: the path asked for first ends in four blocks."""

    def get_submodule(self, target):
        module = self
        names = target.split('.')
        for index, name in enumerate(names):
            if name not in module._modules:
                last = index == len(names) - 1
                module.add_module(name, torch.nn.ModuleList(torch.nn.Linear(2, 2) for _ in range(4))
                                  if last else torch.nn.Module())
            module = module._modules[name]
        return module


def _builder_registrations():
    """The registrations whose plugin folder calls the builder (in its __init__.py or model.py)."""
    from pathlib import Path
    found = []
    for identifier, loader in sorted(model_registry.items()):
        module = sys.modules.get(getattr(loader, '__module__', ''))
        if module is None or not getattr(module, '__file__', None):
            continue
        folder = Path(module.__file__).parent
        sources = [folder / name for name in ('__init__.py', 'model.py') if (folder / name).exists()]
        if any('build_vlm_subject' in source.read_text(encoding='utf-8') for source in sources):
            found.append(identifier)
    return found


def test_every_builder_registration_declares_what_its_parts_can_do(monkeypatch):
    identifiers = _builder_registrations()
    channel_of = {'vision': 'vision:image', 'video': 'vision:video', 'text': 'text', 'audio': 'audio'}
    for identifier in identifiers:      # the worked registrations and every later one
        monkeypatch.setattr(vlm, '_load', lambda *args: (_AnyLayout(), FakeProcessor()))
        subject = model_registry[identifier]()
        inner = subject._subject
        assert subject.identifier == identifier
        assert {channel_of[key] for key in inner.extractor} <= subject.in_channels, identifier
        assert {f'neural:{region}' for region in inner.region_layer_map} <= subject.out_channels, identifier
        assert {'behavior:text', 'behavior:probabilities'} <= subject.out_channels, identifier
        assert subject._provenance['region_layer_map_status'].startswith(vlm.DEPTH_HYPOTHESIS), identifier


def test_a_tiling_image_processor_gives_the_whole_image_of_each_picture(tmp_path):
    from brainscore.models._shared import _load_preprocess_images

    def tiles(images, return_tensors, **settings):
        # Five axes, as SmolVLM gives for a list: (samples, tiles, C, H, W). Called on one
        # picture: its whole image (its red value), then a tile of zeros.
        whole = [torch.full((3, 2, 2), float(np.asarray(image)[0, 0, 0])) for image in images]
        return {'pixel_values': torch.stack([torch.stack(whole + [torch.zeros(3, 2, 2)])])}
    paths = list(_pictures(tmp_path, n=3)['image_file_name'])
    pixels = _load_preprocess_images(paths, processor=tiles)
    assert pixels.shape == (3, 3, 2, 2) and list(pixels[:, 0, 0, 0]) == [0, 40, 80]
