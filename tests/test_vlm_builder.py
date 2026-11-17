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
        pixel_values = pixel_values.reshape(len(pixel_values), -1)[:, :4]    # (B, C, H, W) too
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

    def forward(self, input_ids, attention_mask=None, pixel_values=None, image_grid_thw=None, **_):
        if pixel_values is not None:       # the pictures go through the vision part, as in a real model
            self.model.visual(pixel_values, grid_thw=image_grid_thw)
        hidden = self.model.language_model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        return SimpleNamespace(logits=self.lm_head(hidden))

    def generate(self, input_ids, attention_mask=None, pixel_values=None, image_grid_thw=None, **_):
        # The prompt pass a recording keeps, then the fixed reply.
        self(input_ids=input_ids, attention_mask=attention_mask, pixel_values=pixel_values,
             image_grid_thw=image_grid_thw)
        return torch.cat([input_ids, torch.tensor([[WORDS.index('dog')]])], dim=1)


def _image_pixels(images, return_tensors='pt', **settings):
    """Each picture as (3, 2, 2): its mean colour in every pixel."""
    return {'pixel_values': torch.stack([torch.tensor(np.asarray(image, np.float32).mean((0, 1)) / 255)
                                         [:, None, None].expand(3, 2, 2) for image in images])}


def _clip_patches(videos, video_metadata=None, return_tensors='pt'):
    """A clip as flattened patches: two frames per time step, one patch per step."""
    frames = torch.tensor(np.asarray(videos[0], np.float32)) / 255
    steps = len(frames) // 2
    patches = torch.stack([torch.cat([frames[2 * k:2 * k + 2].mean((0, 1, 2)), torch.ones(1)])
                           for k in range(steps)])
    return {'pixel_values_videos': patches, 'video_grid_thw': torch.tensor([[steps, 1, 1]])}


_clip_patches.temporal_patch_size, _clip_patches.fps = 2, 2


def _expected(module, rows, block):
    """The output of ``blocks[block]`` for input rows, computed here, not by an extractor."""
    x = torch.as_tensor(rows, dtype=torch.float32)
    with torch.no_grad():
        for index in range(block):
            x = torch.tanh(module.blocks[index](x))
        return module.blocks[block](x).numpy()    # a block's own output, before the next tanh


def _mean_colours(stimuli):
    from PIL import Image
    return [[*(np.asarray(Image.open(path).convert('RGB'), np.float32).mean((0, 1)) / 255), 1.0]
            for path in stimuli['image_file_name']]


class FakeProcessor:
    """A chat processor in the flattened-patch layout: word tokens, and one patch of four numbers
    per picture (its mean colour and 1)."""

    def __init__(self):
        self.tokenizer = _tokenizer()
        self.image_processor = _image_pixels
        self.video_processor = _clip_patches

    def apply_chat_template(self, messages, add_generation_prompt=False, **_):
        def words(content):
            return content if isinstance(content, str) else ' '.join(
                '<image>' if part['type'] == 'image' else part.get('text', '') for part in content)
        return ' '.join(f"{m['role']} : {words(m['content'])}" for m in messages) + (
            ' assistant :' if add_generation_prompt else '')

    def __call__(self, text, images=None, return_tensors='pt', padding=True, **_):
        inputs = dict(self.tokenizer(text, return_tensors='pt', padding=True, add_special_tokens=False))
        if images:
            inputs['pixel_values'] = torch.tensor(          # one flattened patch for each picture
                [[*(np.asarray(image, np.float32).mean((0, 1)) / 255), 1.0] for image in images])
            inputs['image_grid_thw'] = torch.tensor([[1, 1, 1]] * len(images))
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


def test_image_processor_kwargs_record_the_vision_part_with_pca():
    from brainscore_vision.model_helpers.activations.pca import LayerPCA
    from brainscore_vision.model_helpers.activations.pytorch import PytorchWrapper
    image = _build(image_processor_kwargs={'do_image_splitting': False})._subject.extractor['vision']
    assert isinstance(image, PytorchWrapper) and LayerPCA.is_hooked(image)
    with pytest.raises(ValueError, match='video'):
        _build(image_processor_kwargs={}, video=True)


def test_image_processor_kwargs_give_the_vision_part_whole_pictures(tmp_path, monkeypatch):
    from brainscore_vision.model_helpers.activations.pca import LayerPCA
    monkeypatch.setattr(LayerPCA, 'hook', classmethod(lambda cls, extractor, n_components: None))  # no ImageNet fit
    subject = _build(image_processor_kwargs={})
    pictures = _pictures(tmp_path)
    pictures.stimulus_paths = dict(zip(pictures['stimulus_id'], pictures['image_file_name']))
    session = stimulus_session(pictures, ['neural:IT'], TaskContext(task_type='passive'))
    subject.participate(session)
    # The stand-in image processor gives each picture its mean colour; the vision part reads its first four values.
    colours = [[row[0]] * 4 for row in _mean_colours(pictures)]     # (3, 2, 2) of one colour: red first
    expected = _expected(subject._subject._model.model.visual, colours, 10)
    assert np.allclose(session.collect('neural:IT').values, expected, atol=1e-6)


def test_precision_and_device_map_reach_the_load():
    vlm.build_vlm_subject('tiny-vlm', 'Org/Tiny-VLM', 'TinyVLMForConditionalGeneration',
                          vision=('model.visual', 'blocks'), language=('model.language_model', 'layers'),
                          dtype='bfloat16', download_gb=0.1, device_map='auto')
    hf_id, settings = TinyVLM.loads[-1]
    assert (hf_id, settings['dtype'], settings['device_map']) == ('Org/Tiny-VLM', torch.bfloat16, 'auto')
    assert 'max_memory' in settings and settings['local_files_only'] is True


def test_every_extractor_keeps_the_loaded_placement(monkeypatch):
    """A device map decides where each part is: no extractor moves a part to its own default device."""
    monkeypatch.setattr('brainscore.model_helpers._device.select_device',
                        lambda device=None: torch.device(device) if device else torch.device('meta'))
    for settings in ({}, {'image_processor_kwargs': {}}):
        inner = _build(device_map='auto', **settings)._subject
        for key in ('vision', 'text'):
            module = inner.extractor[key]._model
            assert {p.device.type for p in module.parameters()} == {'cpu'}, (settings, key)


class _MetaVLM(TinyVLM):
    """A checkpoint whose parts a device map left on meta (offloaded)."""

    @classmethod
    def from_pretrained(cls, hf_id, **settings):
        return cls().to('meta')


def test_a_part_left_on_meta_by_a_device_map_stays_there(monkeypatch):
    monkeypatch.setattr(sys.modules['transformers'], 'MetaVLMForConditionalGeneration', _MetaVLM, raising=False)

    def build(**settings):
        return vlm.build_vlm_subject('meta-vlm', 'Org/Meta-VLM', 'MetaVLMForConditionalGeneration',
                                     vision=('model.visual', 'blocks'), language=('model.language_model', 'layers'),
                                     dtype='float32', download_gb=0.1, device_map='auto', **settings)
    inner = build()._subject
    assert {p.device.type for p in inner.extractor['vision']._model.parameters()} == {'meta'}
    with pytest.raises(ValueError, match='one device'):
        build(image_processor_kwargs={})


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
    visual = subject._subject._model.model.visual
    pictures = _pictures(tmp_path)
    assert np.allclose(responses['neural:IT'].values, _expected(visual, _mean_colours(pictures), 10), atol=1e-6)
    assert np.allclose(responses['neural:V1'].values, _expected(visual, _mean_colours(pictures), 1), atol=1e-6)
    assert responses['neural:IT'].attrs['provenance']['region_layer_map_status'].startswith(vlm.DEPTH_HYPOTHESIS)


def _clip(folder, text=None):
    import cv2
    path = str(folder / 'clip.mp4')
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*'mp4v'), 4, (8, 8))
    for frame in range(4):
        writer.write(np.full((8, 8, 3), 60 * frame, np.uint8))
    writer.release()
    stimuli = StimulusSet({'stimulus_id': ['c0'], 'video_path': [path],
                           **({'sentence': [text]} if text else {})})
    stimuli.identifier = 'vlm-builder-clip' + ('-text' if text else '')
    return stimuli


def test_with_video_the_language_layer_is_recorded_during_an_answer_to_a_clip(tmp_path):
    from brainscore_core.compatibility import check_compatibility
    from brainscore_core.responses import Text
    subject = _build(video=True)
    assert 'vision:video' in subject.output_inputs['neural:language_system']
    assert 'vision:video' not in _build().output_inputs['neural:language_system']
    clip_language = SimpleNamespace(required_input_channels=frozenset({'vision:video'}),
                                    requested_output_channels=frozenset({'neural:language_system'}),
                                    trial_memory='none', identifier='clip-language')
    assert check_compatibility(subject, clip_language).kind == 'full'
    session = stimulus_session(_clip(tmp_path, text='the cat sat'), ['behavior:text', 'neural:language_system'],
                               TaskContext(response=Text(max_tokens=2), present=('video', 'text')))
    subject.participate(session)
    assert session.collect('neural:language_system').shape[0] == 1
    assert list(session.collect('behavior:text').values) == ['dog']


def test_a_choice_session_is_answered_by_the_reply(tmp_path):
    subject = _build()
    pictures = _pictures(tmp_path)
    session = stimulus_session(pictures, ['behavior:choice', 'neural:IT'], TaskContext(response=Choice(['cat', 'dog'])))
    subject.participate(session)
    choices = session.collect('behavior:choice')
    assert list(choices.coords['choice'].values[choices.values.argmax(1)]) == ['dog', 'dog']
    # IT is recorded during the answer, from the pictures the reply was shown.
    expected = _expected(subject._subject._model.model.visual, _mean_colours(pictures), 10)
    assert np.allclose(session.collect('neural:IT').values.reshape(2, -1), expected, atol=1e-6)


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
    status = registration[0]['region_layer_map_status']
    assert status.startswith(vlm.DEPTH_HYPOTHESIS) and status.endswith('no layer search')
    assert all(f'{region} {depth:.2f}' in status for region, depth in vlm.DEPTH_TABLE['vision'].items())
    assert (registration[0]['checkpoint'], registration[0]['dtype']) == ('Org/Tiny-VLM', 'float32')


# --- every registration made with the builder -------------------------------------

class _Part(torch.nn.Module):
    """A model part with four blocks at the stated block path; it runs them in order."""

    def __init__(self, blocks, words=False):
        super().__init__()
        parent = self
        *names, last = blocks.split('.')
        for name in names:
            parent.add_module(name, torch.nn.Module())
            parent = getattr(parent, name)
        parent.add_module(last, torch.nn.ModuleList(torch.nn.Linear(4, 4) for _ in range(4)))
        self._path = blocks
        self.embed = torch.nn.Embedding(len(WORDS), 4) if words else None

    def forward(self, pixel_values=None, input_ids=None, attention_mask=None, grid_thw=None, **_):
        x = self.embed(input_ids) if self.embed is not None else (
            pixel_values.reshape(len(pixel_values), -1)[:, :4].float())
        for block in self.get_submodule(self._path):
            x = torch.tanh(block(x))
        return SimpleNamespace(last_hidden_state=x)


class _StatedLayout(torch.nn.Module):
    """A stand-in with its parts at the paths a registration states (a typo in a path fails
    here, as it would on the real class); it shows pictures to its vision part and writes 'dog'."""

    def __init__(self, vision, language):
        super().__init__()
        torch.manual_seed(0)
        for (root, blocks), words in ((vision, False), (language, True)):
            *names, last = root.split('.')
            parent = self
            for name in names:
                if name not in parent._modules:
                    parent.add_module(name, torch.nn.Module())
                parent = parent._modules[name]
            parent.add_module(last, _Part(blocks, words))
        self._vision, self._language = vision[0], language[0]
        self.head = torch.nn.Linear(4, len(WORDS))

    @property
    def device(self):
        return self.head.weight.device

    def forward(self, input_ids, attention_mask=None, pixel_values=None, image_grid_thw=None, **_):
        if pixel_values is not None:
            self.get_submodule(self._vision)(pixel_values=pixel_values, grid_thw=image_grid_thw)
        hidden = self.get_submodule(self._language)(input_ids=input_ids).last_hidden_state
        return SimpleNamespace(logits=self.head(hidden))

    def generate(self, input_ids, attention_mask=None, pixel_values=None, image_grid_thw=None, **_):
        self(input_ids=input_ids, attention_mask=attention_mask, pixel_values=pixel_values,
             image_grid_thw=image_grid_thw)
        return torch.cat([input_ids, torch.tensor([[WORDS.index('dog')]])], dim=1)


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


WORKED = {'qwen3-vl-2b', 'smolvlm2-256m'}


def _on_stand_ins(monkeypatch):
    """Each registration made with the builder, built on a stand-in with its stated paths."""
    import importlib
    from brainscore_vision.model_helpers.activations.pca import LayerPCA
    monkeypatch.setattr(LayerPCA, 'hook', classmethod(lambda cls, extractor, n_components: None))  # no ImageNet fit
    real = vlm.build_vlm_subject

    def on_stand_in(*args, **kwargs):
        monkeypatch.setattr(vlm, '_load', lambda *a: (_StatedLayout(kwargs['vision'], kwargs['language']),
                                                      FakeProcessor()))
        return real(*args, **{**kwargs, 'dtype': 'float32'})
    identifiers = _builder_registrations()
    assert WORKED <= set(identifiers)
    for identifier in identifiers:
        package = model_registry[identifier].__module__
        for name in (package, f'{package}.model'):
            try:
                module = importlib.import_module(name)
            except ImportError:
                continue
            if hasattr(module, 'build_vlm_subject'):
                monkeypatch.setattr(module, 'build_vlm_subject', on_stand_in)
        yield identifier, model_registry[identifier]()


def _given(subject, channel, folder):
    """The rows one small session gives on ``channel``; for an input, a session that uses it."""
    from brainscore_core.events import Perturbation, Selection, StateChange
    from brainscore_core.responses import Scale, Text
    from brainscore_core.streaming_helpers import apply_state_change
    pictures = _pictures(folder, n=4)
    pictures.stimulus_paths = dict(zip(pictures['stimulus_id'], pictures['image_file_name']))
    fitting = StimulusSet({'stimulus_id': list(pictures['stimulus_id']),
                           'image_file_name': list(pictures['image_file_name']),
                           'image_label': ['cat', 'dog', 'cat', 'dog']})
    fitting.identifier = 'vlm-builder-fitting'
    fitting.stimulus_paths = dict(pictures.stimulus_paths)
    sentences = StimulusSet({'stimulus_id': ['t0', 't1'], 'sentence': ['the cat sat', 'the dog sat']})
    sentences.identifier = 'vlm-builder-sentences'
    continuations = StimulusSet({'stimulus_id': ['t0', 't1'], 'sentence': ['the cat sat', 'the dog sat'],
                                 'options': [('cat', 'dog')] * 2})
    continuations.identifier = 'vlm-builder-continuations'
    choice = Choice(['cat', 'dog'])
    tasks = {
        'behavior': (pictures, TaskContext(response=choice)),
        'behavior:choice': (pictures, TaskContext(response=choice)),
        'behavior:probabilities': (pictures, TaskContext(response=choice, fitting_stimuli=fitting)),
        'behavior:rt': (pictures, TaskContext(response=Choice(['cat', 'dog'], reaction_time='surprisal'),
                                              fitting_stimuli=fitting)),
        'behavior:rating': (sentences, TaskContext(response=Scale(1, 7, 1))),
        'behavior:text': (sentences, TaskContext(response=Text(max_tokens=4))),
        'behavior:log_probabilities': (continuations, TaskContext(response=Choice(labels='options', verbatim=True))),
    }
    if channel == 'perturbation':
        record = apply_state_change(subject, StateChange(
            kind='ablation', target=Selection(layer=subject.region_layer_map['IT']),
            perturbation=Perturbation(kind='zero')))
        apply_state_change(subject, StateChange(kind='reset', handle_id=record.handle_id))
        return int(record.handle_id is not None)
    if channel == 'vision:video':
        stimuli, task, channel = _clip(folder), TaskContext(task_type='passive'), 'neural:IT'
    elif channel.startswith('neural:'):
        # A passive session presents what the recording extractor takes: text for the language layer.
        stimuli = sentences if 'text' in subject.output_inputs[channel] else pictures
        task = TaskContext(task_type='passive')
    else:
        assert channel in tasks, f'no small session for {channel}: add one to _given'
        stimuli, task = tasks[channel]
    session = stimulus_session(stimuli, [channel], task)
    subject.participate(session)
    return session.collect(channel).shape[0]


def test_every_builder_registration_gives_what_it_declares(monkeypatch, tmp_path):
    """Every output a registration made with the builder declares (and clips, when it takes
    them) comes out of one small session on its stand-in."""
    channel_of = {'vision': 'vision:image', 'video': 'vision:video', 'text': 'text', 'audio': 'audio'}
    for identifier, subject in _on_stand_ins(monkeypatch):
        inner = subject._subject
        assert subject.identifier == identifier
        assert subject._provenance['region_layer_map_status'].startswith(vlm.DEPTH_HYPOTHESIS), identifier
        assert {channel_of[key] for key in inner.extractor} <= subject.in_channels, identifier
        assert {f'neural:{region}' for region in inner.region_layer_map} <= subject.out_channels, identifier
        assert {'behavior:choice', 'behavior:probabilities', 'behavior:log_probabilities', 'behavior:rating',
                'behavior:rt', 'behavior:text', 'perturbation', 'neural:VWFA'} <= subject.out_channels, identifier
        assert {'lesion', 'stimulation', 'pharmacological'} <= subject.in_channels, identifier
        for channel in sorted(subject.out_channels) + sorted({'vision:video'} & subject.in_channels):
            if channel == 'neural:A1':
                continue    # the audio extractor is the registration's own part; its test.py runs it
            assert _given(subject, channel, tmp_path) > 0, (identifier, channel)
        subject.reset()


@pytest.mark.parametrize('model_class, vision, language, config', [
    ('Qwen3VLForConditionalGeneration', 'model.visual.blocks', 'model.language_model.layers', lambda t: t.Qwen3VLConfig(
        text_config=dict(num_hidden_layers=2, hidden_size=64, intermediate_size=64, num_attention_heads=2,
                         num_key_value_heads=1, head_dim=32, rope_scaling={'rope_type': 'default', 'mrope_section': [8, 4, 4]}),
        vision_config=dict(depth=3, hidden_size=32, num_heads=2, intermediate_size=32, out_hidden_size=64,
                           deepstack_visual_indexes=[0, 1]))),
    ('SmolVLMForConditionalGeneration', 'model.vision_model.encoder.layers', 'model.text_model.layers', lambda t: t.SmolVLMConfig(
        text_config=dict(vocab_size=1000, pad_token_id=2, num_hidden_layers=3, hidden_size=64, intermediate_size=64,
                         num_attention_heads=2, num_key_value_heads=1),
        vision_config=dict(num_hidden_layers=2, hidden_size=32, intermediate_size=32, num_attention_heads=2))),
])
def test_the_worked_registrations_name_paths_of_their_real_classes(model_class, vision, language, config):
    """The real class, built from a small config on the meta device (no weights): each stated path
    is a list of blocks."""
    live = sys.modules['transformers']
    with torch.device('meta'):
        network = getattr(live, model_class)(config(live))
    for path in (vision, language):
        assert isinstance(network.get_submodule(path), torch.nn.ModuleList) and len(network.get_submodule(path)) > 1


def _wide_pictures(folder, n=2):
    """Pictures three times wider than high: a processor that tiles them gives crops."""
    from PIL import Image
    paths = []
    for index in range(n):
        paths.append(str(folder / f'wide{index}.png'))
        Image.new('RGB', (96, 32), (40 * index, 90, 200)).save(paths[-1])
    return paths


@pytest.mark.parametrize('processor, whole, tiled', [
    ('SmolVLMImageProcessor', {'do_image_splitting': False}, {'do_image_splitting': True}),
    ('Gemma3ImageProcessor', {'do_pan_and_scan': False}, {'do_pan_and_scan': True, 'pan_and_scan_min_crop_size': 8,
                                                           'pan_and_scan_max_num_crops': 4,
                                                           'pan_and_scan_min_ratio_to_activate': 1.0}),
    ('GotOcr2ImageProcessor', {'crop_to_patches': False}, {'crop_to_patches': True, 'min_patches': 2, 'max_patches': 6}),
])
def test_one_whole_image_per_picture_or_a_refusal(tmp_path, processor, whole, tiled):
    """The installed processors of the families: the stated settings give one whole image for
    each picture; settings that give tiles or crops are refused with the setting to state."""
    small = {'SmolVLMImageProcessor': dict(size={'longest_edge': 96}, max_image_size={'longest_edge': 32}),
             'Gemma3ImageProcessor': dict(size={'height': 32, 'width': 32}),
             'GotOcr2ImageProcessor': dict(size={'height': 32, 'width': 32})}
    image_processor = getattr(sys.modules['transformers'], processor)(**small[processor])
    paths = _wide_pictures(tmp_path)
    pixels = vlm._whole_images(paths, image_processor, whole)
    assert pixels.shape == (2, 3, 32, 32)
    with pytest.raises(ValueError, match='not one whole image'):
        vlm._whole_images(paths, image_processor, tiled)
