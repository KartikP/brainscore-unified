"""The study's models: offline on a stand-in of each model's stated layout (no weights: every real
``from_pretrained`` raises), and with the real weights in the tests marked ``slow`` (a GPU machine).

    pytest brainscore/models/umi_dev_models/test.py -m "not slow"     # offline
    BRAINSCORE_SKIP_MODEL_DOWNLOAD_CHECK=1 pytest brainscore/models/umi_dev_models/test.py -m slow -k qwen3-vl-2b
"""
import sys
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip('torch')
transformers = pytest.importorskip('transformers')

from brainscore import model_registry
from brainscore_core.contract import TaskContext
from brainscore_core.responses import Choice, Scale, Text
from brainscore_core.streaming_helpers import stimulus_session
from brainscore_core.supported_data_standards.brainio.stimuli import StimulusSet

from . import model as study

IDENTIFIERS = sorted(study.MODEL_CONFIGS)
WORDS = ['[PAD]', '[UNK]', 'system', 'user', 'assistant', ':', '<image>', 'cat', 'dog', 'the', 'sat', 'Action', '1', '9']
REPLY = ['dog']    # the words the stand-in writes; a test may set another reply


# --- the stand-in ---------------------------------------------------------------------------

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
        self.passes = 0

    def forward(self, pixel_values=None, input_ids=None, attention_mask=None, grid_thw=None, **_):
        self.passes += 1
        x = self.embed(input_ids) if self.embed is not None else (
            pixel_values.reshape(len(pixel_values), -1)[:, :4].to(next(self.parameters()).dtype))
        for block in self.get_submodule(self._path):
            x = torch.tanh(block(x))
        return SimpleNamespace(last_hidden_state=x)


class _StatedLayout(torch.nn.Module):
    """The parts at the paths a configuration states. Pictures and clips go through the vision part,
    and the language part gets the extra inputs the configuration states, as the real model gives
    them (Qwen3-VL: visual_pos_masks; Gemma 3: its attention_mask as a mapping of causal masks)."""

    def __init__(self, config):
        super().__init__()
        torch.manual_seed(0)
        for (root, blocks), words in ((config['vision'], False), (config['language'], True)):
            *names, last = root.split('.')
            parent = self
            for name in names:
                if name not in parent._modules:
                    parent.add_module(name, torch.nn.Module())
                parent = parent._modules[name]
            parent.add_module(last, _Part(blocks, words))
        self._vision, self._language = config['vision'][0], config['language'][0]
        self._extra = config['extra_forward_inputs']
        self.head = torch.nn.Linear(4, len(WORDS))

    @property
    def device(self):
        return self.head.weight.device

    def forward(self, input_ids, attention_mask=None, pixel_values=None, image_grid_thw=None,
                pixel_values_videos=None, video_grid_thw=None, **_):
        shown = False
        for given, grid in ((pixel_values, image_grid_thw), (pixel_values_videos, video_grid_thw)):
            if given is not None:
                self.get_submodule(self._vision)(pixel_values=given, grid_thw=grid)
                shown = True
        extra = {}
        if shown and 'visual_pos_masks' in self._extra:
            extra['visual_pos_masks'] = input_ids == WORDS.index('<image>')
        if any('mapping' in name for name in self._extra):
            attention_mask = {'full_attention': torch.zeros(1, 1, input_ids.shape[1], input_ids.shape[1])}
        hidden = self.get_submodule(self._language)(input_ids=input_ids, attention_mask=attention_mask,
                                                    **extra).last_hidden_state
        return SimpleNamespace(logits=self.head(hidden))

    def generate(self, input_ids, **inputs):
        self(input_ids=input_ids, **inputs)       # the prompt pass a recording keeps
        return torch.cat([input_ids, torch.tensor([[WORDS.index(word) for word in REPLY]])], dim=1)


def _tokenizer():
    from tokenizers import Tokenizer, models, pre_tokenizers
    raw = Tokenizer(models.WordLevel(vocab={w: i for i, w in enumerate(WORDS)}, unk_token='[UNK]'))
    raw.pre_tokenizer = pre_tokenizers.Whitespace()
    return transformers.PreTrainedTokenizerFast(tokenizer_object=raw, unk_token='[UNK]', pad_token='[PAD]',
                                                eos_token='[PAD]', model_max_length=64)


def _image_pixels(images, return_tensors='pt', **settings):
    """Each picture as (3, 2, 2): its mean colour in every pixel (one whole image)."""
    return {'pixel_values': torch.stack([torch.tensor(np.asarray(image, np.float32).mean((0, 1)) / 255)
                                         [:, None, None].expand(3, 2, 2) for image in images])}


def _clip_patches(videos, video_metadata=None, return_tensors='pt'):
    """A clip as flattened patches: two frames per time step, one patch per step."""
    frames = torch.tensor(np.asarray(videos[0], np.float32)) / 255
    steps = len(frames) // 2
    patches = torch.stack([torch.cat([frames[2 * k:2 * k + 2].mean((0, 1, 2)), torch.ones(1)]) for k in range(steps)])
    return {'pixel_values_videos': patches, 'video_grid_thw': torch.tensor([[steps, 1, 1]])}


_clip_patches.temporal_patch_size, _clip_patches.fps = 2, 2


class FakeProcessor:
    """A chat processor with the keys of the Qwen3-VL processor: input_ids, attention_mask; with
    pictures pixel_values and image_grid_thw; with clips pixel_values_videos and video_grid_thw."""

    def __init__(self):
        self.tokenizer, self.image_processor, self.video_processor = _tokenizer(), _image_pixels, _clip_patches

    def apply_chat_template(self, messages, add_generation_prompt=False, **_):
        def words(content):
            return content if isinstance(content, str) else ' '.join(
                '<image>' if part['type'] in ('image', 'video') else part.get('text', '') for part in content)
        return ' '.join(f"{m['role']} : {words(m['content'])}" for m in messages) + (
            ' assistant :' if add_generation_prompt else '')

    def __call__(self, text, images=None, videos=None, return_tensors='pt', padding=True, **_):
        inputs = dict(self.tokenizer(text, return_tensors='pt', padding=True, add_special_tokens=False))
        if videos:
            inputs.update(_clip_patches(videos))
        if images:
            inputs['pixel_values'] = torch.tensor([[*(np.asarray(image, np.float32).mean((0, 1)) / 255), 1.0]
                                                   for image in images])
            inputs['image_grid_thw'] = torch.tensor([[1, 1, 1]] * len(images))
        return transformers.BatchFeature(inputs)


@pytest.fixture
def stand_in(monkeypatch, tmp_path):
    """Every model of the study on a stand-in of its stated layout, in its stated precision, with no
    checkpoint and no download; the pixel path's LayerPCA fits for real (two components, four pictures)."""
    from brainscore_vision.model_helpers.activations import pca

    def refuse(*args, **kwargs):
        raise AssertionError('a test must not load a checkpoint')
    monkeypatch.setattr(transformers.PreTrainedModel, 'from_pretrained', classmethod(refuse))
    monkeypatch.setattr('brainscore.model_helpers._device.select_device', lambda device=None: 'cpu')
    monkeypatch.setattr(study, '_refuse', lambda identifier, config: None)

    def load(identifier, config):
        return _StatedLayout(config).to(getattr(torch, config['dtype'])), FakeProcessor()
    monkeypatch.setattr(study, '_load', load)
    corpus = list(_pictures(tmp_path, n=4)['image_file_name'])
    monkeypatch.setattr(pca, '_get_imagenet_val', lambda num_images: corpus)
    hook = pca.LayerPCA.hook.__func__
    monkeypatch.setattr(pca.LayerPCA, 'hook', classmethod(lambda cls, extractor, n_components: hook(cls, extractor, 2)))
    return tmp_path


def _pictures(folder, n=2, labels=None):
    from PIL import Image
    paths = []
    for index in range(n):
        paths.append(str(folder / f'picture{index}.png'))
        Image.new('RGB', (8, 8), (40 * index, 90, 200)).save(paths[-1])
    stimuli = StimulusSet({'stimulus_id': [f'p{i}' for i in range(n)], 'image_file_name': paths,
                           **({'image_label': labels} if labels else {})})
    stimuli.identifier = f'umi-dev-pictures-{n}-{bool(labels)}'
    stimuli.stimulus_paths = dict(zip(stimuli['stimulus_id'], paths))    # the legacy image extractor reads these
    return stimuli


def _clip(folder):
    import cv2
    path = str(folder / 'clip.mp4')
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*'mp4v'), 4, (64, 64))   # real video processors need 32 or more
    for frame in range(4):
        writer.write(np.full((64, 64, 3), 60 * frame, np.uint8))
    writer.release()
    stimuli = StimulusSet({'stimulus_id': ['c0'], 'video_path': [path]})
    stimuli.identifier = 'umi-dev-clip'
    return stimuli


def _collect(subject, stimuli, outputs, task):
    session = stimulus_session(stimuli, list(outputs), task)
    subject.participate(session)
    return {channel: session.collect(channel) for channel in outputs}


# --- offline tests --------------------------------------------------------------------------

def test_the_depth_rule():
    vision = study.depth_layers([f'blocks.{i}' for i in range(32)], study.DEPTH_TABLE['vision'])
    assert vision == {'V1': 'blocks.3', 'V2': 'blocks.8', 'V4': 'blocks.16', 'IT': 'blocks.28', 'VWFA': 'blocks.28'}
    assert study.depth_layers([f'l{i}' for i in range(36)], study.DEPTH_TABLE['language']) == {'language_system': 'l23'}
    assert study.depth_layers([f'l{i}' for i in range(26)], {'V1': 0.10}) == {'V1': 'l3'}     # 2.5 rounds up


def test_every_identifier_is_registered_with_its_suffix():
    assert set(IDENTIFIERS) <= set(model_registry) and all(name.endswith('_UMI-dev') for name in IDENTIFIERS)


@pytest.mark.parametrize('identifier', IDENTIFIERS)
def test_each_declared_output_comes_out_of_one_session(stand_in, identifier):
    """The declarations follow from the stated parts, and every declared output (and clips, when
    the model takes them) comes out of a small session through participate."""
    global REPLY
    from brainscore_core.events import Perturbation, Selection, StateChange
    from brainscore_core.streaming_helpers import apply_state_change
    config, subject = study.MODEL_CONFIGS[identifier], model_registry[identifier]()
    folder = stand_in
    pictures, fitting = _pictures(folder, 4), _pictures(folder, 4, labels=['cat', 'dog', 'cat', 'dog'])
    sentences = StimulusSet({'stimulus_id': ['t0', 't1'], 'sentence': ['the cat sat', 'the dog sat']})
    sentences.identifier = 'umi-dev-sentences'
    passive = TaskContext(task_type='passive')
    provenance = subject._provenance
    assert provenance['region_layer_map_status'].startswith(study.DEPTH_HYPOTHESIS)
    assert provenance['readout_layer'].startswith(subject.region_layer_map['IT'])
    assert provenance['revision'] == config['revision']
    assert {f'neural:{r}' for r in ('V1', 'V2', 'V4', 'IT', 'VWFA', 'language_system')} | {
        'behavior:choice', 'behavior:probabilities', 'behavior:log_probabilities', 'behavior:rating',
        'behavior:rt', 'behavior:text', 'motor'} <= subject.out_channels
    assert subject.output_inputs['motor'] == {'observation'}
    assert ('vision:video' in subject.in_channels) == config['video']
    pixel = config['image_processor_kwargs'] is not None
    assert ('perturbation' in subject.out_channels) == (not pixel) == ('lesion' in subject.in_channels)
    # The language layer during an answer to a picture is declared only where v2's recording reads that pass.
    assert ('vision:image' in subject.output_inputs['neural:language_system']) == (not config['extra_forward_inputs'])

    for region in ('V1', 'V2', 'V4', 'IT', 'VWFA'):
        values = _collect(subject, pictures, [f'neural:{region}'], passive)[f'neural:{region}'].values
        assert values.shape[0] == 4 and values.dtype == np.float32 and np.isfinite(values).all(), region
        assert values.shape[1] == (2 if pixel else 4), region         # the fitted PCA's components, or the hidden width
    language = _collect(subject, sentences, ['neural:language_system'], passive)['neural:language_system']
    assert language.shape[0] == 2 and np.isfinite(language.values).all()
    if 'vision:image' in subject.output_inputs['neural:language_system']:
        during = _collect(subject, pictures, ['behavior:text', 'neural:language_system'], TaskContext(response=Text(max_tokens=2)))
        assert during['neural:language_system'].shape[0] == 4
    else:   # not declared (v2's recording cannot read this model's answering pass): N/A in pre-flight
        from brainscore_core.compatibility import CompatibilityError, check_compatibility
        with pytest.raises(CompatibilityError, match='language_system'):
            check_compatibility(subject, SimpleNamespace(
                identifier='picture-language', trial_memory='none', required_input_channels=frozenset({'vision:image'}),
                requested_output_channels=frozenset({'behavior:text', 'neural:language_system'})))
    if config['video']:
        assert _collect(subject, _clip(folder), ['neural:IT'], passive)['neural:IT'].shape[0] == 1

    choice = Choice(['cat', 'dog'])
    for channel, stimuli, task in [
            ('behavior:choice', pictures, TaskContext(response=choice)),
            ('behavior:text', sentences, TaskContext(response=Text(max_tokens=2))),
            ('behavior:probabilities', pictures, TaskContext(response=choice, fitting_stimuli=fitting)),
            ('behavior:rt', pictures, TaskContext(response=Choice(['cat', 'dog'], reaction_time='surprisal'),
                                                  fitting_stimuli=fitting))]:
        answer = _collect(subject, stimuli, [channel], task)[channel]
        assert answer.shape[0] == len(stimuli), channel
    continuations = StimulusSet({'stimulus_id': ['t0'], 'sentence': ['the cat'], 'options': [('sat', 'dog')]})
    continuations.identifier = 'umi-dev-continuations'
    logp = _collect(subject, continuations, ['behavior:log_probabilities'],
                    TaskContext(response=Choice(labels='options', verbatim=True)))['behavior:log_probabilities']
    assert np.isfinite(logp.values).all()
    given = REPLY
    try:
        REPLY = ['1']
        rating = _collect(subject, sentences, ['behavior:rating'], TaskContext(response=Scale(1, 7, 1)))['behavior:rating']
        assert list(rating.values) == [1.0, 1.0]
        REPLY = ['9']
        outside = _collect(subject, sentences, ['behavior:rating'], TaskContext(response=Scale(1, 7, 1)))['behavior:rating']
        assert set(outside['response_state'].values) == {'invalid'}
        REPLY = ['Action', ':', '1']
        from brainscore.harnesses.grid_game import GridGameEnv, play_game
        assert set(play_game(subject, GridGameEnv(size=5, seed=1, max_steps=3))['actions']) == {1}
    finally:
        REPLY = given
    subject.reset()

    if not pixel:   # a lesion changes exactly the selected units and is gone after its undo
        intact = _collect(subject, pictures, ['neural:IT'], passive)['neural:IT'].values
        record = apply_state_change(subject, StateChange(
            kind='ablation', target=Selection(layer=subject.region_layer_map['IT'], indices=[0, 1]),
            perturbation=Perturbation(kind='zero')))
        lesioned = _collect(subject, pictures, ['neural:IT'], passive)['neural:IT'].values
        assert (lesioned[:, :2] == 0).all() and np.array_equal(lesioned[:, 2:], intact[:, 2:])
        apply_state_change(subject, StateChange(kind='reset', handle_id=record.handle_id))
        assert np.array_equal(_collect(subject, pictures, ['neural:IT'], passive)['neural:IT'].values, intact)


@pytest.mark.parametrize('identifier', IDENTIFIERS)
def test_a_written_answer_with_the_language_layer_runs_or_is_refused_before_compute(stand_in, identifier):
    """A trial of a picture and text, asking for a written answer and the language layer: the answering
    pass is recorded where v2 can read it, else the session is refused before any forward pass. Pre-flight
    cannot tell the two apart (it accepts the text input for the language layer)."""
    from brainscore_core.compatibility import ExecutionNotFeasibleError, check_compatibility
    config, subject = study.MODEL_CONFIGS[identifier], model_registry[identifier]()
    pictures = _pictures(stand_in, 2)
    trials = StimulusSet({'stimulus_id': ['p0', 'p1'], 'image_file_name': list(pictures['image_file_name']),
                          'sentence': ['the cat', 'the dog']})
    trials.identifier = 'umi-dev-picture-and-text'
    outputs = ['behavior:text', 'neural:language_system']
    assert check_compatibility(subject, SimpleNamespace(
        identifier='picture-and-text', trial_memory='none', required_input_channels=frozenset({'vision:image', 'text'}),
        requested_output_channels=frozenset(outputs))).kind == 'full'
    task = TaskContext(response=Text(max_tokens=2), present=('image', 'text'))
    network = subject._subject._model
    vision = network.get_submodule(config['vision'][0])
    if config['extra_forward_inputs']:
        with pytest.raises(ExecutionNotFeasibleError, match='language_system'):
            _collect(subject, trials, outputs, task)
        assert vision.passes == 0
    else:
        assert _collect(subject, trials, outputs, task)['neural:language_system'].shape[0] == 2


@pytest.mark.parametrize('processor, whole, tiled', [
    ('SmolVLMImageProcessor', {'do_image_splitting': False}, {'do_image_splitting': True}),
    ('Gemma3ImageProcessor', {'do_pan_and_scan': False}, {'do_pan_and_scan': True, 'pan_and_scan_min_crop_size': 8,
                                                           'pan_and_scan_max_num_crops': 4,
                                                           'pan_and_scan_min_ratio_to_activate': 1.0}),
    ('GotOcr2ImageProcessor', {'crop_to_patches': False}, {'crop_to_patches': True, 'min_patches': 2, 'max_patches': 6}),
])
def test_one_whole_image_per_picture_or_a_refusal(tmp_path, processor, whole, tiled):
    """The installed image processors of the families: the stated settings give one whole image."""
    from PIL import Image
    small = {'SmolVLMImageProcessor': dict(size={'longest_edge': 96}, max_image_size={'longest_edge': 32}),
             'Gemma3ImageProcessor': dict(size={'height': 32, 'width': 32}),
             'GotOcr2ImageProcessor': dict(size={'height': 32, 'width': 32})}
    image_processor = getattr(sys.modules['transformers'], processor)(**small[processor])
    paths = []
    for index in range(2):
        paths.append(str(tmp_path / f'wide{index}.png'))
        Image.new('RGB', (96, 32), (40 * index, 90, 200)).save(paths[-1])
    assert study._whole_images(paths, image_processor, whole).shape == (2, 3, 32, 32)
    with pytest.raises(ValueError, match='not one whole image'):
        study._whole_images(paths, image_processor, tiled)


def test_a_part_offloaded_or_spread_is_refused():
    part = torch.nn.Linear(2, 2).to('meta')
    with pytest.raises(ValueError, match='one accelerator'):
        study._one_device('m', part, 'vision')


def test_a_missing_package_and_a_missing_class_are_named_before_any_download(monkeypatch):
    """The one statement of a package is requirements.txt; a load without it says what to install."""
    assert study.REQUIREMENTS == ['num2words']
    import transformers.utils
    from brainscore import LoadError
    monkeypatch.setattr(transformers.utils, 'is_num2words_available', lambda: False)
    monkeypatch.setattr(study, '_load', lambda *a: pytest.fail('no load before the refusal'))
    with pytest.raises(LoadError, match="needs the package 'num2words', which is not installed. Next step"):
        model_registry['smolvlm2-256m_UMI-dev']()
    monkeypatch.setattr(transformers.utils, 'is_num2words_available', lambda: True)
    with pytest.raises(LoadError, match='NoSuchVLM'):
        study._refuse('m', dict(study.MODEL_CONFIGS['qwen3-vl-2b_UMI-dev'], model_class='NoSuchVLM'))


@pytest.mark.parametrize('model_class, config', [
    ('Qwen3VLForConditionalGeneration', lambda t: t.Qwen3VLConfig(
        text_config=dict(num_hidden_layers=2, hidden_size=64, intermediate_size=64, num_attention_heads=2, num_key_value_heads=1,
                         head_dim=32, rope_scaling={'rope_type': 'default', 'mrope_section': [8, 4, 4]}),
        vision_config=dict(depth=3, hidden_size=32, num_heads=2, intermediate_size=32, out_hidden_size=64,
                           deepstack_visual_indexes=[0, 1]))),
    ('Qwen3VLMoeForConditionalGeneration', lambda t: t.Qwen3VLMoeConfig(
        text_config=dict(num_hidden_layers=2, hidden_size=64, intermediate_size=64, num_attention_heads=2, num_key_value_heads=1,
                         head_dim=32, rope_scaling={'rope_type': 'default', 'mrope_section': [8, 4, 4]}, num_experts=2,
                         num_experts_per_tok=1, moe_intermediate_size=32),
        vision_config=dict(depth=3, hidden_size=32, num_heads=2, intermediate_size=32, out_hidden_size=64,
                           deepstack_visual_indexes=[0, 1]))),
    ('Qwen2_5_VLForConditionalGeneration', lambda t: t.Qwen2_5_VLConfig(
        text_config=dict(vocab_size=300, num_hidden_layers=2, hidden_size=64, intermediate_size=64, num_attention_heads=2,
                         num_key_value_heads=1, rope_scaling={'type': 'mrope', 'mrope_section': [8, 8, 16]}),
        vision_config=dict(depth=3, hidden_size=32, num_heads=2, intermediate_size=32, out_hidden_size=64,
                           fullatt_block_indexes=[2]))),
    ('SmolVLMForConditionalGeneration', lambda t: t.SmolVLMConfig(
        text_config=dict(vocab_size=1000, pad_token_id=2, num_hidden_layers=3, hidden_size=64, intermediate_size=64,
                         num_attention_heads=2, num_key_value_heads=1),
        vision_config=dict(num_hidden_layers=2, hidden_size=32, intermediate_size=32, num_attention_heads=2))),
    ('Gemma3ForConditionalGeneration', lambda t: t.Gemma3Config(
        text_config=dict(vocab_size=300, hidden_size=32, intermediate_size=32, num_hidden_layers=3, num_attention_heads=2,
                         num_key_value_heads=1, head_dim=16),
        vision_config=dict(hidden_size=32, intermediate_size=32, num_hidden_layers=2, num_attention_heads=2,
                           image_size=28, patch_size=14), mm_tokens_per_image=4)),
    ('InternVLForConditionalGeneration', lambda t: t.InternVLConfig(
        text_config=dict(model_type='qwen2', vocab_size=300, hidden_size=32, intermediate_size=32, num_hidden_layers=3,
                         num_attention_heads=2, num_key_value_heads=1),
        vision_config=dict(hidden_size=32, intermediate_size=32, num_hidden_layers=2, num_attention_heads=2,
                           image_size=[28, 28], patch_size=[14, 14]))),
])
def test_the_stated_paths_are_block_lists_of_the_real_classes(model_class, config):
    """Each family's class, built from a small config on the meta device (no weights)."""
    live = sys.modules['transformers']
    network_class = getattr(live, model_class)
    with torch.device('meta'):
        network = network_class(config(live))
    for row in (c for c in study.MODEL_CONFIGS.values() if c['model_class'] == model_class):
        for root, blocks in (row['vision'], row['language']):
            assert isinstance(network.get_submodule(f'{root}.{blocks}'), torch.nn.ModuleList)


# --- with the real weights ------------------------------------------------------------------

@pytest.mark.slow
@pytest.mark.parametrize('identifier', IDENTIFIERS)
def test_the_real_model(identifier, tmp_path):
    subject = model_registry[identifier]()
    neural = _collect(subject, _pictures(tmp_path, 2), ['neural:V1', 'neural:IT'], TaskContext(task_type='passive'))
    assert all(a.shape[0] == 2 and np.isfinite(a.values).all() for a in neural.values())
    sentences = StimulusSet({'stimulus_id': ['s0', 's1'], 'sentence': ['the cat sat on the mat', 'a dog runs fast']})
    sentences.identifier = f'{identifier}-sentences'
    assert _collect(subject, sentences, ['neural:language_system'], TaskContext(task_type='passive'))[
        'neural:language_system'].shape[0] == 2
    (tmp_path / 'fit').mkdir()
    probabilities = _collect(subject, _pictures(tmp_path, 2), ['behavior:probabilities'], TaskContext(
        response=Choice(['cat', 'dog']),
        fitting_stimuli=_pictures(tmp_path / 'fit', 4, labels=['cat', 'dog', 'cat', 'dog'])))['behavior:probabilities']
    assert probabilities.shape == (2, 2) and np.allclose(probabilities.values.sum(1), 1)
    assert _collect(subject, _pictures(tmp_path, 2), ['behavior:choice'],
                    TaskContext(response=Choice(['cat', 'dog'])))['behavior:choice'].shape[0] == 2


# --- qualification: the real processor and a tiny model of the real class (no weights) --------
#
# Needs the family's settings files in the Hugging Face cache (the command is in README.md); no
# weight file. Marked `qualification`: not part of the offline run of this file.

def _small(model_class, config):
    """The family's real configuration with few layers and a small width; the parts the processor
    and the forward depend on (patch and merge sizes, rope sections, token ids, vocabulary) stay."""
    text, vision = config.text_config, config.vision_config
    for key, value in dict(num_hidden_layers=2, intermediate_size=64, num_attention_heads=2, num_key_value_heads=1).items():
        setattr(text, key, value)
    if model_class.startswith('Qwen3VL'):
        text.hidden_size, text.head_dim = 64, 128                # rope sections sum to head_dim / 2
        vision.depth, vision.hidden_size, vision.num_heads, vision.intermediate_size = 4, 64, 2, 64
        vision.out_hidden_size, vision.deepstack_visual_indexes = 64, [1, 2]
        if 'Moe' in model_class:
            text.num_experts, text.num_experts_per_tok, text.moe_intermediate_size = 4, 2, 32
    elif model_class.startswith('Qwen2_5_VL'):
        text.hidden_size = 256                                   # 2 heads of 128: the rope sections
        vision.depth, vision.hidden_size, vision.num_heads, vision.intermediate_size = 4, 64, 2, 64
        vision.out_hidden_size, vision.fullatt_block_indexes = 256, [3]
    else:                                                        # SmolVLM, Gemma 3, InternVL: SigLIP-like vision parts
        text.hidden_size = 64
        if hasattr(text, 'head_dim'):
            text.head_dim = 32
        vision.num_hidden_layers, vision.hidden_size, vision.intermediate_size, vision.num_attention_heads = 4, 32, 32, 2
    return config


QUALIFY = ['qwen3-vl-2b_UMI-dev', 'qwen3-vl-30b-a3b_UMI-dev', 'qwen2.5-vl-3b_UMI-dev', 'smolvlm2-256m_UMI-dev',
           'gemma-3-4b_UMI-dev', 'internvl3-1b_UMI-dev']    # one model of each family (and the mixture of experts)


@pytest.mark.qualification
@pytest.mark.parametrize('identifier', QUALIFY)
def test_qualify_the_family(identifier, monkeypatch, tmp_path):
    from brainscore import LoadError
    from brainscore_core.compatibility import ExecutionNotFeasibleError
    from brainscore_core.hf_compat import pin_image_processor
    from brainscore_vision.model_helpers.activations import pca
    live = sys.modules['transformers']
    config = study.MODEL_CONFIGS[identifier]
    try:
        settings = live.AutoConfig.from_pretrained(config['hf_id'], local_files_only=True)
    except OSError:
        pytest.skip(f"the settings files of {config['hf_id']} are not in the cache (README: Add a model)")
    try:                    # the needed packages are present, or the clear message appears
        study._refuse(identifier, config)
    except LoadError as error:
        assert 'which is not installed. Next step' in str(error)
        # Go on as a machine with the folder's requirements would: a processor that imports a
        # missing package then fails when it is built.
        monkeypatch.setattr(study, 'REQUIREMENTS', [])
    try:
        processor = pin_image_processor(live.AutoProcessor.from_pretrained(config['hf_id'], local_files_only=True),
                                        config['hf_id'], local_files_only=True)
    except ImportError as error:
        pytest.skip(f'{identifier}: the real processor cannot be built here: {error}')
    network_class = getattr(live, config['model_class'])
    torch.manual_seed(0)
    network = network_class(_small(config['model_class'], settings)).to(getattr(torch, config['dtype'])).eval()
    monkeypatch.setattr(study, '_load', lambda *a: (network, processor))
    corpus = list(_pictures(tmp_path, 4)['image_file_name'])
    monkeypatch.setattr(pca, '_get_imagenet_val', lambda num_images: corpus)
    hook = pca.LayerPCA.hook.__func__
    monkeypatch.setattr(pca.LayerPCA, 'hook', classmethod(lambda cls, extractor, n_components: hook(cls, extractor, 2)))
    subject = model_registry[identifier]()

    # The regions are the stated depths of the real blocks.
    blocks = len(network.get_submodule('.'.join(config['vision'])))
    expected = study.depth_layers([f"{config['vision'][1]}.{i}" for i in range(blocks)], study.DEPTH_TABLE['vision'])
    assert {region: subject.region_layer_map[region] for region in expected} == expected
    # One whole image for each picture, by the real processor.
    from PIL import Image
    wide = str(tmp_path / 'wide.png')
    Image.new('RGB', (96, 48), (200, 40, 40)).save(wide)
    if config['image_processor_kwargs'] is not None:
        assert study._whole_images([wide, wide], processor.image_processor, config['image_processor_kwargs']).shape[0] == 2
    else:
        assert len(processor.image_processor(images=[Image.open(wide)], return_tensors='pt')['image_grid_thw']) == 1
    # Each declared output through participate, exported as finite float32.
    pictures = _pictures(tmp_path, 4)
    fitting = _pictures(tmp_path / 'fit' if (tmp_path / 'fit').mkdir() is None else tmp_path, 4,
                        labels=['cat', 'dog', 'cat', 'dog'])
    sentences = StimulusSet({'stimulus_id': ['t0', 't1'], 'sentence': ['the cat sat', 'the dog sat']})
    sentences.identifier = 'umi-dev-qualify-sentences'
    passive = TaskContext(task_type='passive')
    for channel in sorted(c for c in subject.out_channels if c.startswith('neural:')):
        stimuli = sentences if channel == 'neural:language_system' else pictures
        values = _collect(subject, stimuli, [channel], passive)[channel].values
        assert values.dtype == np.float32 and np.isfinite(values).all() and values.shape[0] == len(stimuli), channel
    choice = Choice(['cat', 'dog'])
    for channel, stimuli, task in [
            ('behavior:choice', pictures, TaskContext(response=choice)),
            ('behavior:text', sentences, TaskContext(response=Text(max_tokens=4))),
            ('behavior:rating', sentences, TaskContext(response=Scale(1, 7, 1))),
            ('behavior:probabilities', pictures, TaskContext(response=choice, fitting_stimuli=fitting)),
            ('behavior:rt', pictures, TaskContext(response=Choice(['cat', 'dog'], reaction_time='surprisal'),
                                                  fitting_stimuli=fitting))]:
        assert _collect(subject, stimuli, [channel], task)[channel].shape[0] == len(stimuli), channel
    continuations = StimulusSet({'stimulus_id': ['t0'], 'sentence': ['the cat'], 'options': [('sat', 'ran')]})
    continuations.identifier = 'umi-dev-qualify-continuations'
    assert np.isfinite(_collect(subject, continuations, ['behavior:log_probabilities'], TaskContext(
        response=Choice(labels='options', verbatim=True)))['behavior:log_probabilities'].values).all()
    from brainscore.harnesses.grid_game import GridGameEnv, play_game
    assert play_game(subject, GridGameEnv(size=5, seed=1, max_steps=2))['actions']
    if config['video']:
        assert _collect(subject, _clip(tmp_path), ['neural:IT'], passive)['neural:IT'].shape[0] == 1
    # A written answer with the language layer, on a picture and text: recorded, or refused before compute.
    trials = StimulusSet({'stimulus_id': ['p0'], 'image_file_name': [pictures['image_file_name'][0]], 'sentence': ['the cat']})
    trials.identifier = 'umi-dev-qualify-mixed'
    mixed = TaskContext(response=Text(max_tokens=2), present=('image', 'text'))
    if config['extra_forward_inputs']:
        with pytest.raises(ExecutionNotFeasibleError):
            _collect(subject, trials, ['behavior:text', 'neural:language_system'], mixed)
    else:
        assert _collect(subject, trials, ['behavior:text', 'neural:language_system'], mixed)['neural:language_system'].shape[0] == 1
