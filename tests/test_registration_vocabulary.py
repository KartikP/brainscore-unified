"""The two UMI vision registrations wire the same parts through the public slots.

``clip_vit_b_32_unified`` and ``qwen25_vl_3b_unified`` (in brainscore_vision)
moved from the deprecated ``preprocessors=`` / ``activations_model=`` /
``generation_fn=`` constructor names to ``extractor=`` and ``answer=``. These
fixtures rebuild the old constructor call from the same parts and check that the
move changed nothing: the same wrapper answers each input kind, the same layers
are recorded, and assemblies, readouts and generated answers are identical.

Torch, transformers and the wrappers are deterministic stand-ins, so this runs
offline without model weights. It checks constructor wiring only; it is NOT
checkpoint qualification. Real CLIP/Qwen score parity is a separate remote run.
"""
import ast
import contextlib
import importlib.util
import sys
import types
import warnings
import zlib
from importlib.util import find_spec
from pathlib import Path

import numpy as np
import pytest
import xarray as xr
from PIL import Image

from brainscore_core.model_interface import BrainScoreModel, TaskContext
from brainscore_core.supported_data_standards.brainio.assemblies import NeuroidAssembly
from brainscore_core.supported_data_standards.brainio.stimuli import StimulusSet
from brainscore_core.streaming_helpers import behavioral_response, neural_response

pytestmark = pytest.mark.unit

FOLDERS = {'clip-vit-b-32': 'clip_vit_b_32_unified', 'qwen2.5-vl-3b': 'qwen25_vl_3b_unified'}
DEPRECATED = {'preprocessors', 'activations_model', 'generation_fn', 'action_fn', 'state_change_fn'}
PUBLIC_SLOTS = {'extractor', 'answer', 'policy', 'intervention'}
RESPONSES = {4: ' Real ', 5: 'it looks pseudo to me', 6: 'unsure'}  # keyed by image width


# --- deterministic stand-ins -------------------------------------------------

class _Part:
    def __init__(self, name):
        self.name = name

    def __repr__(self):
        return f'<{self.name}>'


class FakeExtractor:
    """Wrapper stand-in. Values depend on wrapper identifier, layer and stimulus,
    so an input routed to the wrong wrapper or layer changes the assembly."""
    instances = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.identifier = kwargs['identifier']
        self.calls, self.resets = [], 0
        FakeExtractor.instances.append(self)

    # Same parameters as the real TextWrapper / VLMVisionWrapper __call__
    # (checked below), which is what the old constructor normalization inspects.
    def __call__(self, stimuli, layers, stimuli_identifier=None, **kwargs):
        stimulus_ids = [str(value) for value in stimuli['stimulus_id']]
        self.calls.append((tuple(stimulus_ids), tuple(layers)))
        units = [(layer, unit) for layer in layers for unit in range(3)]
        values = np.array([[zlib.crc32(f'{self.identifier}|{layer}|{sid}|{unit}'.encode()) / 2 ** 32
                            for layer, unit in units] for sid in stimulus_ids], dtype=np.float32)
        coords = {column: ('presentation', list(stimuli[column].values)) for column in stimuli.columns}
        coords['neuroid_id'] = ('neuroid', [f'{self.identifier}.{layer}.{unit}' for layer, unit in units])
        coords['layer'] = ('neuroid', [layer for layer, _ in units])
        return NeuroidAssembly(values, coords=coords, dims=['presentation', 'neuroid'])

    def reset(self):
        self.resets += 1


class FakePytorchWrapper(FakeExtractor):
    pass


class FakeTextWrapper(FakeExtractor):
    pass


class FakeVLMVisionWrapper(FakeExtractor):
    pass


class FakeTokenizer:
    def decode(self, ids, skip_special_tokens):
        assert skip_special_tokens
        return RESPONSES[int(ids[-1])]


class FakeInputs(dict):
    def to(self, device):
        self['device'] = device
        return self


class FakeProcessor:
    def __init__(self):
        self.tokenizer = FakeTokenizer()
        self.image_processor = 'default'
        self.images_seen, self.prompts = [], []

    def __call__(self, images, return_tensors, text=None, padding=None):
        assert return_tensors == 'pt'
        self.images_seen.extend(images)
        if text is None:   # CLIP image preprocessing
            pixels = np.stack([np.asarray(image, dtype=np.float32) for image in images])
            return {'pixel_values': types.SimpleNamespace(numpy=lambda: pixels)}
        assert padding is True and len(images) == 1
        return FakeInputs(input_ids=np.array([[101, 102, images[0].size[0]]]))

    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        assert tokenize is False and add_generation_prompt is True
        self.prompts.append(messages[0]['content'][1]['text'])
        return 'PROMPT ' + self.prompts[-1]


class FakeCLIP:
    def __init__(self):
        self.vision_model, self.text_model = _Part('vision_model'), _Part('text_model')


class FakeQwen:
    def __init__(self):
        self.model = types.SimpleNamespace(visual=_Part('visual'),
                                           language_model=_Part('language_model'))
        self.generate_calls = []

    def parameters(self):
        yield types.SimpleNamespace(device='fake-device')

    def generate(self, **inputs):
        self.generate_calls.append({k: v for k, v in inputs.items() if k != 'input_ids'})
        ids = inputs['input_ids']
        return np.concatenate([ids, ids[:, -1:]], axis=1)   # the image width is the answer token


@pytest.fixture
def build(monkeypatch):
    """Build a registration with Torch, transformers and wrappers stubbed.

    The stubs are in sys.modules only while the module loads and get_model()
    runs, so later evaluation (sklearn, scipy) never sees a fake torch.
    """
    loads = []

    def loader(kind, factory):
        def from_pretrained(*args, **kwargs):
            made = factory()
            loads.append((kind, args, kwargs, made))
            return made
        return types.SimpleNamespace(from_pretrained=from_pretrained)

    torch = types.ModuleType('torch')
    torch.float16 = _Part('torch.float16')
    torch.no_grad = contextlib.nullcontext
    transformers = types.ModuleType('transformers')
    transformers.CLIPModel = loader('CLIPModel', FakeCLIP)
    transformers.CLIPProcessor = loader('CLIPProcessor', FakeProcessor)
    transformers.Qwen2_5_VLForConditionalGeneration = loader('Qwen2_5_VL', FakeQwen)
    transformers.AutoProcessor = loader('AutoProcessor', FakeProcessor)
    transformers.AutoImageProcessor = loader('AutoImageProcessor', lambda: 'pinned-image-processor')
    modules = {'torch': torch, 'transformers': transformers}
    for name, attribute, fake in (
            ('brainscore_vision.model_helpers.activations.pytorch', 'PytorchWrapper', FakePytorchWrapper),
            ('brainscore.model_helpers.text_wrapper', 'TextWrapper', FakeTextWrapper),
            ('brainscore.model_helpers.vlm_vision_wrapper', 'VLMVisionWrapper', FakeVLMVisionWrapper)):
        modules[name] = types.ModuleType(name)
        setattr(modules[name], attribute, fake)
    monkeypatch.setattr(FakeExtractor, 'instances', [])

    def build_registration(identifier):
        path = _registration_path(FOLDERS[identifier])
        spec = importlib.util.spec_from_file_location(f'umi_registration_{FOLDERS[identifier]}', path)
        module = importlib.util.module_from_spec(spec)
        with monkeypatch.context() as patch, warnings.catch_warnings(record=True) as caught:
            for name, stub in modules.items():
                patch.setitem(sys.modules, name, stub)
            warnings.simplefilter('always')
            spec.loader.exec_module(module)
            model = module.get_model(identifier)
        deprecations = [str(w.message) for w in caught if issubclass(w.category, DeprecationWarning)]
        assert not deprecations, deprecations
        module.loads = loads
        return module, model
    return build_registration


def _registration_path(folder):
    spec = find_spec('brainscore_vision')  # a top-level lookup does not import the package
    return Path(list(spec.submodule_search_locations)[0]) / 'models' / folder / 'model.py'


def _old_wiring(identifier, module, new):
    """The constructor call as it was before the migration, given the same parts."""
    vision, text = FakeExtractor.instances
    with pytest.warns(DeprecationWarning, match='deprecated'):
        if identifier == 'clip-vit-b-32':
            return BrainScoreModel(
                identifier=identifier,
                model=new._model,
                region_layer_map=module.REGION_LAYER_MAP,
                preprocessors={'vision': vision.kwargs['preprocessing'], 'text': text},
                activations_model=vision,
                visual_degrees=8,
                behavioral_readout_layer='encoder.layers.10',
            )
        return BrainScoreModel(
            identifier=identifier,
            model=new._model,
            region_layer_map=module.REGION_LAYER_MAP,
            preprocessors={'vision': vision, 'text': text},
            visual_degrees=8,
            generation_fn=new._generation_fn,
            behavioral_readout_layer='blocks.28',
        )


@pytest.fixture(params=sorted(FOLDERS))
def pair(request, build):
    identifier = request.param
    module, new = build(identifier)
    return identifier, module, _old_wiring(identifier, module, new), new


# --- stimuli ------------------------------------------------------------------

def _images(tmp_path, name, widths, labels=None):
    rows, paths = [], {}
    for index, width in enumerate(widths):
        stimulus_id = f'{name}{index}'
        path = tmp_path / f'{stimulus_id}.png'
        Image.fromarray(np.full((4, width, 4), index * 40, dtype=np.uint8), mode='RGBA').save(path)
        row = {'stimulus_id': stimulus_id, 'image_file_name': str(path),
               'object_name': ['dog', 'cat'][index % 2]}
        if labels is not None:
            row['image_label'] = labels[index]
        rows.append(row)
        paths[stimulus_id] = str(path)
    stimuli = StimulusSet(rows)
    stimuli.identifier, stimuli.stimulus_paths = f'registration-fixture-{name}', paths
    return stimuli


def _sentences():
    stimuli = StimulusSet({'stimulus_id': ['s0', 's1', 's2'],
                           'sentence': ['the cat sat', 'a dog ran', 'neurons fire']})
    stimuli.identifier = 'registration-fixture-sentences'
    return stimuli


def _assert_same(old, new):
    assert type(old) is type(new)
    xr.testing.assert_identical(old, new)
    assert old.dtype == new.dtype
    assert list(old.coords) == list(new.coords)
    for name in old.coords:
        assert old[name].dtype == new[name].dtype, name
        np.testing.assert_array_equal(old[name].values, new[name].values)


def _call_logs():
    logs = [list(part.calls) for part in FakeExtractor.instances]
    for part in FakeExtractor.instances:
        part.calls.clear()
    return logs


# --- wiring ---------------------------------------------------------------------

@pytest.mark.parametrize('identifier', sorted(FOLDERS))
def test_factories_emit_no_deprecated_constructor_warning(build, identifier):
    _, model = build(identifier)   # build() fails on any DeprecationWarning
    assert isinstance(model, BrainScoreModel)


def test_registrations_pass_only_public_slot_names():
    for folder in FOLDERS.values():
        tree = ast.parse(_registration_path(folder).read_text(encoding='utf-8'))
        calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
                 and getattr(node.func, 'id', None) == 'BrainScoreModel']
        assert len(calls) == 1, folder
        keywords = {keyword.arg for keyword in calls[0].keywords}
        assert not keywords & DEPRECATED, folder
        assert 'extractor' in keywords, folder


def test_old_and_new_constructors_normalize_identically(pair):
    identifier, module, old, new = pair
    vision, text = FakeExtractor.instances
    assert [(key, extractor) for key, extractor in new._extractors.items()] == \
        [('vision', vision), ('text', text)]
    assert [(key, id(value)) for key, value in old._extractors.items()] == \
        [(key, id(value)) for key, value in new._extractors.items()]
    assert old._layer_extractors == new._layer_extractors == {'vision', 'text'}
    # Raw layer strings, no modality routing added: same as before the migration.
    assert dict(old.region_layer_map) == dict(new.region_layer_map) == module.REGION_LAYER_MAP
    assert all(isinstance(value, str) for value in new.region_layer_map.values())
    assert old._region_modality_map == new._region_modality_map == {}
    assert list(old.capability_config.items()) == list(new.capability_config.items())
    assert old._generation_fn is new._generation_fn
    assert (old._action_fn, old._state_change_fn) == (new._action_fn, new._state_change_fn) == (None, None)
    for name in ('identifier', 'backbone_id', 'supported_modalities', 'required_modalities',
                 'in_channels', 'out_channels', 'required_channels'):
        assert getattr(old, name) == getattr(new, name), name
    assert old.visual_degrees() == new.visual_degrees() == 8
    assert old._model is new._model


CASES = {
    'clip-vit-b-32': [('V1', 'images'), ('V2', 'images'), ('V4', 'images'), ('IT', 'images'),
                      ('language_system', 'sentences'), (['V4', 'IT'], 'images'),
                      # Existing limitation, preserved: without modality routing a text
                      # stimulus recorded at 'IT' reads the text tower's encoder.layers.10.
                      ('IT', 'sentences')],
    'qwen2.5-vl-3b': [('V1', 'images'), ('V2', 'images'), ('V4', 'images'), ('IT', 'images'),
                      ('language_system', 'sentences'), (['V4', 'IT'], 'images'),
                      ('IT', 'sentences')],
}


def test_neural_assemblies_are_identical(pair, tmp_path):
    identifier, module, old, new = pair
    images, sentences = _images(tmp_path, 'image', [4, 5, 6]), _sentences()
    vision_wrapper = identifier if identifier == 'clip-vit-b-32' else f'{identifier}-vision'
    expected_wrapper = {'images': vision_wrapper, 'sentences': f'{identifier}-text'}
    for record, kind in CASES[identifier]:
        stimuli = images if kind == 'images' else sentences
        outputs = []
        for subject in (old, new):
            subject.start_recording(record, time_bins=[(70, 170)])
            outputs.append((subject.process(stimuli), _call_logs()))
            if isinstance(record, str):
                outputs.append((neural_response(subject=subject, stimulus_set=stimuli,
                                                record=record, time_bins=[(70, 170)]), _call_logs()))
        half = len(outputs) // 2
        for (old_assembly, old_log), (new_assembly, new_log) in zip(outputs[:half], outputs[half:]):
            _assert_same(old_assembly, new_assembly)
            assert old_log == new_log and any(new_log), (record, kind)
        layers = [module.REGION_LAYER_MAP[r] for r in ([record] if isinstance(record, str) else record)]
        assert set(new_assembly['layer'].values) == set(layers)
        assert all(str(n).startswith(expected_wrapper[kind] + '.') for n in new_assembly['neuroid_id'].values)


def test_readout_behavior_is_identical(pair, tmp_path):
    identifier, module, old, new = pair
    fitting = _images(tmp_path, 'fit', [4, 5, 6, 4, 5, 6], labels=['dog', 'cat'] * 3)
    scoring = _images(tmp_path, 'score', [5, 4, 6])
    results = []
    for subject in (old, new):
        task = TaskContext(task_type='probabilities', fitting_stimuli=fitting,
                           prefer_path='readout', metadata={'stimuli': scoring})
        results.append((behavioral_response(subject, task), _call_logs()))
        subject.start_task(task)
        results.append((subject.process(scoring), _call_logs()))
    for (old_output, old_log), (new_output, new_log) in zip(results[:2], results[2:]):
        _assert_same(old_output, new_output)
        assert old_log == new_log
    readout = {'clip-vit-b-32': 'encoder.layers.10', 'qwen2.5-vl-3b': 'blocks.28'}[identifier]
    assert all(layers == (readout,) for log in results[3][1] for _, layers in log)


def test_qwen_answer_path_is_identical(build, tmp_path):
    module, new = build('qwen2.5-vl-3b')
    old = _old_wiring('qwen2.5-vl-3b', module, new)
    qwen = new._model
    processor = next(made for kind, _, _, made in module.loads if kind == 'AutoProcessor')
    stimuli = _images(tmp_path, 'word', [4, 5, 6])
    task = TaskContext(task_type='label', instruction='Is this a real word?',
                       label_set=['real', 'pseudo'], metadata={'stimuli': stimuli})
    results = []
    for subject in (old, new):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            output = behavioral_response(subject, task)
        results.append((output, [str(w.message) for w in caught], list(qwen.generate_calls),
                        list(processor.prompts)))
        qwen.generate_calls.clear()
        processor.prompts.clear()
    (old_output, old_warnings, old_calls, old_prompts), (new_output, new_warnings, new_calls, new_prompts) = results
    _assert_same(old_output, new_output)
    assert old_warnings == new_warnings and any('unsure' in w for w in new_warnings)
    assert old_calls == new_calls == [{'device': 'fake-device', 'max_new_tokens': 8, 'do_sample': False}] * 3
    assert old_prompts == new_prompts == ['Is this a real word? Answer with exactly one of: real, pseudo.'] * 3
    # exact match -> 'real', word match -> 'pseudo', unparseable -> first label
    np.testing.assert_array_equal(new_output.values, [[1, 0], [0, 1], [1, 0]])


def test_reset_reaches_the_same_wrappers(pair):
    _, _, old, new = pair
    for subject in (old, new):
        subject.reset()
        assert [part.resets for part in FakeExtractor.instances] == [1, 1]
        for part in FakeExtractor.instances:
            part.resets = 0


# --- retained configuration ----------------------------------------------------

def test_clip_keeps_its_wrappers_and_preprocessing(build, tmp_path):
    module, model = build('clip-vit-b-32')
    assert [(kind, args, kwargs) for kind, args, kwargs, _ in module.loads] == [
        ('CLIPProcessor', ('openai/clip-vit-base-patch32',), {}),
        ('AutoImageProcessor', ('openai/clip-vit-base-patch32',), {'use_fast': False}),
        ('CLIPModel', ('openai/clip-vit-base-patch32',), {}),
    ]
    processor, clip = module.loads[0][3], module.loads[2][3]
    assert processor.image_processor == 'pinned-image-processor'
    vision, text = FakeExtractor.instances
    assert type(vision) is FakePytorchWrapper and type(text) is FakeTextWrapper
    preprocessing = vision.kwargs['preprocessing']
    assert vision.kwargs == {'identifier': 'clip-vit-b-32', 'model': clip.vision_model,
                             'preprocessing': preprocessing}
    assert vision.image_size == 224
    assert preprocessing.func is module._load_preprocess_images
    assert preprocessing.keywords == {'processor': processor, 'image_size': 224}
    assert text.kwargs == {'model': clip.text_model, 'tokenizer': processor.tokenizer,
                           'identifier': 'clip-vit-b-32-text',
                           'layer_aggregation': 'mean_tokens', 'max_length': 77}
    path = _images(tmp_path, 'pixels', [4]).stimulus_paths['pixels0']
    assert preprocessing([path]).shape == (1, 4, 4, 3)   # RGBA converted to RGB
    assert processor.images_seen[-1].mode == 'RGB'
    assert model._behavioral_readout_layer == 'encoder.layers.10'
    assert model._generation_fn is None


def test_qwen_keeps_its_wrappers_dtype_and_generation(build):
    module, model = build('qwen2.5-vl-3b')
    torch = module.torch   # the stub bound at import
    assert [(kind, args, kwargs) for kind, args, kwargs, _ in module.loads] == [
        ('Qwen2_5_VL', ('Qwen/Qwen2.5-VL-3B-Instruct',), {'torch_dtype': torch.float16}),
        ('AutoProcessor', ('Qwen/Qwen2.5-VL-3B-Instruct',), {}),
        ('AutoImageProcessor', ('Qwen/Qwen2.5-VL-3B-Instruct',), {'use_fast': False}),
    ]
    qwen, processor = module.loads[0][3], module.loads[1][3]
    assert processor.image_processor == 'pinned-image-processor'
    vision, text = FakeExtractor.instances
    assert type(vision) is FakeVLMVisionWrapper and type(text) is FakeTextWrapper
    patch_count_fn = vision.kwargs['patch_count_fn']
    assert vision.kwargs == {'model': qwen.model.visual, 'processor': processor,
                             'identifier': 'qwen2.5-vl-3b-vision', 'image_input_key': 'pixel_values',
                             'forward_kwargs_map': {'grid_thw': 'image_grid_thw'},
                             'patch_count_fn': patch_count_fn,
                             'layer_aggregation': 'mean_patches', 'batch_size': 4}
    assert patch_count_fn({'image_grid_thw': [(1, 2, 3), (2, 2, 2)]}) == [6, 8]
    assert text.kwargs == {'model': qwen.model.language_model, 'tokenizer': processor.tokenizer,
                           'identifier': 'qwen2.5-vl-3b-text',
                           'layer_aggregation': 'last_token', 'max_length': 512}
    assert model._behavioral_readout_layer == 'blocks.28'
    assert model._generation_fn is not None


def test_stand_ins_match_the_real_wrapper_signatures():
    """The old wiring found extractors in preprocessors by their call signature."""
    root = Path(list(find_spec('brainscore').submodule_search_locations)[0]) / 'model_helpers'
    expected = ['self', 'stimuli', 'layers', 'stimuli_identifier']
    for filename, class_name in (('text_wrapper.py', 'TextWrapper'),
                                 ('vlm_vision_wrapper.py', 'VLMVisionWrapper')):
        tree = ast.parse((root / filename).read_text(encoding='utf-8'))
        cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == class_name)
        call = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == '__call__')
        assert [argument.arg for argument in call.args.args] == expected, class_name
        assert call.args.kwarg is not None, class_name
