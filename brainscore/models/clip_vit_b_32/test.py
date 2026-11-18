import numpy as np
import os
import tempfile

import pandas as pd
import pytest
from PIL import Image

from brainscore import BrainScoreModel
from brainscore_core.legacy_facade import LegacyBenchmarkFacade
from brainscore_core.contract import Subject
from brainscore_core.supported_data_standards.brainio.stimuli import StimulusSet


@pytest.fixture(scope='module')
def model():
    from .model import get_model
    return get_model('clip-vit-b-32')


_img_counter = 0

def _make_image_stimuli(n=3):
    """Create a fresh image stimulus set with a unique identifier."""
    global _img_counter
    _img_counter += 1
    tmpdir = tempfile.mkdtemp()
    paths = {}
    for i in range(n):
        img = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
        img_path = os.path.join(tmpdir, f'img{i}.png')
        img.save(img_path)
        paths[f'img{i}'] = img_path

    stimuli = StimulusSet(pd.DataFrame({
        'stimulus_id': list(paths.keys()),
        'image_file_name': list(paths.values()),
    }))
    stimuli.identifier = f'test_images_{_img_counter}'
    stimuli.stimulus_paths = paths
    return stimuli, tmpdir


@pytest.fixture
def image_stimuli():
    stimuli, tmpdir = _make_image_stimuli()
    yield stimuli
    import shutil
    shutil.rmtree(tmpdir)


@pytest.fixture
def text_stimuli():
    stimuli = StimulusSet(pd.DataFrame({
        'sentence': ['the quick brown fox', 'a dog runs fast', 'neurons fire together'],
        'stimulus_id': ['s0', 's1', 's2'],
    }))
    stimuli.identifier = 'test_text'
    return stimuli


class TestModelIdentity:
    def test_is_unified_model(self, model):
        assert isinstance(model, Subject)

    def test_is_brainscore_model(self, model):
        assert isinstance(model, BrainScoreModel)

    def test_identifier(self, model):
        assert model.identifier == 'clip-vit-b-32'

    def test_in_channels(self, model):
        assert {'vision:image', 'text'} <= model.in_channels

    def test_region_layer_map(self, model):
        rlm = model.region_layer_map
        assert 'V1' in rlm
        assert 'IT' in rlm
        assert 'language_system' in rlm

    def test_visual_degrees(self, model):
        assert model.visual_degrees() == 8


class TestVisionPath:
    def test_look_at_returns_assembly(self, model, image_stimuli):
        # A legacy vision benchmark reaches the model through the facade.
        facade = LegacyBenchmarkFacade(model, visual_degrees=model.visual_degrees())
        facade.start_recording('IT', time_bins=[(70, 170)])
        result = facade.look_at(image_stimuli)
        assert result.dims == ('presentation', 'neuroid')
        assert result.shape[0] == 3

    def test_different_layers(self, model):
        # Use separate stimulus sets to avoid caching conflicts
        tmpdir = tempfile.mkdtemp()
        try:
            paths = {}
            for i in range(2):
                img = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
                img_path = os.path.join(tmpdir, f'layer_test_{i}.png')
                img.save(img_path)
                paths[f'lt{i}'] = img_path

            stim_v1 = StimulusSet(pd.DataFrame({
                'stimulus_id': list(paths.keys()),
                'image_file_name': list(paths.values()),
            }))
            stim_v1.identifier = 'layer_test_v1'
            stim_v1.stimulus_paths = dict(paths)

            stim_it = StimulusSet(pd.DataFrame({
                'stimulus_id': list(paths.keys()),
                'image_file_name': list(paths.values()),
            }))
            stim_it.identifier = 'layer_test_it'
            stim_it.stimulus_paths = dict(paths)

            facade = LegacyBenchmarkFacade(model, visual_degrees=model.visual_degrees())
            facade.start_recording('V1', time_bins=[(70, 170)])
            v1 = facade.look_at(stim_v1)

            facade.start_recording('IT', time_bins=[(70, 170)])
            it = facade.look_at(stim_it)

            assert v1.shape[0] == it.shape[0] == 2
        finally:
            import shutil
            shutil.rmtree(tmpdir)

    def test_process_detects_vision(self, model):
        stimuli, tmpdir = _make_image_stimuli(2)
        try:
            model.start_recording('IT', time_bins=[(70, 170)])
            result = model.process(stimuli)
            assert result.dims == ('presentation', 'neuroid')
            assert result.shape[0] == 2
        finally:
            import shutil
            shutil.rmtree(tmpdir)


class TestTextPath:
    def test_process_detects_text(self, model, text_stimuli):
        model.start_recording('language_system', time_bins=[(70, 170)])
        result = model.process(text_stimuli)
        assert result.dims == ('presentation', 'neuroid')
        assert result.shape[0] == 3

    def test_digest_text_compat(self, model, text_stimuli):
        # A legacy language benchmark reaches the model through the facade.
        facade = LegacyBenchmarkFacade(model)
        facade.start_neural_recording('language_system', 'fMRI')
        result = facade.digest_text(list(text_stimuli['sentence'].values))
        assert 'neural' in result
        assert result['neural'].shape[0] == 3


class TestModalities:
    def test_vision_and_text_different_features(self, model, text_stimuli):
        stimuli, tmpdir = _make_image_stimuli(3)
        try:
            model.start_recording('IT', time_bins=[(70, 170)])
            vision_result = model.process(stimuli)

            model.start_recording('language_system', time_bins=[(70, 170)])
            text_result = model.process(text_stimuli)

            # Vision and text produce different feature dimensions
            assert vision_result.shape[1] != text_result.shape[1]
        finally:
            import shutil
            shutil.rmtree(tmpdir)


# --- the zero-shot registration, with stand-ins (no weights) ---------------------

from pathlib import Path

from brainscore_core.contract import TaskContext
from brainscore_core.responses import Choice


class _StandInExtractor:
    """Stands in for a wrapper: two features from the parity of the stimulus number."""

    def __init__(self, *args, **kwargs):
        self.identifier = kwargs.get('identifier')

    def __call__(self, stimuli, layers, **kwargs):
        from brainscore_core.supported_data_standards.brainio.assemblies import NeuroidAssembly
        ids = list(stimuli['stimulus_id'])
        values = np.array([[2.0, 0.0] if int(i[-1]) % 2 == 0 else [0.0, 2.0] for i in ids])
        return NeuroidAssembly(values, dims=['presentation', 'neuroid'], coords={
            'stimulus_id': ('presentation', ids),
            'neuroid_id': ('neuroid', ['n0', 'n1']), 'layer': ('neuroid', [layers[0]] * 2)})


class _StandInProcessor:
    """A CLIP processor for an 8-pixel network: pixels from the image, token ids from the text."""

    def __call__(self, images, return_tensors):
        import torch
        pixels = [torch.tensor(np.asarray(image.resize((8, 8)), dtype=np.float32) / 255).permute(2, 0, 1)
                  for image in images]
        return {'pixel_values': torch.stack(pixels)}

    def tokenizer(self, texts, padding, return_tensors):
        import torch
        ids = torch.tensor([[3 + (ord(c) % 28) for c in text[:12].ljust(12)] for text in texts])
        return {'input_ids': ids, 'attention_mask': torch.ones_like(ids)}


def _tiny_clip():
    import torch
    from transformers import CLIPConfig
    from transformers import CLIPModel as _Network
    config = CLIPConfig(
        text_config=dict(vocab_size=32, hidden_size=8, intermediate_size=16, num_hidden_layers=1,
                         num_attention_heads=2, max_position_embeddings=16),
        vision_config=dict(image_size=8, patch_size=4, hidden_size=8, intermediate_size=16,
                           num_hidden_layers=1, num_attention_heads=2),
        projection_dim=4)
    torch.manual_seed(0)
    return _Network(config).eval()


@pytest.fixture
def stand_ins(monkeypatch):
    from . import model as family
    network = _tiny_clip()
    monkeypatch.setattr(family, 'load_clip', lambda random_init=False: (network, _StandInProcessor()))
    monkeypatch.setattr(family, 'PytorchWrapper', _StandInExtractor)
    monkeypatch.setattr(family, 'TextWrapper', _StandInExtractor)
    return family


def _pictures(tmp_path, name, n, labels=None):
    rows, paths = [], {}
    for index in range(n):
        path = tmp_path / f'{name}{index}.png'
        Image.fromarray(np.random.RandomState(index).randint(0, 255, (8, 8, 3), dtype=np.uint8)).save(path)
        row = {'stimulus_id': f'{name}{index}', 'image_file_name': str(path)}
        if labels is not None:
            row['image_label'] = labels[index % len(labels)]
        rows.append(row)
        paths[f'{name}{index}'] = str(path)
    stimuli = StimulusSet(pd.DataFrame(rows))
    stimuli.identifier, stimuli.stimulus_paths = f'clip-zero-shot-{name}', paths
    return stimuli


def _choice(subject, stimuli, task, output='behavior:choice'):
    from brainscore_core.streaming_helpers import stimulus_session
    session = stimulus_session(stimuli, [output], task)
    subject.participate(session)
    return session.collect(output)


LABELS = ['dog', 'cat', 'car']


def test_both_identifiers_are_registered():
    from brainscore_core.plugin_management.parse_plugin_changes import get_plugin_ids
    found = get_plugin_ids('models', ['clip_vit_b_32'], str(Path(__file__).parents[2]))
    assert {'clip-vit-b-32', 'clip-vit-b-32-zero-shot'} <= set(found)


def test_the_first_registration_keeps_its_declaration(stand_ins):
    subject = stand_ins.get_model('clip-vit-b-32')
    assert subject.identifier == 'clip-vit-b-32'
    assert subject._answer is None
    assert subject._behavioral.owners['behavior:choice'] == 'readout'
    assert subject._behavioral_readout_layer == 'encoder.layers.10'
    assert 'instruction' not in subject.in_channels
    assert [part.identifier for part in subject.extractor.values()] == ['clip-vit-b-32', 'clip-vit-b-32-text']


def test_the_zero_shot_registration_shares_the_network_and_its_cache_names(stand_ins):
    first, zero_shot = stand_ins.get_model('clip-vit-b-32'), stand_ins.get_model('clip-vit-b-32-zero-shot')
    assert zero_shot.identifier == 'clip-vit-b-32-zero-shot'
    assert ([part.identifier for part in zero_shot.extractor.values()]
            == [part.identifier for part in first.extractor.values()])
    assert {r: s.layer_path for r, s in zero_shot.region_layer_selectors.items()} == stand_ins.REGION_LAYER_MAP
    assert zero_shot._behavioral.owners['behavior:choice'] == 'generation'
    assert {'neural:IT', 'behavior:choice'} <= zero_shot.out_channels


def test_zero_shot_answers_a_choice_without_fitting_stimuli(stand_ins, tmp_path):
    subject = stand_ins.get_model('clip-vit-b-32-zero-shot')
    choice = _choice(subject, _pictures(tmp_path, 'zs', 4), TaskContext(response=Choice(LABELS)))
    assert choice.shape[0] == 4
    assert list(choice['answer_valid'].values) == [True] * 4
    np.testing.assert_array_equal(np.nansum(choice.values, axis=1), [1] * 4)
    record = choice.attrs['response']
    assert record['method'] == 'generation'


def test_zero_shot_follows_clips_own_image_text_logits(stand_ins, tmp_path):
    import torch
    network, processor = stand_ins.load_clip()
    answer = stand_ins.zero_shot_answer(network, processor)
    for stimulus_id, path in _pictures(tmp_path, 'logits', 3).stimulus_paths.items():
        with Image.open(path) as image:
            pixels = processor(images=[image.convert('RGB')], return_tensors='pt')['pixel_values']
        tokens = processor.tokenizer([stand_ins.ZERO_SHOT_PROMPT.format(label=label) for label in LABELS],
                                     padding=True, return_tensors='pt')
        with torch.no_grad():
            logits = network(pixel_values=pixels, **tokens).logits_per_image[0]
        expected = LABELS[int(logits.argmax())]
        assert answer({'stimulus_id': stimulus_id, 'image_file_name': path}, None, LABELS) == expected


def test_zero_shot_refuses_labels_that_are_not_text(stand_ins, tmp_path):
    network, processor = stand_ins.load_clip()
    answer = stand_ins.zero_shot_answer(network, processor)
    path = next(iter(_pictures(tmp_path, 'bad', 1).stimulus_paths.values()))
    with pytest.raises(ValueError, match='labels as text'):
        answer({'stimulus_id': 'bad0', 'image_file_name': path}, None, [1, 2])


def test_the_readout_answers_when_the_task_gives_fitting_stimuli(stand_ins, tmp_path):
    subject = stand_ins.get_model('clip-vit-b-32')
    task = TaskContext(response=Choice(['dog', 'cat']),
                       fitting_stimuli=_pictures(tmp_path, 'fit', 6, labels=['dog', 'cat']))
    choice = _choice(subject, _pictures(tmp_path, 'score', 3), task)
    np.testing.assert_array_equal(choice.values, [[1, 0], [0, 1], [1, 0]])
    assert choice.attrs['response']['method'] == 'readout'


def test_a_format_the_registrations_cannot_give_stays_refused(stand_ins, tmp_path):
    from brainscore_core.compatibility import ExecutionNotFeasibleError
    from brainscore_core.responses import Scale
    stimuli = _pictures(tmp_path, 'refused', 2)
    with pytest.raises(ExecutionNotFeasibleError, match='cannot give a Choice'):
        _choice(stand_ins.get_model('clip-vit-b-32'), stimuli, TaskContext(response=Choice(LABELS)))
    with pytest.raises(ExecutionNotFeasibleError):
        _choice(stand_ins.get_model('clip-vit-b-32-zero-shot'), stimuli,
                TaskContext(response=Scale(1, 5)), output='behavior:rating')
