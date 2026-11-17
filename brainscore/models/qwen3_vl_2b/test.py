"""qwen3-vl-2b with its real weights (a download): run on a GPU machine with ``-m slow``.

The pixel-input vision part of a model with ``image_processor_kwargs`` reduces each layer with
``LayerPCA(1000)``, which fits on ImageNet validation images the first time (``MT_IMAGENET_PATH``).
"""
import numpy as np
import pandas as pd
import pytest
from PIL import Image

from brainscore_core.contract import Subject, TaskContext
from brainscore_core.responses import Choice, Text
from brainscore_core.streaming_helpers import stimulus_session
from brainscore_core.supported_data_standards.brainio.stimuli import StimulusSet

pytestmark = pytest.mark.slow


@pytest.fixture(scope='module')
def model():
    from .model import get_model
    return get_model('qwen3-vl-2b')


def _images(folder, n, labels=None):
    paths = {}
    for i in range(n):
        paths[f'img{i}'] = str(folder / f'img{i}.png')
        Image.fromarray(np.random.RandomState(i).randint(0, 255, (224, 224, 3), dtype=np.uint8)).save(paths[f'img{i}'])
    table = {'stimulus_id': list(paths), 'image_file_name': list(paths.values())}
    if labels:
        table['image_label'] = labels
    stimuli = StimulusSet(pd.DataFrame(table))
    stimuli.identifier = f'qwen3-vl-2b-test-images-{n}-{bool(labels)}'
    stimuli.stimulus_paths = paths     # the legacy image extractor reads the paths from here
    return stimuli


def _collect(model, stimuli, outputs, task):
    session = stimulus_session(stimuli, outputs, task)
    model.participate(session)
    return {channel: session.collect(channel) for channel in outputs}


def test_declarations(model):
    assert isinstance(model, Subject) and model.identifier == 'qwen3-vl-2b'
    assert {'vision:image', 'vision:video', 'text'} <= model.in_channels
    assert {'neural:V1', 'neural:IT', 'neural:VWFA', 'neural:language_system', 'behavior:choice',
            'behavior:probabilities', 'perturbation'} <= model.out_channels


def test_records_the_vision_part_from_pictures(model, tmp_path):
    answers = _collect(model, _images(tmp_path, 2), ['neural:V1', 'neural:IT'], TaskContext(task_type='passive'))
    for channel, assembly in answers.items():
        assert assembly.shape[0] == 2 and np.isfinite(assembly.values).all(), channel


def test_records_the_language_model_from_text(model):
    sentences = StimulusSet(pd.DataFrame({'stimulus_id': ['s0', 's1'],
                                          'sentence': ['the cat sat on the mat', 'a dog runs fast']}))
    sentences.identifier = 'qwen3-vl-2b-test-sentences'
    assembly = _collect(model, sentences, ['neural:language_system'],
                        TaskContext(task_type='passive'))['neural:language_system']
    assert assembly.shape[0] == 2 and np.isfinite(assembly.values).all()


def test_answers_a_choice_and_a_written_reply(model, tmp_path):
    pictures = _images(tmp_path, 2)
    choices = _collect(model, pictures, ['behavior:choice'], TaskContext(response=Choice(['cat', 'dog'])))
    assert choices['behavior:choice'].shape[0] == 2
    texts = _collect(model, pictures, ['behavior:text'], TaskContext(response=Text(max_tokens=8)))
    assert all(isinstance(text, str) for text in texts['behavior:text'].values)


def test_the_readout_gives_probabilities(model, tmp_path):
    (tmp_path / 'fit').mkdir()
    fitting = _images(tmp_path / 'fit', 4, labels=['cat', 'dog', 'cat', 'dog'])
    answer = _collect(model, _images(tmp_path, 2), ['behavior:probabilities'],
                      TaskContext(response=Choice(['cat', 'dog']), fitting_stimuli=fitting))['behavior:probabilities']
    assert answer.shape == (2, 2) and np.allclose(answer.values.sum(1), 1)
    assert answer.attrs['response']['method'] == 'readout'
