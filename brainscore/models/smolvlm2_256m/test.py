"""smolvlm2-256m with its real weights (a download): run on a GPU machine with ``-m slow``."""
import numpy as np
import pandas as pd
import pytest
from PIL import Image

from brainscore_core.contract import Subject, TaskContext
from brainscore_core.responses import Choice
from brainscore_core.streaming_helpers import stimulus_session
from brainscore_core.supported_data_standards.brainio.stimuli import StimulusSet

pytestmark = pytest.mark.slow


@pytest.fixture(scope='module')
def model():
    from .model import get_model
    return get_model('smolvlm2-256m')


@pytest.fixture
def image_stimuli(tmp_path):
    paths = []
    for i in range(2):
        paths.append(str(tmp_path / f'img{i}.png'))
        Image.fromarray(np.random.RandomState(i).randint(0, 255, (224, 224, 3), dtype=np.uint8)).save(paths[-1])
    stimuli = StimulusSet(pd.DataFrame({'stimulus_id': ['img0', 'img1'], 'image_file_name': paths}))
    stimuli.identifier = 'smolvlm2-256m-test-images'
    return stimuli


def test_declarations(model):
    assert isinstance(model, Subject) and model.identifier == 'smolvlm2-256m'
    assert {'vision:image', 'text'} <= model.in_channels
    assert {'neural:V1', 'neural:IT', 'neural:language_system', 'behavior:choice'} <= model.out_channels


def test_records_the_vision_part(model, image_stimuli):
    session = stimulus_session(image_stimuli, ['neural:V1', 'neural:IT'], TaskContext(task_type='passive'))
    model.participate(session)
    assert all(event.payload.shape[0] == 2 for event in session.emitted)


def test_answers_a_choice(model, image_stimuli):
    session = stimulus_session(image_stimuli, ['behavior:choice'], TaskContext(response=Choice(['cat', 'dog'])))
    model.participate(session)
    assert session.collect('behavior:choice').shape[0] == 2
