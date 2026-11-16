"""Tests are the spec. A tool should: load by identifier, and give the result it promises
on a small subject. Move this file to tests/test_<your_name>.py and adapt it.
"""
import os

os.environ.setdefault('RESULTCACHING_DISABLE', '1')

import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image

from brainscore.tools import list_tools, load_tool, register_tool


def _tool_module():
    """The installed module, or this template beside the test."""
    try:
        from brainscore.tools import your_name as module    # TODO: your module name
    except ImportError:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        import tool as module
    if 'your-tool' not in list_tools():        # the template is not in the tool table
        register_tool('your-tool', module.YourTool)
    return module


_WORKSPACE = tempfile.TemporaryDirectory()      # the test images; removed when the tests end


def tiny_subject_and_stimuli():
    """A two-layer network and four images. Replace with what your tool needs."""
    from brainscore import BrainScoreModel
    from brainscore_core.supported_data_standards.brainio.stimuli import StimulusSet
    from brainscore_vision.model_helpers.activations.pytorch import (
        PytorchWrapper, load_preprocess_images)
    with torch.random.fork_rng():
        torch.manual_seed(0)
        net = torch.nn.Sequential(torch.nn.Conv2d(3, 4, 3), torch.nn.ReLU()).eval()
    subject = BrainScoreModel(
        identifier='tool-test-cnn', model=net, region_layer_map={'IT': '1'},
        extractor={'vision': PytorchWrapper(
            identifier='tool-test-cnn', model=net,
            preprocessing=lambda paths: load_preprocess_images(paths, image_size=16))})
    folder = _WORKSPACE.name
    rows = []
    for i in range(4):
        path = os.path.join(folder, f'{i}.png')
        Image.fromarray(np.random.RandomState(i).randint(0, 256, (16, 16, 3), dtype=np.uint8)).save(path)
        rows.append({'stimulus_id': f's{i}', 'image_path': path})
    stimuli = StimulusSet(pd.DataFrame(rows))
    stimuli.stimulus_paths = {row['stimulus_id']: row['image_path'] for row in rows}
    stimuli.identifier = 'tool-test-images'
    return subject, stimuli


@pytest.fixture(scope='module')
def subject_and_stimuli():
    return tiny_subject_and_stimuli()


def test_loads_by_identifier():
    module = _tool_module()
    assert isinstance(load_tool('your-tool'), module.YourTool)


def test_mean_equals_a_direct_recording(subject_and_stimuli):
    from brainscore_core.streaming_helpers import neural_response
    module = _tool_module()
    subject, stimuli = subject_and_stimuli
    means = module.region_means(subject, stimuli, ['IT'])
    direct = neural_response(subject, stimuli, ['neural:IT'])
    assert means['IT'] == pytest.approx(float(np.asarray(direct.values).mean()))
