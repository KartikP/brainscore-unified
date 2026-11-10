"""A new user's own images and layer names: what works and what fails early."""
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image

from brainscore import BrainScoreModel
from brainscore_core.supported_data_standards.brainio.stimuli import StimulusSet
from brainscore.model_helpers.vision_wrapper import VisionWrapper


class Net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 2, 3)
        self.head = torch.nn.Flatten()

    def forward(self, x):
        return self.head(self.conv(x))


def _preprocess(paths):
    return torch.stack([torch.from_numpy(np.asarray(Image.open(p).convert('RGB'), dtype=np.float32)
                                         ).permute(2, 0, 1) for p in paths])


def _images(tmp_path):
    paths = []
    for i in range(3):
        path = tmp_path / f'{i}.png'
        Image.fromarray(np.full((8, 8, 3), i * 50, dtype=np.uint8)).save(path)
        paths.append(str(path))
    return paths


def test_own_images_need_only_an_image_path_column(tmp_path):
    net = Net().eval()
    wrapper = VisionWrapper(net, _preprocess, identifier='mine', device='cpu')
    model = BrainScoreModel('mine', extractor={'vision': wrapper}, region_layer_map={'IT': 'conv'})
    stimuli = StimulusSet(pd.DataFrame({'stimulus_id': ['a', 'b', 'c'], 'image_path': _images(tmp_path)}))
    stimuli.identifier = 'own-images'
    model.start_recording('IT')
    assert model.process(stimuli).shape == (3, 2 * 6 * 6)
    assert not hasattr(stimuli, 'stimulus_paths')


def test_sub_module_wrapping_validates_relative_layer_paths():
    outer = torch.nn.Module()
    outer.vision_model = Net()
    wrapper = VisionWrapper(outer.vision_model, _preprocess, identifier='sub', device='cpu')
    BrainScoreModel('sub', model=outer, extractor={'vision': wrapper}, region_layer_map={'IT': 'conv'})
    with pytest.raises(ValueError, match=r"'vision_model.conv'.*Close matches: \['conv'\]"):
        BrainScoreModel('sub', model=outer, extractor={'vision': wrapper},
                        region_layer_map={'IT': 'vision_model.conv'})


@pytest.mark.parametrize('module, name', [
    ('brainscore.model_helpers.text_wrapper', 'TextWrapper'),
    ('brainscore.model_helpers.audio_wrapper', 'AudioWrapper'),
    ('brainscore.model_helpers.video_wrapper', 'VideoWrapper'),
    ('brainscore.model_helpers.vlm_vision_wrapper', 'VLMVisionWrapper'),
    ('brainscore.model_helpers.vlm_video_wrapper', 'VLMVideoWrapper'),
])
def test_wrappers_declare_the_wrapped_module_paths(module, name):
    import importlib
    cls = getattr(importlib.import_module(module), name)
    net = torch.nn.Module()
    net.layers = torch.nn.ModuleList([torch.nn.Linear(1, 1)])
    assert cls.layer_names(SimpleNamespace(_model=net)) == ['layers', 'layers.0']
