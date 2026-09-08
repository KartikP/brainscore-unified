"""Registered model -> registered benchmark, with CPU toy weights and data."""
import numpy as np
import pytest
import torch
from PIL import Image

import brainscore
from brainscore_core.model_interface import StateChange, Selection, Perturbation
from brainscore_core.supported_data_standards.brainio.stimuli import StimulusSet

pytestmark = pytest.mark.integration


class Inputs(dict):
    def to(self, device):
        return Inputs({key: value.to(device) for key, value in self.items()})


class Tokenizer:
    truncation_side = 'left'

    def decode(self, ids, **kwargs):
        return 'real' if int(ids[0]) else 'pseudo'


class Processor:
    tokenizer = Tokenizer()

    def apply_chat_template(self, *args, **kwargs):
        return 'toy prompt'

    def __call__(self, images, **kwargs):
        values = [[float(image.getpixel((0, 0))[0]), 1.] for image in images]
        return Inputs(pixel_values=torch.tensor(values),
                      input_ids=torch.ones((len(images), 1), dtype=torch.long),
                      image_grid_thw=torch.ones((len(images), 3), dtype=torch.long))

    def cache_config(self):
        return {'toy_processor': 1}


class Visual(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(1))
        self.blocks = torch.nn.ModuleList([torch.nn.Identity() for _ in range(29)])

    def forward(self, values, grid_thw=None):
        for block in self.blocks:
            values = block(values)
        return values * self.weight


class Qwen(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = torch.nn.Module()
        self.model.visual = Visual()
        self.model.language_model = torch.nn.Linear(2, 2)
        self.generated_features = []

    def generate(self, pixel_values, input_ids, **kwargs):
        values = self.model.visual(pixel_values)
        self.generated_features.append(values.detach().clone())
        predicted = (values[:, 0] > 100).long().unsqueeze(1)
        return torch.cat([input_ids, predicted], dim=1)


@pytest.fixture
def registered_toy(monkeypatch, tmp_path):
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
    import brainscore.models.qwen25_vl_3b.model as registration
    import brainscore.benchmarks.roar_yeatman2021.benchmark as roar
    monkeypatch.setattr('brainscore.model_helpers._device.select_device', lambda: 'cpu')
    net = Qwen().eval()
    monkeypatch.setattr(Qwen2_5_VLForConditionalGeneration, 'from_pretrained', lambda *a, **k: net)
    monkeypatch.setattr(AutoProcessor, 'from_pretrained', lambda *a, **k: Processor())
    monkeypatch.setattr(registration, 'pin_image_processor', lambda processor, *a, **k: processor)
    paths = {}
    for i, value in enumerate([150, 180, 20, 40]):
        path = tmp_path / f'{i}.png'
        Image.new('RGB', (2, 2), (value, 0, 0)).save(path)
        paths[str(i)] = path
    stimuli = StimulusSet({'stimulus_id': list(paths), 'image_label': ['real', 'real', 'pseudo', 'pseudo'],
                          'word': ['cat', 'dog', 'gat', 'pob'],
                          'image_file_name': [str(path) for path in paths.values()]})
    stimuli.identifier = 'registered-toy'
    stimuli.stimulus_paths = paths
    monkeypatch.setattr(roar, 'load_stimulus_set', lambda identifier: stimuli)
    monkeypatch.setattr(roar, 'load_dataset', lambda identifier: object())
    monkeypatch.setattr(roar, '_human_accuracy_on_stimuli', lambda *a, **k: 1.)
    # Shrink only the data split; keep real generation, localization, ablation,
    # matched random controls, and benchmark scoring orchestration intact.
    monkeypatch.setattr(roar, '_split_train_test', lambda stimuli: (stimuli, stimuli))
    return net, stimuli


def test_registered_model_benchmark_path_and_hook_cleanup(registered_toy):
    net, stimuli = registered_toy
    candidate = brainscore.load_model('qwen2.5-vl-3b-vwfa')
    benchmark = brainscore.load_benchmark('Yeatman2021-induced_dyslexia-image')
    assert candidate.region_layer_map['VWFA'] == 'blocks.28'
    score = benchmark(candidate)
    assert np.isfinite(float(score))
    assert set(('baseline_accuracy', 'lesioned_accuracy', 'random_control_accuracy')).issubset(score.attrs)
    assert len(net.generated_features) == 3 * len(stimuli)
    baseline, lesion, control = np.split(torch.cat(net.generated_features).numpy(), 3)
    assert baseline.any()
    assert not lesion.any() and not control.any()  # toy has fewer than 500 units
    assert not net.model.visual.blocks[28]._forward_hooks
    assert not candidate._perturbations.active_perturbations
    # Root agreement is checked explicitly: recording and generate() use the
    # same module, and a selective perturbation reaches both paths.
    candidate.start_recording('VWFA')
    before = candidate.process(stimuli)
    candidate.process(StateChange('ablation', target=Selection('blocks.28', [0]),
                                  perturbation=Perturbation('zero')))
    after = candidate.process(stimuli)
    np.testing.assert_array_equal(after.values[:, 0], 0)
    np.testing.assert_array_equal(after.values[:, 1], before.values[:, 1])
    candidate.reset()
    candidate.start_recording('VWFA')
    np.testing.assert_array_equal(candidate.process(stimuli), before)


def test_original_registration_is_unchanged(registered_toy):
    candidate = brainscore.load_model('qwen2.5-vl-3b')
    assert 'VWFA' not in candidate.region_layer_map
    assert candidate._state_change_fn is None
