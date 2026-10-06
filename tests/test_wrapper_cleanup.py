"""Extraction must remove its own hooks, including partial registration."""
from collections import OrderedDict

import numpy as np
from PIL import Image
import pytest
import torch

from brainscore.model_helpers.text_wrapper import TextWrapper
from brainscore.model_helpers.video_wrapper import VideoWrapper
from brainscore.model_helpers.vlm_vision_wrapper import VLMVisionWrapper
from brainscore_vision.model_helpers.activations.pytorch import PytorchWrapper


class FailingModel(torch.nn.Module):
    def __init__(self, error):
        super().__init__()
        self.layer = torch.nn.Linear(2, 2)
        self.error = error

    def forward(self, inputs=None, **kwargs):
        result = self.layer(torch.ones(1, 2))
        if self.error:
            raise self.error('interrupted extraction')
        return result


@pytest.mark.parametrize('kind', ['vision', 'text', 'video', 'vlm'])
@pytest.mark.parametrize('failure', ['forward', 'interrupt', 'registration'])
def test_failed_extraction_removes_only_its_hooks(kind, failure, tmp_path):
    error = KeyboardInterrupt if failure == 'interrupt' else RuntimeError
    model = FailingModel(error)
    retained = model.layer.register_forward_hook(lambda *_: None)
    classes = dict(vision=PytorchWrapper, text=TextWrapper,
                   video=VideoWrapper, vlm=VLMVisionWrapper)
    wrapper = classes[kind].__new__(classes[kind])
    wrapper._model = model
    wrapper._device = torch.device('cpu')
    wrapper._forward_kwargs = {}
    wrapper._input_key = None
    wrapper._max_length = 8
    wrapper._tokenizer = lambda *a, **kw: {'inputs': torch.ones(1, 2)}
    wrapper._preprocessing = lambda frames: torch.ones(1, 2)
    wrapper._processor_call = lambda images: {'inputs': torch.ones(1, 2)}
    wrapper._forward = lambda processed: model(**processed)
    wrapper._patch_count_fn = None
    wrapper._layer_aggregation = 'last_token'
    wrapper._flatten_layer_output = lambda value: value
    wrapper._aggregate_per_image = lambda value, *a, **kw: value

    def get_layer(name):
        if name == 'missing':
            raise RuntimeError('missing layer')
        return model.layer

    def register(layer, name, target_dict):
        return layer.register_forward_hook(
            lambda module, args, output: target_dict.__setitem__(
                name, output.detach().numpy()))

    wrapper.get_layer = wrapper._get_layer = get_layer
    wrapper.register_hook = wrapper._register_hook = register
    path = tmp_path / 'image.png'
    Image.new('RGB', (2, 2)).save(path)
    layers = ['layer', 'missing'] if failure == 'registration' else ['layer']

    def extract(names):
        if kind == 'video':
            return wrapper._run_model([[np.ones((2, 2, 3))]], names)
        inputs = [np.ones(2, dtype=np.float32)] if kind == 'vision' else [str(path)]
        return wrapper.get_activations(inputs, names)

    for _ in range(3):
        with pytest.raises(error):
            extract(layers)
        assert list(model.layer._forward_hooks) == [retained.id]

    model.error = None
    result = extract(['layer'])
    assert isinstance(result, (dict, OrderedDict))
    assert np.isfinite(result['layer']).all()
    assert list(model.layer._forward_hooks) == [retained.id]
    retained.remove()
