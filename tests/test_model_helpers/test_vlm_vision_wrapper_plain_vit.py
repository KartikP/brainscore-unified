"""A plain Hugging Face vision model (an image processor, no chat template) through VLMVisionWrapper.

The network is a tiny vision transformer built from a config: no weights are downloaded.
"""
import numpy as np
import pytest
from PIL import Image

from brainscore import BrainScoreModel
from brainscore.model_helpers.vlm_vision_wrapper import VLMVisionWrapper
from brainscore_core.streaming_helpers import stimulus_session
from brainscore_core.supported_data_standards.brainio.stimuli import StimulusSet

pytestmark = pytest.mark.unit


def test_a_plain_vision_transformer_records_one_layer_through_participate(tmp_path):
    from transformers import ViTConfig, ViTImageProcessor, ViTModel
    config = ViTConfig(image_size=32, patch_size=8, hidden_size=16, num_hidden_layers=2,
                       num_attention_heads=2, intermediate_size=32)
    network = ViTModel(config).eval()
    processor = ViTImageProcessor(size={'height': 32, 'width': 32})
    wrapper = VLMVisionWrapper(model=network, processor=processor, identifier='tiny-vit', device='cpu')
    model = BrainScoreModel('tiny-vit', extractor={'vision': wrapper},
                            region_layer_map={'IT': 'encoder.layer.1'})
    ids = ['a', 'b', 'c']
    paths = []
    for value, stimulus_id in enumerate(ids):
        path = tmp_path / f'{stimulus_id}.png'
        Image.fromarray(np.full((32, 32, 3), 60 * value, dtype=np.uint8)).save(path)
        paths.append(str(path))
    stimuli = StimulusSet({'stimulus_id': ids, 'image_path': paths})
    stimuli.stimulus_paths = dict(zip(ids, paths))
    stimuli.identifier = 'tiny-vit-images'
    session = stimulus_session(stimuli, ['neural:IT'])
    model.participate(session)
    assembly = session.collect('neural:IT')
    assert list(assembly['stimulus_id'].values) == ids
    assert assembly.sizes['neuroid'] == config.hidden_size
    assert np.isfinite(assembly.values).all()
