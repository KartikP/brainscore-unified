"""No-download coverage for wrapper assembly-builder integration."""

from collections import OrderedDict

import numpy as np


def test_text_wrapper_package_uses_shared_builder():
    from brainscore.model_helpers.text_wrapper import TextWrapper

    wrapper = TextWrapper.__new__(TextWrapper)
    wrapper._identifier = 'text-model'

    assembly = wrapper._package(OrderedDict([
        ('encoder.layers.1', np.arange(6, dtype=np.float32).reshape(2, 3)),
        ('encoder.layers.2', np.arange(4, dtype=np.float32).reshape(2, 2)),
    ]), texts=['a', 'b'])

    assert assembly.dims == ('presentation', 'neuroid')
    assert assembly.shape == (2, 5)
    assert list(assembly['neuroid_id'].values) == [
        'text-model.encoder.layers.1.0',
        'text-model.encoder.layers.1.1',
        'text-model.encoder.layers.1.2',
        'text-model.encoder.layers.2.0',
        'text-model.encoder.layers.2.1',
    ]
    assert list(assembly['layer'].values) == [
        'encoder.layers.1',
        'encoder.layers.1',
        'encoder.layers.1',
        'encoder.layers.2',
        'encoder.layers.2',
    ]


def test_vlm_vision_wrapper_package_uses_shared_builder():
    from brainscore.model_helpers.vlm_vision_wrapper import VLMVisionWrapper

    wrapper = VLMVisionWrapper.__new__(VLMVisionWrapper)
    wrapper._identifier = 'vlm-vision'

    assembly = wrapper._package(OrderedDict([
        ('blocks.1', np.arange(12, dtype=np.float32).reshape(2, 2, 3)),
    ]), paths=['a.png', 'b.png'])

    assert assembly.dims == ('presentation', 'neuroid')
    assert assembly.shape == (2, 6)
    assert list(assembly['neuroid_id'].values) == [
        'vlm-vision.blocks.1.0',
        'vlm-vision.blocks.1.1',
        'vlm-vision.blocks.1.2',
        'vlm-vision.blocks.1.3',
        'vlm-vision.blocks.1.4',
        'vlm-vision.blocks.1.5',
    ]
    assert set(assembly['layer'].values) == {'blocks.1'}
