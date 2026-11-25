from brainscore import load_model


def test_has_identifier():
    model = load_model('random-vit-b-32')
    assert model.identifier == 'random-vit-b-32'


def test_cortex_takes_every_vision_block_as_clip_does():
    from brainscore.models.clip_vit_b_32.model import CORTEX_LAYERS
    model = load_model('random-vit-b-32')
    assert model.region_layer_selectors['cortex'].layer_paths == CORTEX_LAYERS
    assert 'neural:cortex' in model.out_channels
