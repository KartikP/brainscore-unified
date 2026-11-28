from brainscore import load_model


def test_has_identifier():
    model = load_model('vjepa2-vitl')
    assert model.identifier == 'vjepa2-vitl'


def test_cortex_takes_every_second_of_the_24_encoder_blocks():
    from brainscore.models.vjepa.model import CORTEX_LAYERS, region_layers
    assert region_layers()['cortex'].layer_paths == CORTEX_LAYERS and len(CORTEX_LAYERS) == 12
    assert CORTEX_LAYERS[-1] == 'encoder.layer.23'
