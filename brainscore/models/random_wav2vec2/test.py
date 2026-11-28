from brainscore import load_model


def test_has_identifier():
    model = load_model('random-wav2vec2-base')
    assert model.identifier == 'random-wav2vec2-base'


def test_cortex_takes_the_12_encoder_layers():
    from brainscore.models.random_wav2vec2.model import CORTEX_LAYERS, region_layers
    assert region_layers()['cortex'].layer_paths == CORTEX_LAYERS and len(CORTEX_LAYERS) == 12
