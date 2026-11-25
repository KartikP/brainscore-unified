from brainscore import load_model


def test_has_identifier():
    model = load_model('videomae-base')
    assert model.identifier == 'videomae-base'


def test_cortex_takes_every_encoder_block():
    from brainscore.models.videomae_base.model import CORTEX_LAYERS, region_layers
    assert region_layers()['cortex'].layer_paths == CORTEX_LAYERS
