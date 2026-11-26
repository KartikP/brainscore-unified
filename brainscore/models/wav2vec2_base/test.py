import pytest


def test_cortex_takes_the_12_encoder_layers_and_a1_is_kept():
    from brainscore.models.wav2vec2_base.model import CORTEX_LAYERS, region_layers
    regions = region_layers()
    assert regions['cortex'].layer_paths == CORTEX_LAYERS and len(CORTEX_LAYERS) == 12
    assert regions['A1'] == 'encoder.layers.6'


@pytest.mark.slow
def test_it_takes_sound_only_and_answers_the_mario_cortex_benchmark():
    from brainscore import load_benchmark, load_model
    model, benchmark = load_model('wav2vec2-base'), load_benchmark('Paugam2025-fMRI-mario-cortex')
    assert model.in_channels == {'audio'} and 'neural:cortex' in model.out_channels
    assert 'audio' in benchmark.required_input_channels
