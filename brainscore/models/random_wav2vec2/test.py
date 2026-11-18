from brainscore import load_model


def test_has_identifier():
    model = load_model('random-wav2vec2-base')
    assert model.identifier == 'random-wav2vec2-base'
