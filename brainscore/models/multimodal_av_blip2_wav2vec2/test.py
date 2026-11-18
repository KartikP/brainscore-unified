from brainscore import load_model


def test_has_identifier():
    model = load_model('blip2-wav2vec2')
    assert model.identifier == 'blip2-wav2vec2'
