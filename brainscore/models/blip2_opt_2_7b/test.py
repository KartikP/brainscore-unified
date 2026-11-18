from brainscore import load_model


def test_has_identifier():
    model = load_model('blip2-opt-2.7b')
    assert model.identifier == 'blip2-opt-2.7b'
