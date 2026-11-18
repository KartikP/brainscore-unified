from brainscore import load_model


def test_has_identifier():
    model = load_model('random-vit-b-32')
    assert model.identifier == 'random-vit-b-32'
