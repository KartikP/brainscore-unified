from brainscore import load_model


def test_has_identifier():
    model = load_model('vjepa1-vitl')
    assert model.identifier == 'vjepa1-vitl'
