from brainscore import load_model


def test_has_identifier():
    model = load_model('vjepa2-vitl')
    assert model.identifier == 'vjepa2-vitl'
