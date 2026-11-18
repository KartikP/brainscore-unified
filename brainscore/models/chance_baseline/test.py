from brainscore import load_model


def test_has_identifier():
    model = load_model('chance-baseline')
    assert model.identifier == 'chance-baseline'
