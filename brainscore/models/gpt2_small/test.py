from brainscore import load_model


def test_has_identifier():
    model = load_model('gpt2-small')
    assert model.identifier == 'gpt2-small'
