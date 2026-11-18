from brainscore import load_model


def test_has_identifier():
    model = load_model('videomae-base')
    assert model.identifier == 'videomae-base'
