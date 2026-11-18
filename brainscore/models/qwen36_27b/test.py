from brainscore import load_model


def test_has_identifier():
    model = load_model('qwen3.6-27b')
    assert model.identifier == 'qwen3.6-27b'
