from brainscore import load_model


def test_has_identifier():
    model = load_model('qwen2.5-vl-wav2vec2')
    assert model.identifier == 'qwen2.5-vl-wav2vec2'
