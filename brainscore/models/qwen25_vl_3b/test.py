import pytest

from brainscore import load_model



@pytest.mark.parametrize('identifier', ['qwen2.5-vl-3b', 'qwen2.5-vl-3b-vwfa', 'qwen2.5-vl-7b', 'qwen2.5-vl-7b-vwfa', 'qwen2.5-vl-3b-bf16'])
def test_has_identifier(identifier):
    model = load_model(identifier)
    assert model.identifier == identifier
