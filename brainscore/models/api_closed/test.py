import pytest

from brainscore import load_model
from .model import REGISTRY


@pytest.mark.parametrize('identifier', sorted(REGISTRY))
def test_has_identifier(identifier):
    model = load_model(identifier)
    assert model.identifier == identifier
