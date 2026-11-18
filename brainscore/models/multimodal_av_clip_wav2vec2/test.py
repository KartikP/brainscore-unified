import pytest

from brainscore import load_model
from ._ids import SUPPORTED_IDENTIFIERS


@pytest.mark.parametrize('identifier', SUPPORTED_IDENTIFIERS)
def test_has_identifier(identifier):
    model = load_model(identifier)
    assert model.identifier == identifier
