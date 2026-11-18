from pathlib import Path

import pytest

from brainscore import load_model
from brainscore_core.plugin_management.parse_plugin_changes import get_plugin_ids

HERE = Path(__file__).parent


@pytest.mark.parametrize('identifier', get_plugin_ids('models', [HERE.name], str(HERE.parents[1])))
def test_has_identifier(identifier):
    model = load_model(identifier)
    assert model.identifier == identifier
