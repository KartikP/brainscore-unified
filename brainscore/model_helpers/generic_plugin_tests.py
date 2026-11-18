"""Checks that run once for every identifier a model plugin registers, after a real load.

Core's plugin runner (``brainscore_core.plugin_management``) finds this file at
``<package>/model_helpers/generic_plugin_tests.py`` and runs it with
``--plugin_directory <plugin folder>``; ``pytest_generate_tests`` then gives each
test the identifiers registered in that folder's ``__init__.py``. A model author
writes no extra file. To run it by hand:

    pytest brainscore/model_helpers/generic_plugin_tests.py --plugin_directory brainscore/models/<plugin>

(``conftest.py`` in this folder adds the ``--plugin_directory`` option.)

The checks hold for every subject and use only the subject contract: the
identifier and the declaration. The subject is loaded once per identifier.

Limit: no declared output is exercised with input; add that when a production
job for unified exists.
"""
from brainscore_core import io_catalog
from brainscore_core.compatibility import validate_channels
# the following import is needed to configure pytest
# noinspection PyUnresolvedReferences
from brainscore_core.plugin_management.generic_plugin_tests_helper import pytest_generate_tests  # noqa: F401
from brainscore import load_model


def test_identifier_and_declaration(identifier: str):
    model = load_model(identifier)
    assert model.identifier == identifier
    in_channels, out_channels = set(model.in_channels), set(model.out_channels)
    assert in_channels, 'the model declares no input channel'
    assert out_channels, 'the model declares no output channel'
    # Core's public check: catalog names, forms, directions and addresses.
    validate_channels(in_channels, io_catalog.INPUT, f"'{identifier}' in_channels")
    validate_channels(out_channels, io_catalog.OUTPUT, f"'{identifier}' out_channels")
    extra = sorted(set(model.required_channels) - in_channels)
    assert not extra, f'required channels that are not input channels: {extra}'
