"""Checks that run once for every identifier a model plugin registers, after a real load.

Core's plugin runner (``brainscore_core.plugin_management``) finds this file at
``<package>/model_helpers/generic_plugin_tests.py`` and runs it with
``--plugin_directory <plugin folder>``; ``pytest_generate_tests`` then gives each
test the identifiers registered in that folder's ``__init__.py``. A model author
writes no extra file. To run it by hand:

    pytest brainscore/model_helpers/generic_plugin_tests.py --plugin_directory brainscore/models/<plugin>

(``conftest.py`` in this folder adds the ``--plugin_directory`` option.)

No check runs a forward pass. The session plan and the region check use the
public parts of ``BrainScoreModel``; a plugin that registers another kind of
subject skips them.
"""
import pytest

from brainscore_core.io_catalog import form_problem, has
# the following import is needed to configure pytest
# noinspection PyUnresolvedReferences
from brainscore_core.plugin_management.generic_plugin_tests_helper import pytest_generate_tests  # noqa: F401
from brainscore import BrainScoreModel, load_model


def test_identifier(identifier: str):
    model = load_model(identifier)
    assert model.identifier is not None
    assert model.identifier == identifier


def test_channels(identifier: str):
    model = load_model(identifier)
    in_channels, out_channels = set(model.in_channels), set(model.out_channels)
    assert in_channels, 'the model declares no input channel'
    assert out_channels, 'the model declares no output channel'
    unknown = sorted(c for c in in_channels | out_channels if not has(c) or form_problem(c))
    assert not unknown, f'channels not in the channel catalog: {unknown}'
    extra = sorted(set(model.required_channels) - in_channels)
    assert not extra, f'required channels that are not input channels: {extra}'


def _brainscore_model(identifier: str) -> BrainScoreModel:
    model = load_model(identifier)
    if not isinstance(model, BrainScoreModel):
        pytest.skip('the subject is not a BrainScoreModel: it has no session plan or region_layer_map')
    return model


def test_each_output_has_a_part(identifier: str):
    """The model's own session plan accepts a request for each declared output, before any input."""
    model = _brainscore_model(identifier)
    for channel in sorted(model.out_channels):
        model.check_session_support({channel})


def test_region_layers_exist(identifier: str):
    """Each region names a layer of the loaded network.

    Assigning the map back runs the model's own region validation against its
    extractors. An extractor that cannot list its layers is not checked.
    """
    model = _brainscore_model(identifier)
    model.region_layer_map = model.region_layer_map
