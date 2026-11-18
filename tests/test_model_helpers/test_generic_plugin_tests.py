"""The generic model plugin suite, run on small stand-in subjects (``load_model`` is patched)."""
from types import SimpleNamespace

import pytest

from brainscore_core.compatibility import CompatibilityError
# Do not import the ``test_*`` functions by name: pytest would collect them here.
from brainscore.model_helpers import generic_plugin_tests

pytestmark = pytest.mark.unit

LOAD = 'brainscore.model_helpers.generic_plugin_tests.load_model'


def _declared(identifier='dummy', in_channels=('text',), out_channels=('behavior:choice',),
              required_channels=()):
    return SimpleNamespace(identifier=identifier, in_channels=set(in_channels),
                           out_channels=set(out_channels), required_channels=set(required_channels))


def test_a_valid_declaration_passes_with_one_load(mocker):
    load = mocker.patch(LOAD, return_value=_declared(in_channels=('text', 'vision:image'),
                                                     out_channels=('neural:IT', 'behavior:choice'),
                                                     required_channels=('text',)))
    generic_plugin_tests.test_identifier_and_declaration('dummy')
    load.assert_called_once_with('dummy')


@pytest.mark.parametrize('declared', [
    dict(identifier=None),
    dict(identifier='other'),
    dict(in_channels=()),
    dict(out_channels=()),
    dict(in_channels=('no-such-channel',)),
    dict(in_channels=('vision',)),                  # a visual input names its form
    dict(in_channels=('behavior:choice',)),         # reversed direction: an output as input
    dict(in_channels=('text:bogus',)),              # illegal address: text takes none
    dict(required_channels=('audio',)),             # not an input channel
])
def test_a_bad_identifier_or_declaration_fails(mocker, declared):
    mocker.patch(LOAD, return_value=_declared(**declared))
    with pytest.raises((AssertionError, ValueError, CompatibilityError)):
        generic_plugin_tests.test_identifier_and_declaration('dummy')
