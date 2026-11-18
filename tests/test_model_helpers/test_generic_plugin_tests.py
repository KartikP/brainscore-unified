"""The generic model plugin suite, run on small stand-in subjects (``load_model`` is patched)."""
from types import SimpleNamespace

import pytest

from brainscore import BrainScoreModel
from brainscore_core.compatibility import ExecutionNotFeasibleError
# Do not import the ``test_*`` functions by name: pytest would collect them here.
from brainscore.model_helpers import generic_plugin_tests

pytestmark = pytest.mark.unit

LOAD = 'brainscore.model_helpers.generic_plugin_tests.load_model'


def _declared(identifier='dummy', in_channels=('text',), out_channels=('behavior:choice',),
              required_channels=()):
    return SimpleNamespace(identifier=identifier, in_channels=set(in_channels),
                           out_channels=set(out_channels), required_channels=set(required_channels))


class _Extractor:
    """Lists its layers; it must never run."""

    def __init__(self, names):
        self.names, self.identifier = list(names), 'stand-in'

    def layer_names(self):
        return list(self.names)

    def __call__(self, stimuli, layers, **kwargs):
        raise AssertionError('the generic suite must not run a forward pass')


def _model(names=('layer1', 'layer4')):
    extractor = _Extractor(names)
    return BrainScoreModel('dummy', extractor={'vision': extractor}, region_layer_map={'IT': 'layer4'}), extractor


class TestIdentifier:
    def test_registered_identifier_passes(self, mocker):
        mocker.patch(LOAD, return_value=_declared())
        generic_plugin_tests.test_identifier('dummy')

    def test_no_identifier_fails(self, mocker):
        mocker.patch(LOAD, return_value=_declared(identifier=None))
        with pytest.raises(AssertionError):
            generic_plugin_tests.test_identifier('dummy')

    def test_other_identifier_fails(self, mocker):
        mocker.patch(LOAD, return_value=_declared(identifier='other'))
        with pytest.raises(AssertionError):
            generic_plugin_tests.test_identifier('dummy')


class TestChannels:
    def test_catalog_channels_pass(self, mocker):
        mocker.patch(LOAD, return_value=_declared(in_channels=('text', 'vision:image'),
                                                  out_channels=('neural:IT', 'behavior:choice'),
                                                  required_channels=('text',)))
        generic_plugin_tests.test_channels('dummy')

    @pytest.mark.parametrize('declared', [
        dict(in_channels=()),
        dict(out_channels=()),
        dict(in_channels=('no-such-channel',)),
        dict(in_channels=('vision',)),                  # a visual input names its form
        dict(required_channels=('audio',)),             # not an input channel
    ])
    def test_bad_declaration_fails(self, mocker, declared):
        mocker.patch(LOAD, return_value=_declared(**declared))
        with pytest.raises(AssertionError):
            generic_plugin_tests.test_channels('dummy')


class TestEachOutputHasAPart:
    def test_wired_outputs_pass(self, mocker):
        model, _ = _model()
        mocker.patch(LOAD, return_value=model)
        generic_plugin_tests.test_each_output_has_a_part('dummy')

    def test_output_without_a_part_fails(self, mocker, monkeypatch):
        model, _ = _model()

        def refuse(channels):
            raise ExecutionNotFeasibleError('no part', channels=channels)
        monkeypatch.setattr(model, 'check_session_support', refuse)
        mocker.patch(LOAD, return_value=model)
        with pytest.raises(ExecutionNotFeasibleError):
            generic_plugin_tests.test_each_output_has_a_part('dummy')

    def test_other_subject_is_skipped(self, mocker):
        mocker.patch(LOAD, return_value=_declared())
        with pytest.raises(pytest.skip.Exception):
            generic_plugin_tests.test_each_output_has_a_part('dummy')


class TestRegionLayersExist:
    def test_existing_layer_passes(self, mocker):
        model, _ = _model()
        mocker.patch(LOAD, return_value=model)
        generic_plugin_tests.test_region_layers_exist('dummy')

    def test_missing_layer_fails(self, mocker):
        model, extractor = _model()
        extractor.names.remove('layer4')                # the loaded network has no such layer
        mocker.patch(LOAD, return_value=model)
        with pytest.raises(ValueError, match='layer4'):
            generic_plugin_tests.test_region_layers_exist('dummy')
