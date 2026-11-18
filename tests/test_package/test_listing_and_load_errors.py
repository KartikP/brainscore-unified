"""Listing every model and benchmark, and readable reasons when one cannot load."""
import urllib.error

import pytest

import brainscore
from brainscore import LoadError, _explained

pytestmark = pytest.mark.unit


def test_listing_covers_every_codebase_without_loading():
    models = brainscore.list_models()
    assert list(models.columns) == ['identifier', 'source']
    assert {'unified', 'vision', 'language'} <= set(models['source'])
    assert 'clip-vit-b-32' in set(models['identifier'])
    assert models['identifier'].is_unique
    assert {'unified', 'vision'} <= set(brainscore.list_benchmarks()['source'])


def _raise_inside(error, files='weights'):
    with pytest.raises(LoadError) as caught, _explained('some-model', 'vision', files):
        raise error
    return str(caught.value)


def test_missing_weights_say_so_and_who_to_tell():
    message = _raise_inside(urllib.error.HTTPError('https://x/w.pth', 404, 'Not Found', {}, None))
    assert 'weights are no longer at the address' in message
    assert 'not your setup' in message and 'github.com/brain-score/vision/issues' in message


def test_missing_weights_are_found_through_wrapping_errors():
    try:
        try:
            raise urllib.error.HTTPError('https://x/w.pth', 404, 'Not Found', {}, None)
        except urllib.error.HTTPError as gone:
            raise RuntimeError('could not build model') from gone
    except RuntimeError as wrapped:
        assert 'no longer at the address' in _raise_inside(wrapped)


def test_s3_missing_key_counts_as_gone():
    class ClientError(Exception):
        response = {'Error': {'Code': 'NoSuchKey'}}
    assert 'data are no longer' in _raise_inside(ClientError('missing'), files='data')


def test_missing_package_names_it_and_the_next_step():
    message = _raise_inside(ModuleNotFoundError("No module named 'timm'", name='timm'))
    assert "needs the package 'timm'" in message and 'BS_INSTALL_DEPENDENCIES' in message


def test_dependency_conflict_keeps_the_plugin_explanation():
    from brainscore_core.plugin_management.import_plugin import PluginDependencyError
    assert _raise_inside(PluginDependencyError('conflict explained')) == 'conflict explained'


def test_other_errors_pass_through_unchanged():
    with pytest.raises(ZeroDivisionError), _explained('some-model', 'vision', 'weights'):
        1 / 0
