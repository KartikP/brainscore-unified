"""The explicit activation cache key of the unified wrappers.

The key is built from the kind (``<module>.<Class>``), the module's ``_VERSION``,
the settings (``CACHE_FIELDS``), the model state, the input device and the
inputs. The identifier and the source text are not part of it.
"""
import inspect
import sys

import numpy as np
import pytest
import torch

from brainscore_core.extraction_cache import file_inputs, implementation_config
from brainscore.model_helpers._extractors import cache_signature
from tests.test_wrapper_cache_guard import (
    MODULES, _audio_files, _image_files, _video_files, audio_wrapper, text_wrapper, video_wrapper,
    vlm_vision_wrapper,
)

BUILD = {'text': text_wrapper, 'audio': audio_wrapper, 'video': video_wrapper, 'vlm_vision': vlm_vision_wrapper}
FILES = {'audio': _audio_files, 'video': _video_files, 'vlm_vision': _image_files}
WRAPPERS = sorted(BUILD)


@pytest.fixture(autouse=True)
def _enabled(tmp_path, monkeypatch):
    # The key is computed only when the cache is enabled; no entry is read or written here.
    monkeypatch.setenv('RESULTCACHING_DISABLE', '0')
    monkeypatch.chdir(tmp_path)


def _inputs(wrapper):
    if wrapper == 'text':
        return ['the cat sat', 'dog ran'], ['the cat sat', 'the dog ran']
    names = FILES[wrapper]()
    other = 'other' + names[0][names[0].index('.'):]
    with open(names[0], 'rb') as source, open(other, 'wb') as target:
        target.write(source.read() + b'\0')
    return file_inputs(names), file_inputs([other] + names[1:])


def _key(wrapper, inputs):
    stored = '_from_texts_stored' if type(wrapper).__name__ == 'TextWrapper' else '_from_paths_stored'
    key = cache_signature(wrapper, stored, inputs, 'stimuli')
    assert key is not None
    return key


@pytest.mark.parametrize('name', WRAPPERS)
def test_same_weights_and_settings_under_two_identifiers_give_one_key(name):
    inputs, _ = _inputs(name)
    one, two = BUILD[name](), BUILD[name]()
    two._identifier = 'another-registration'
    assert _key(one, inputs) == _key(two, inputs)


@pytest.mark.parametrize('name', WRAPPERS)
def test_an_edited_docstring_or_comment_keeps_the_key(name, monkeypatch):
    inputs, _ = _inputs(name)
    wrapper = BUILD[name]()
    cls, before = type(wrapper), _key(wrapper, inputs)
    old_form = implementation_config(cls)
    getsource = inspect.getsource
    monkeypatch.setattr(inspect, 'getsource', lambda obj: getsource(obj) + '\n# an edited comment\n')
    monkeypatch.setattr(cls, '__doc__', (cls.__doc__ or '') + ' An edited docstring.')
    assert implementation_config(cls) != old_form       # the old key, built from source, would change
    assert _key(wrapper, inputs) == before
    assert 'implementation' not in wrapper.cache_config()


@pytest.mark.parametrize('name', WRAPPERS)
def test_changed_weights_change_the_key(name):
    inputs, _ = _inputs(name)
    wrapper = BUILD[name]()
    before = _key(wrapper, inputs)
    with torch.no_grad():
        next(wrapper._model.parameters()).add_(0.125)
    assert _key(wrapper, inputs) != before


@pytest.mark.parametrize('name', WRAPPERS)
def test_a_changed_setting_changes_the_key(name):
    inputs, _ = _inputs(name)
    wrapper = BUILD[name]()
    before = _key(wrapper, inputs)
    setting = {'text': '_max_length', 'audio': '_max_duration_sec', 'video': '_num_frames',
               'vlm_vision': '_layer_aggregation'}[name]
    setattr(wrapper, setting, {'_layer_aggregation': 'first_patch'}.get(setting, 3))
    assert _key(wrapper, inputs) != before


@pytest.mark.parametrize('name', WRAPPERS)
def test_another_batch_size_changes_the_key(name):
    inputs, _ = _inputs(name)
    wrapper = BUILD[name]()
    before = _key(wrapper, inputs)
    wrapper._batch_size += 1
    assert _key(wrapper, inputs) != before


@pytest.mark.parametrize('name', WRAPPERS)
def test_another_input_file_changes_the_key(name):
    inputs, other = _inputs(name)
    wrapper = BUILD[name]()
    assert _key(wrapper, inputs) != _key(wrapper, other)


@pytest.mark.parametrize('name', WRAPPERS)
def test_another_input_device_changes_the_key(name):
    inputs, _ = _inputs(name)
    wrapper = BUILD[name]()
    before = _key(wrapper, inputs)
    wrapper._device = torch.device('meta')
    assert _key(wrapper, inputs) != before


@pytest.mark.parametrize('name', WRAPPERS)
def test_a_raised_version_changes_the_key(name, monkeypatch):
    inputs, _ = _inputs(name)
    wrapper = BUILD[name]()
    before = _key(wrapper, inputs)
    module = sys.modules[MODULES[name]]
    monkeypatch.setattr(module, '_VERSION', module._VERSION + 1)
    assert _key(wrapper, inputs) != before


@pytest.mark.parametrize('name', WRAPPERS)
def test_the_key_names_the_kind_not_the_identifier(name):
    wrapper = BUILD[name]()
    config = wrapper.cache_config()
    assert config['kind'] == f'{MODULES[name]}.{type(wrapper).__name__}'
    assert config['version'] == sys.modules[MODULES[name]]._VERSION
    assert wrapper.identifier not in repr(config['settings'])
    assert not hasattr(wrapper, 'backbone_id')
    assert 'backbone_id' not in inspect.signature(type(wrapper)).parameters
