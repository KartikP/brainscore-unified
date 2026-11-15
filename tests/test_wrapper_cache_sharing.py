"""Two wrappers that differ only in the identifier share one stored entry, and each gets its own labels.

The cache key leaves the identifier out, so the second wrapper reads the entry
the first one wrote. The identifier labels the ``model`` and ``neuroid_id``
coordinates; a warm assembly must equal the cold one of the same wrapper in
every coordinate and value. The store is a temporary folder that the test
removes; no other cache is enabled.
"""
import inspect
import tempfile
from pathlib import Path

import pytest
import result_caching
import xarray

from tests.test_wrapper_cache_guard import (
    _audio_files, _image_files, _stimuli, _video_files, audio_wrapper, digest, text_wrapper, video_wrapper,
    vlm_vision_wrapper,
)

CASES = {
    'text': (text_wrapper, lambda: _stimuli('share-text', sentence=['the cat sat', 'dog ran on the mat']),
             ['enc', 'proj']),
    'text_per_token': (lambda: text_wrapper(layer_aggregation='per_token'),
                       lambda: _stimuli('share-per-token', sentence=['the cat sat', 'dog']), ['enc', 'proj']),
    'audio': (audio_wrapper, lambda: _stimuli('share-audio', audio_path=_audio_files()), ['frames', 'head']),
    'video': (video_wrapper, lambda: _stimuli('share-video', video_path=_video_files()), ['block', 'head']),
    'vlm_vision': (vlm_vision_wrapper, lambda: _stimuli('share-vlm', image_file_name=_image_files()),
                   ['blocks.0', 'blocks.1']),
}


def _storage(method):
    return next(cell.cell_contents for cell in method.__func__.__closure__
                if isinstance(cell.cell_contents, result_caching._Storage))


@pytest.fixture
def store(monkeypatch):
    """A temporary store for the wrapper methods only; ``store.enabled = False`` gives cold runs."""
    folder = tempfile.TemporaryDirectory()
    state = type('Store', (), {'enabled': True, 'folder': Path(folder.name)})()

    def enable(identifier):
        return state.enabled and identifier.startswith(('brainscore.model_helpers.', __name__ + '.'))
    monkeypatch.setattr(result_caching, 'is_enabled', enable)
    monkeypatch.setattr('brainscore_core.extraction_cache.is_enabled', enable)

    def install(wrapper):
        method = getattr(wrapper, '_from_texts_stored', None) or wrapper._from_paths_stored
        monkeypatch.setattr(_storage(method), '_storage_directory', folder.name)
    state.install = install
    yield state
    folder.cleanup()


@pytest.mark.parametrize('name', sorted(CASES))
def test_a_second_identifier_gets_its_own_labels_from_a_warm_cache(name, store, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    build, stimuli, layers = CASES[name]
    stimuli = stimuli()
    first, second = build(), build()
    second._identifier = 'second'
    store.install(first)
    first_cold = first(stimuli, layers=layers)
    entries = sorted(store.folder.rglob('*.pkl'))
    assert len(entries) == 1
    second_warm = second(stimuli, layers=layers)
    assert sorted(store.folder.rglob('*.pkl')) == entries
    store.enabled = False
    second_cold = second(stimuli, layers=layers)
    assert set(second_cold['model'].values) == {'second'}
    assert digest(second_warm) == digest(second_cold)
    xarray.testing.assert_identical(second_warm, second_cold)
    store.enabled = True
    first_warm = first(stimuli, layers=layers)
    assert digest(first_warm) == digest(first_cold)
    xarray.testing.assert_identical(first_warm, first_cold)


def test_a_warm_video_result_keeps_the_requested_layer_order(store, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    stimuli = _stimuli('share-video-order', video_path=_video_files())
    first, second = video_wrapper(), video_wrapper()
    second._identifier = 'second'
    store.install(first)
    first(stimuli, layers=['block'])
    second_warm = second(stimuli, layers=['head', 'block'])
    store.enabled = False
    second_cold = second(stimuli, layers=['head', 'block'])
    assert list(second_cold['layer'].values) == ['head'] * 2 + ['block'] * 4
    xarray.testing.assert_identical(second_warm, second_cold)


class _Unversioned(__import__('brainscore.model_helpers.text_wrapper', fromlist=['TextWrapper']).TextWrapper):
    """A subclass with code of its own and no ``_VERSION`` of its own."""

    def _reduce(self, layer_results, mask, _row=None):
        return super()._reduce(layer_results, mask, _row)


class _Versioned(_Unversioned):
    _VERSION = 1


class _SameModule(_Unversioned.__mro__[1]):
    """A subclass declared, by its module name, in the module of its parent; no ``_VERSION`` of its own."""
    __module__ = 'brainscore.model_helpers.text_wrapper'


def _subclass(cls):
    wrapper = text_wrapper()
    wrapper.__class__ = cls
    return wrapper


@pytest.mark.parametrize('cls', [_Unversioned, _SameModule], ids=['other-module', 'parent-module'])
def test_a_subclass_without_its_own_version_does_not_use_the_cache(cls, store, tmp_path, monkeypatch):
    from brainscore.model_helpers import _extractors
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(_extractors, '_UNVERSIONED', set())
    stimuli = _stimuli('share-subclass', sentence=['the cat sat', 'dog ran'])
    wrapper = _subclass(cls)
    store.install(wrapper)
    calls = []
    extract = wrapper._from_texts
    monkeypatch.setattr(wrapper, '_from_texts', lambda *a, **k: calls.append(1) or extract(*a, **k))
    with pytest.warns(UserWarning, match=rf'{cls.__qualname__} has no _VERSION of its own') as warned:
        first = wrapper(stimuli, layers=['enc'])
        second = wrapper(stimuli, layers=['enc'])
    assert len([w for w in warned if '_VERSION' in str(w.message)]) == 1
    assert 'add _VERSION = 1 to its class body and a guard test' in str(warned[0].message)
    assert len(calls) == 2 and digest(first) == digest(second)
    assert list(store.folder.rglob('*.pkl')) == []


def test_a_subclass_with_its_own_version_is_cached_under_its_own_key(store, tmp_path, monkeypatch):
    import warnings
    from brainscore.model_helpers._extractors import cache_signature
    monkeypatch.chdir(tmp_path)
    stimuli = _stimuli('share-versioned', sentence=['the cat sat', 'dog ran'])
    wrapper, parent = _subclass(_Versioned), text_wrapper()
    store.install(wrapper)
    calls = []
    extract = wrapper._from_texts
    monkeypatch.setattr(wrapper, '_from_texts', lambda *a, **k: calls.append(1) or extract(*a, **k))
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        first = wrapper(stimuli, layers=['enc'])
        second = wrapper(stimuli, layers=['enc'])
    assert len(calls) == 1 and digest(first) == digest(second)
    assert len(list(store.folder.rglob('*.pkl'))) == 1
    texts = list(stimuli['sentence'])
    assert cache_signature(wrapper, '_from_texts_stored', texts, 'x') != \
        cache_signature(parent, '_from_texts_stored', texts, 'x')
    assert wrapper.cache_config()['version'] == 1
