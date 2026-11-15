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

from tests.test_wrapper_cache_guard import (
    _audio_files, _image_files, _stimuli, _video_files, audio_wrapper, digest, text_wrapper, video_wrapper,
    vlm_vision_wrapper,
)

CASES = {
    'text': (text_wrapper, lambda: _stimuli('share-text', sentence=['the cat sat', 'dog ran on the mat']),
             ['enc', 'proj']),
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
        return state.enabled and identifier.startswith('brainscore.model_helpers.')
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
    assert digest(first(stimuli, layers=layers)) == digest(first_cold)
