"""Benchmark variants that show the same clips give the video extractor one cache entry.

The Friends video variant shows the clips, the video-with-sound variant shows the
same clips and their sound. The video extractor reads only the clips, so the two
tables must reach one stored entry; the audio extractor reads the sound and keeps
its own entry. Other clips, another order or another frame sampling give another
entry.
The store is a temporary folder; no other cache is enabled.
"""
import shutil

import numpy as np
import pandas as pd

from brainscore.data.algonauts2025 import MEDIA, per_scan_stimuli
from tests.test_model_helpers.test_wrapper_cache_guard import _video_files, audio_wrapper, video_wrapper
from tests.test_model_helpers.test_wrapper_cache_sharing import store  # noqa: F401 (fixture)

LAYERS = ['block', 'head']


def _friends(root, movies=('m1', 'm2')):
    """Two films of two scans each: a clip and a sound file for every scan."""
    clips = _video_files()
    rng = np.random.default_rng(0)
    for movie in movies:
        for kind in ('clips', 'sound'):
            folder, suffix, first, _ = MEDIA[kind]
            (root / folder / movie).mkdir(parents=True, exist_ok=True)
            for t in range(2):
                path = root / folder / movie / f'{t + first:04d}{suffix}'
                if kind == 'clips':
                    shutil.copy(clips[t], path)
                else:
                    with open(path, 'wb') as handle:       # the test audio loader reads .npy bytes
                        np.save(handle, rng.integers(-3, 4, 32).astype(np.float32))
    return pd.DataFrame({'stimulus_id': list(movies), 'video_path': [f'{m}.mkv' for m in movies]})


def _two_frames(path, num_frames, target_fps=None):
    from brainscore.model_helpers.video_wrapper import default_uniform_frame_sampler
    return default_uniform_frame_sampler(path, num_frames=2)


def _table(films, media, root, scans=2):
    return per_scan_stimuli(films, {m: scans for m in films['stimulus_id']}, media, 'friends', 1, root)


def _entries(store):
    return set(store.folder.rglob('*.pkl'))


def test_the_friends_video_variants_share_one_video_entry_and_the_sound_keeps_its_own(
        store, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    films = _friends(tmp_path)
    video, audio = video_wrapper(), audio_wrapper()
    store.install(video)
    store.install(audio)
    video(_table(films, ['clips'], tmp_path), layers=LAYERS)
    first = _entries(store)
    assert len(first) == 1
    with_sound = _table(films, ['clips', 'sound'], tmp_path)
    video(with_sound, layers=LAYERS)
    assert _entries(store) == first
    audio(with_sound, layers=['frames', 'head'])
    assert len(_entries(store)) == 2


def test_other_clips_another_order_or_another_sampling_get_another_video_entry(
        store, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    films = _friends(tmp_path)
    video = video_wrapper()
    store.install(video)
    video(_table(films, ['clips'], tmp_path), layers=LAYERS)
    seen = _entries(store)
    for wrapper, table in (
            (video, _table(films, ['clips'], tmp_path, scans=1)),             # other clips
            (video, _table(films.iloc[::-1], ['clips'], tmp_path)),           # another order
            (video_wrapper(frame_sampler=_two_frames), _table(films, ['clips'], tmp_path))):  # another sampling
        store.install(wrapper)
        wrapper(table, layers=LAYERS)
        assert not _entries(store) <= seen
        seen = _entries(store)

