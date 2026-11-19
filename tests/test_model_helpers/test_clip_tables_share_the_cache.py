"""Which Friends and Lahner2024 clip tables share a video cache entry.

The Friends table name signs every file of the table, so the clips-with-sound
table keeps its own name on purpose: a store that keys on the name alone must
not mistake it for the clips-only table. Each table gets its own video entry.
Other clips, another order or another frame sampling give another entry.
Lahner2024 shows its visual and its two-stream subjects one clip table.
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


def test_the_friends_clips_with_sound_table_keeps_its_own_name_and_entries(store, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    films = _friends(tmp_path)
    video, audio = video_wrapper(), audio_wrapper()
    store.install(video)
    store.install(audio)
    clips, with_sound = _table(films, ['clips'], tmp_path), _table(films, ['clips', 'sound'], tmp_path)
    assert clips.identifier != with_sound.identifier
    video(clips, layers=LAYERS)
    first = _entries(store)
    video(with_sound, layers=LAYERS)
    second = _entries(store)
    assert first < second
    audio(with_sound, layers=['frames', 'head'])
    assert second < _entries(store)


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


def test_lahner_shows_the_visual_and_the_two_stream_subject_one_clip_table(monkeypatch, tmp_path):
    import brainscore
    from tests.lahner_synthetic import Recorder, install
    install(monkeypatch, tmp_path)
    visual = Recorder({'vision:video'})
    brainscore.load_benchmark('Lahner2024-fMRI-naturalistic')(visual)
    both = Recorder({'vision:video', 'audio'}, regions=('IT', 'A1'))
    brainscore.load_benchmark('Lahner2024-fMRI-naturalistic-multimodal')(both)
    sessions = [(ids, name) for inputs, _, ids, name in visual.seen + both.seen if inputs == ['vision:video']]
    assert len(sessions) == 2 and sessions[0] == sessions[1]
