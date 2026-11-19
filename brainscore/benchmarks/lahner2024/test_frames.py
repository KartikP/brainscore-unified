"""The Lahner2024 frames: a local asset cut by the packaging script, read by the benchmark.

The extractor is in ``brainscore/data/lahner2024/data_packaging/prepare_frames.py``.
A tiny video written here shows what it writes and what it refuses.
"""
from pathlib import Path

import numpy as np
import pytest

from brainscore.data import local
from brainscore.data.lahner2024 import LAHNER_FRAMES, LAHNER_SOUND, frame_file
from brainscore.data.lahner2024.data_packaging.prepare_frames import extract_frame

pytestmark = pytest.mark.unit

TIMES_MS = (0, 500, 1500, 2500, 3200)   # the last is past the end: OpenCV reads no frame


def _video(folder):
    cv2 = pytest.importorskip('cv2')
    path = folder / 'clip_0001.avi'
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'MJPG'), 10.0, (32, 24))
    rng = np.random.default_rng(0)
    for _ in range(30):                                  # 3 s at 10 frames per second
        writer.write(rng.integers(0, 256, (24, 32, 3), dtype=np.uint8))
    writer.release()
    return path


def test_extractor_writes_one_frame_file_for_each_time(tmp_path):
    video = _video(tmp_path)
    for time_ms in TIMES_MS[:-1]:
        written = Path(extract_frame(video, time_ms, tmp_path / 'new'))
        assert written.name == f'clip_0001_t{time_ms}.png'
        assert written == frame_file(tmp_path / 'new', video, time_ms)
        assert written.stat().st_size > 0


def test_a_frame_past_the_end_fails_as_before(tmp_path):
    video = _video(tmp_path)
    with pytest.raises(IOError, match='failed to read frame'):
        extract_frame(video, TIMES_MS[-1], tmp_path / 'new')


def test_an_existing_frame_is_kept(tmp_path):
    video = _video(tmp_path)
    kept = frame_file(tmp_path / 'frames', video, 500)
    kept.parent.mkdir()
    kept.write_bytes(b'kept')
    assert extract_frame(video, 500, tmp_path / 'frames') == str(kept)
    assert kept.read_bytes() == b'kept'


@pytest.fixture
def stimuli(tmp_path):
    import pandas as pd
    from brainscore_core.supported_data_standards.brainio.stimuli import StimulusSet
    table = StimulusSet(pd.DataFrame({'stimulus_id': ['a', 'b']}))
    table.identifier = 'two-clips'
    table.stimulus_paths = {'a': str(tmp_path / 'a.mp4'), 'b': str(tmp_path / 'b.mp4')}
    return table


def test_the_benchmark_reads_prepared_frames_and_names_the_command(tmp_path, monkeypatch, stimuli):
    from brainscore.benchmarks.lahner2024._util import frame_table
    monkeypatch.setenv('BRAINSCORE_DATA_HOME', str(tmp_path))
    frames = tmp_path / 'lahner2024-frames'
    with pytest.raises(local.LocalDataMissing, match='prepare_frames'):
        frame_table(stimuli, [500, 1500])               # no folder
    frames.mkdir()
    frame_file(frames, 'a.mp4', 500).write_bytes(b'png')
    with pytest.raises(local.LocalDataMissing, match='(?s)3 of 4 files .*prepare_frames --sample-times-ms 500 1500'):
        frame_table(stimuli, [500, 1500])
    for clip in ('a', 'b'):
        for time_ms in (500, 1500):
            frame_file(frames, f'{clip}.mp4', time_ms).write_bytes(b'png')
    table = frame_table(stimuli, [500, 1500])
    assert list(table['clip_id']) == ['a', 'a', 'b', 'b']
    assert set(table.stimulus_paths.values()) == {
        str(frames / f'{clip}_t{t}.png') for clip in 'ab' for t in (500, 1500)}
    frame_file(frames, 'a.mp4', 500).write_bytes(b'another png')     # replaced at the same path
    assert frame_table(stimuli, [500, 1500]).identifier != table.identifier


def test_the_benchmark_reads_prepared_sound_and_names_the_command(tmp_path, monkeypatch, stimuli):
    from brainscore.benchmarks.lahner2024._util import clip_table
    monkeypatch.setenv('BRAINSCORE_DATA_HOME', str(tmp_path))
    sound = tmp_path / 'lahner2024-sound'
    with pytest.raises(local.LocalDataMissing, match='prepare_audio_tracks'):
        clip_table(stimuli, 'sound', sound=True)
    sound.mkdir()
    (sound / 'a.wav').write_bytes(b'wav')
    with pytest.raises(local.LocalDataMissing, match='1 of 2 files'):
        clip_table(stimuli, 'sound', sound=True)
    (sound / 'b.wav').write_bytes(b'wav')
    assert list(clip_table(stimuli, 'sound', sound=True)['audio_path']) == [
        str(sound / 'a.wav'), str(sound / 'b.wav')]


def test_the_default_folders_are_under_the_default_data_home(monkeypatch):
    monkeypatch.delenv('BRAINSCORE_DATA_HOME', raising=False)
    assert LAHNER_FRAMES.resolved() == Path.home() / 'brainscore-data' / 'lahner2024-frames'
    assert LAHNER_SOUND.resolved() == Path.home() / 'brainscore-data' / 'lahner2024-sound'
