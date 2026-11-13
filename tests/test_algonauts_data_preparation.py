"""The prepared stimuli of the Courtois NeuroMod Friends data plugin, on tiny synthetic data.

``data_packaging/prepare_stimuli.py`` cuts the films into one file per scan;
``per_scan_stimuli`` builds the table the benchmark shows. The ffmpeg tests
make a short movie with ffmpeg's ``testsrc`` and ``sine`` and skip when ffmpeg
is absent. No real data, no model.
"""
import hashlib
import inspect
import shutil
import subprocess
import wave
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

import brainscore.data.algonauts2025 as data
from brainscore.data.algonauts2025 import (
    CLIP_FRAMES, TR_SEC, per_scan_stimuli, scans_per_stimulus)
from brainscore.data.algonauts2025.data_packaging import prepare_stimuli as prep
from brainscore_core.supported_data_standards.brainio.stimuli import StimulusSet

needs_ffmpeg = pytest.mark.skipif(
    shutil.which('ffmpeg') is None or shutil.which('ffprobe') is None,
    reason='ffmpeg and ffprobe are needed')

SAMPLES = round(TR_SEC * 16000)
N_FULL = 6                              # whole TRs in the movie
MOVIE_SECONDS = N_FULL * TR_SEC + 0.7   # ends in the middle of a TR


def _films(tmp_path, movies=('m1',)):
    for movie in movies:
        (tmp_path / f'{movie}.mkv').write_bytes(b'')
    return pd.DataFrame({'stimulus_id': list(movies),
                         'video_path': [str(tmp_path / f'{m}.mkv') for m in movies]})


def _place(root, kind, movie, n):
    folder, suffix, first, _ = data.MEDIA[kind]
    (root / folder / movie).mkdir(parents=True, exist_ok=True)
    for t in range(n):
        (root / folder / movie / f'{t + first:04d}{suffix}').write_bytes(b'')


def _samples(path):
    with wave.open(str(path)) as wav:
        return wav.getnframes(), wav.getframerate(), wav.getnchannels()


def _probe(path, entries):
    return subprocess.run(
        ['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-count_frames',
         '-show_entries', entries, '-of', 'csv=p=0:s=,', str(path)],
        capture_output=True, text=True, check=True).stdout.strip()


def _frame_count(path):
    return int(_probe(path, 'stream=nb_read_frames'))


def _duration(path):
    return float(subprocess.run(
        ['ffprobe', '-v', 'error', '-show_entries', 'format=duration', '-of', 'csv=p=0', str(path)],
        capture_output=True, text=True, check=True).stdout.strip())


@pytest.fixture(scope='module')
def movie(tmp_path_factory):
    path = tmp_path_factory.mktemp('movie') / 'movie.mkv'
    # NTSC frame rate on purpose: 1.49 s is not a whole number of source frames.
    subprocess.run(
        ['ffmpeg', '-y', '-loglevel', 'error',
         '-f', 'lavfi', '-i', f'testsrc=duration={MOVIE_SECONDS}:size=320x240:rate=30000/1001',
         '-f', 'lavfi', '-i', f'sine=frequency=440:duration={MOVIE_SECONDS}:sample_rate=44100',
         '-shortest', '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-c:a', 'aac', '-ac', '2',
         str(path)], check=True)
    return path


# ── The table from prepared files ─────────────────────────────────


def test_the_table_identifier_carries_a_signature_of_the_sorted_columns(tmp_path):
    """The activations cache keys on the identifier, not on the columns: a table
    with other columns must miss the cache, and column order alone must not."""
    assert "'|'.join(sorted(map(str, df.columns)))" in inspect.getsource(per_scan_stimuli)
    films = _films(tmp_path)
    _place(tmp_path, 'frames', 'm1', 2)
    _place(tmp_path, 'clips', 'm1', 2)
    frames = per_scan_stimuli(films, {'m1': 2}, ['frames'], 'friends', 1, tmp_path)
    frame_signature = hashlib.md5(
        b'clip_id|frame_id|image_file_name|stimulus_id|t_within_run').hexdigest()[:8]
    assert frames.identifier == f'algonauts2025-friends-sub01-frames-{frame_signature}'
    clips = per_scan_stimuli(films, {'m1': 2}, ['clips'], 'friends', 1, tmp_path)
    clip_signature = hashlib.md5(b'clip_id|frame_id|stimulus_id|t_within_run|video_path').hexdigest()[:8]
    assert clips.identifier == f'algonauts2025-friends-sub01-clips-{clip_signature}'
    assert list(clips['video_path']) == [str(tmp_path / 'clips' / 'm1' / f'{t:04d}.mp4')
                                         for t in range(2)]
    assert list(frames['image_file_name']) == [str(tmp_path / 'frames' / 'm1' / f'{t:04d}.jpg')
                                               for t in (1, 2)]


def test_the_table_has_a_video_and_an_audio_column_per_scan(tmp_path):
    films = _films(tmp_path)
    _place(tmp_path, 'clips', 'm1', 3)
    _place(tmp_path, 'sound', 'm1', 3)
    table = per_scan_stimuli(films, {'m1': 3}, ['clips', 'sound'], 'friends', 1, tmp_path)
    assert isinstance(table, StimulusSet)
    assert list(table.columns) == ['stimulus_id', 'frame_id', 'clip_id', 't_within_run',
                                   'video_path', 'audio_path']
    assert list(table['stimulus_id']) == [f'm1_t{t:04d}' for t in range(3)]
    assert list(table['audio_path']) == [str(tmp_path / 'sound' / 'm1' / f'{t:04d}.wav')
                                         for t in range(3)]
    assert table.identifier.startswith('algonauts2025-friends-sub01-clips-sound-')
    assert table.stimulus_paths == dict(zip(table['stimulus_id'], table['video_path']))


def test_a_missing_prepared_file_names_the_packaging_command(tmp_path):
    films = _films(tmp_path)
    _place(tmp_path, 'clips', 'm1', 2)                 # clips are there, sound is not
    with pytest.raises(FileNotFoundError, match='data_packaging.prepare_stimuli --split friends '
                                                '--participant 1 --media clips sound'):
        per_scan_stimuli(films, {'m1': 2}, ['clips', 'sound'], 'friends', 1, tmp_path)


@pytest.mark.parametrize('prepared', ['complete', 'partial', 'absent'])
def test_the_table_needs_the_prepared_files_and_not_the_movie_files(tmp_path, prepared):
    """The prepared files the assembly counts decide the table; the movie files are only
    the packaging script's input."""
    films = _films(tmp_path, ('a', 'b'))
    for movie in ('a', 'b'):
        (tmp_path / f'{movie}.mkv').unlink()
    for movie, n in {'complete': (('a', 2), ('b', 2)), 'partial': (('a', 2),), 'absent': ()}[prepared]:
        _place(tmp_path, 'frames', movie, n)
    if prepared == 'complete':
        table = per_scan_stimuli(films, {'a': 2, 'b': 2}, ['frames'], 'friends', 1, tmp_path)
        assert list(table['clip_id']) == ['a', 'a', 'b', 'b']
    else:
        with pytest.raises(FileNotFoundError, match='prepare_stimuli --split friends'):
            per_scan_stimuli(films, {'a': 2, 'b': 2}, ['frames'], 'friends', 1, tmp_path)


def test_packaging_refuses_an_absent_movie_file_it_must_cut(tmp_path):
    films = _films(tmp_path, ('a',))
    (tmp_path / 'a.mkv').unlink()
    with pytest.raises(FileNotFoundError, match='Movie files to cut are absent'):
        prep.prepare_stimuli(films, {'a': 2}, ['frames'], tmp_path / 'prepared')


def test_the_packaging_command_keeps_a_root_with_spaces(tmp_path):
    import shlex
    root = tmp_path / 'Friends prepared; $HOME'
    root.mkdir()
    with pytest.raises(FileNotFoundError) as refused:
        per_scan_stimuli(_films(tmp_path), {'m1': 2}, ['clips', 'sound'], 'friends', 1, root)
    command = str(refused.value).split('`')[1]
    assert shlex.split(command) == [
        'python', '-m', 'brainscore.data.algonauts2025.data_packaging.prepare_stimuli', '--split', 'friends',
        '--participant', '1', '--media', 'clips', 'sound', '--prepared-root', str(root)]


def test_scans_per_stimulus_counts_the_scans_of_each_film():
    import xarray as xr
    assembly = xr.DataArray([0] * 5, dims='presentation',
                            coords={'stimulus_id': ('presentation', ['a', 'a', 'b', 'b', 'b'])})
    assert scans_per_stimulus(assembly) == {'a': 2, 'b': 3}


# ── The commands and the cutters ──────────────────────────────────


def test_sound_command_cuts_16khz_mono_chunks_of_one_tr():
    cmd = prep._sound_ffmpeg_command('/m.mkv', '/out/%04d.wav', n_TRs=7)
    assert cmd[:6] == ['ffmpeg', '-y', '-loglevel', 'error', '-i', '/m.mkv']
    assert cmd[cmd.index('-t') + 1] == f'{TR_SEC * 7:.4f}'
    assert '-vn' in cmd
    audio_filter = cmd[cmd.index('-af') + 1]
    assert 'aresample=16000' in audio_filter and 'channel_layouts=mono' in audio_filter
    assert f'asetnsamples=n={SAMPLES}:p=0' in audio_filter
    assert cmd[cmd.index('-f') + 1] == 'segment'
    assert cmd[cmd.index('-segment_time') + 1] == f'{TR_SEC}'
    assert cmd[cmd.index('-segment_format') + 1] == 'wav'
    assert cmd[-1] == '/out/%04d.wav'
    assert '-t' not in prep._sound_ffmpeg_command('/m.mkv', '/o/%04d.wav')


def _cut(movie, out_dir, n_TRs):
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    prep._cut_clips_one((str(movie), str(out_dir), 224, n_TRs))
    return out_dir


@needs_ffmpeg
def test_every_clip_has_the_same_frame_count_size_and_duration(movie, tmp_path):
    out = _cut(movie, tmp_path / 'clips', N_FULL)
    clips = sorted(Path(out).glob('*.mp4'))
    assert [c.name for c in clips] == [f'{t:04d}.mp4' for t in range(N_FULL)]
    for clip in clips:
        assert _probe(clip, 'stream=width,height,nb_read_frames') == f'224,224,{CLIP_FRAMES}'
        assert _duration(clip) == pytest.approx(TR_SEC, abs=1e-3)
    assert not (Path(out) / '_cut').exists()


@needs_ffmpeg
def test_a_short_clip_at_the_end_of_the_movie_is_not_kept(movie, tmp_path):
    # Ask for more TRs than the movie holds in whole: the 0.7 s rest is no clip.
    out = _cut(movie, tmp_path / 'clips', N_FULL + 3)
    assert len(list(Path(out).glob('*.mp4'))) == N_FULL


@needs_ffmpeg
def test_clip_t_covers_the_movie_time_t_times_tr(movie, tmp_path):
    """Clip t starts on a key frame and shows the movie from t x 1.49 s.

    testsrc draws the running time in the picture, so the first frame of clip t
    is compared with a frame cut straight from the movie at t x 1.49 s.
    """
    import numpy as np
    from PIL import Image
    out = _cut(movie, tmp_path / 'clips', N_FULL)
    for t in (0, 2, N_FULL - 1):
        first, reference, other = (tmp_path / f'{name}_{t}.png' for name in ('first', 'ref', 'other'))
        subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-i', str(out / f'{t:04d}.mp4'),
                        '-frames:v', '1', '-vf', 'scale=224:224', str(first)], check=True)
        for path, start in ((reference, t), (other, t + 1)):
            subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-ss', f'{start * TR_SEC:.4f}',
                            '-i', str(movie), '-frames:v', '1', '-vf', 'scale=224:224',
                            str(path)], check=True)
        a, b, c = (np.asarray(Image.open(p), dtype=float) for p in (first, reference, other))
        # Closer to the movie at t x TR than to the movie one TR later.
        assert np.abs(a - b).mean() < np.abs(a - c).mean()
    # Key frame at the start of every clip.
    keys = subprocess.run(
        ['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_entries',
         'packet=flags', '-of', 'csv=p=0', str(out / '0003.mp4')],
        capture_output=True, text=True, check=True).stdout.split()
    assert keys[0].startswith('K') and sum(f.startswith('K') for f in keys) == 1


@needs_ffmpeg
def test_every_sound_chunk_has_one_tr_of_16khz_mono_samples(movie, tmp_path):
    out = tmp_path / 'sound'
    out.mkdir()
    prep._cut_sound_one((str(movie), str(out), N_FULL + 3))
    chunks = sorted(out.glob('*.wav'))
    # The 0.7 s rest is no chunk; nothing past the movie is cut.
    assert [c.name for c in chunks] == [f'{t:04d}.wav' for t in range(N_FULL)]
    for chunk in chunks:
        assert _samples(chunk) == (SAMPLES, 16000, 1)
    assert not (out / '_cut').exists()


# ── prepare_stimuli: cut, fill, refuse ────────────────────────────


@needs_ffmpeg
def test_preparation_cuts_clips_and_sound_and_fills_the_trailing_files(movie, tmp_path):
    films = pd.DataFrame({'stimulus_id': ['m1'], 'video_path': [str(movie)]})
    n_TRs = N_FULL + 2                   # the scan goes on for 2 TRs after the movie ends
    prep.prepare_stimuli(films, {'m1': n_TRs}, ['clips', 'sound'], tmp_path, n_workers=2)
    table = per_scan_stimuli(films, {'m1': n_TRs}, ['clips', 'sound'], 'friends', 1, tmp_path)
    assert len(table) == n_TRs and list(table['t_within_run']) == list(range(n_TRs))
    for path in table['video_path']:
        assert _frame_count(path) == CLIP_FRAMES
        assert _duration(path) == pytest.approx(TR_SEC, abs=1e-3)
    for path in table['audio_path']:
        assert _samples(path) == (SAMPLES, 16000, 1)
    for column in ('video_path', 'audio_path'):
        last = Path(table[column].iloc[N_FULL - 1]).read_bytes()
        for t in (N_FULL, N_FULL + 1):
            assert Path(table[column].iloc[t]).read_bytes() == last   # copies of the last file
    # A second run finds every file and cuts nothing.
    with patch.object(prep, '_run_pool', side_effect=AssertionError('cut again')):
        prep.prepare_stimuli(films, {'m1': n_TRs}, ['clips', 'sound'], tmp_path)


@pytest.mark.parametrize('media, message', [(['clips'], 'clips were not cut'),
                                            (['clips', 'sound'], 'sound chunks were not cut')])
def test_a_film_that_could_not_be_cut_stops_the_preparation(tmp_path, media, message):
    films = _films(tmp_path)                 # the film exists, but nothing can cut it
    if media == ['clips', 'sound']:
        _place(tmp_path, 'clips', 'm1', 2)   # clips are there, sound is not
    with patch.object(prep, '_run_pool'), pytest.raises(FileNotFoundError, match=message):
        prep.prepare_stimuli(films, {'m1': 2}, media, tmp_path)


def test_only_the_films_with_missing_files_are_cut(tmp_path):
    films = _films(tmp_path, ('m1', 'm2', 'm3'))
    _place(tmp_path, 'frames', 'm2', 4)
    jobs = []

    def cut(worker, planned, n_workers):
        jobs.append((worker, [Path(job[0]).stem for job in planned]))
        for video, out_dir, *_ in planned:   # stand in for ffmpeg
            for t in range(4):
                (Path(out_dir) / f'{t + 1:04d}.jpg').write_bytes(b'')

    with patch.object(prep, '_run_pool', side_effect=cut):
        prep.prepare_stimuli(films, {'m1': 4, 'm2': 4}, ['frames'], tmp_path)
    assert jobs == [(prep._extract_frames_one, ['m1'])]        # m2 is there, m3 has no scans
    assert not (tmp_path / 'frames' / 'm3').exists()
