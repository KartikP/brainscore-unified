"""The clip cutter of the Algonauts 2025 video form, on a tiny synthetic movie.

The movie is made with ffmpeg's ``testsrc``; no real data, no model. The tests skip
when ffmpeg or ffprobe is absent.
"""
import shutil
import subprocess
from pathlib import Path

import pandas as pd
import pytest

from brainscore.benchmarks.algonauts2025 import benchmark as algo
from brainscore.benchmarks.algonauts2025.benchmark import (
    Algonauts2025Friends, CLIP_FRAMES, TR_SEC)

pytestmark = pytest.mark.skipif(
    shutil.which('ffmpeg') is None or shutil.which('ffprobe') is None,
    reason='ffmpeg and ffprobe are needed')

N_FULL = 6                       # whole TRs in the movie
MOVIE_SECONDS = N_FULL * TR_SEC + 0.7   # ends in the middle of a TR


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
        ['ffmpeg', '-y', '-loglevel', 'error', '-f', 'lavfi', '-i',
         f'testsrc=duration={MOVIE_SECONDS}:size=320x240:rate=30000/1001',
         '-c:v', 'libx264', '-pix_fmt', 'yuv420p', str(path)], check=True)
    return path


def _cut(movie, out_dir, n_TRs):
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    algo._ffmpeg_cut_clips_one((str(movie), str(out_dir), 224, n_TRs))
    return out_dir


def test_every_clip_has_the_same_frame_count_size_and_duration(movie, tmp_path):
    out = _cut(movie, tmp_path / 'clips', N_FULL)
    clips = sorted(Path(out).glob('*.mp4'))
    assert [c.name for c in clips] == [f'{t:04d}.mp4' for t in range(N_FULL)]
    for clip in clips:
        assert _probe(clip, 'stream=width,height,nb_read_frames') == f'224,224,{CLIP_FRAMES}'
        assert _duration(clip) == pytest.approx(TR_SEC, abs=1e-3)
    assert not (Path(out) / '_cut').exists()


def test_a_short_clip_at_the_end_of_the_movie_is_not_kept(movie, tmp_path):
    # Ask for more TRs than the movie holds in whole: the 0.7 s rest is no clip.
    out = _cut(movie, tmp_path / 'clips', N_FULL + 3)
    assert len(list(Path(out).glob('*.mp4'))) == N_FULL


def test_clip_t_covers_the_movie_time_t_times_tr(movie, tmp_path):
    """Clip t starts on a key frame and shows the movie from t x 1.49 s.

    testsrc draws the running time in the picture, so the first frame of clip t
    is compared with a frame cut straight from the movie at t x 1.49 s.
    """
    out = _cut(movie, tmp_path / 'clips', N_FULL)
    import numpy as np
    for t in (0, 2, N_FULL - 1):
        first = tmp_path / f'first_{t}.png'
        reference = tmp_path / f'ref_{t}.png'
        subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-i', str(out / f'{t:04d}.mp4'),
                        '-frames:v', '1', '-vf', 'scale=224:224', str(first)], check=True)
        subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-ss', f'{t * TR_SEC:.4f}',
                        '-i', str(movie), '-frames:v', '1', '-vf', 'scale=224:224',
                        str(reference)], check=True)
        from PIL import Image
        a = np.asarray(Image.open(first), dtype=float)
        b = np.asarray(Image.open(reference), dtype=float)
        other = tmp_path / f'other_{t}.png'
        subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-ss', f'{(t + 1) * TR_SEC:.4f}',
                        '-i', str(movie), '-frames:v', '1', '-vf', 'scale=224:224',
                        str(other)], check=True)
        c = np.asarray(Image.open(other), dtype=float)
        # Closer to the movie at t x TR than to the movie one TR later.
        assert np.abs(a - b).mean() < np.abs(a - c).mean()
    # Key frame at the start of every clip.
    keys = subprocess.run(
        ['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_entries',
         'packet=flags', '-of', 'csv=p=0', str(out / '0003.mp4')],
        capture_output=True, text=True, check=True).stdout.split()
    assert keys[0].startswith('K') and sum(f.startswith('K') for f in keys) == 1


def test_the_benchmark_cuts_the_movie_and_fills_the_trailing_clips(movie, tmp_path):
    from brainscore_core.supported_data_standards.brainio.stimuli import StimulusSet
    benchmark = Algonauts2025Friends(participant=1, form='video', assembly_root=tmp_path)
    benchmark._stimulus_set = pd.DataFrame({'stimulus_id': ['m1'], 'video_path': [str(movie)]})
    n_TRs = N_FULL + 2                      # the assembly has 2 TRs after the movie ends
    benchmark._stim_id_to_n_TRs = lambda: {'m1': n_TRs}
    table = benchmark._expand_to_per_TR_clips(n_workers=2)
    assert isinstance(table, StimulusSet)
    assert len(table) == n_TRs
    assert list(table['t_within_run']) == list(range(n_TRs))
    assert set(table['clip_id']) == {'m1'}
    assert list(table['stimulus_id']) == [f'm1_t{t:04d}' for t in range(n_TRs)]
    for path in table['video_path']:
        assert _frame_count(path) == CLIP_FRAMES
        assert _duration(path) == pytest.approx(TR_SEC, abs=1e-3)
    last_cut = Path(table['video_path'].iloc[N_FULL - 1]).read_bytes()
    for t in (N_FULL, N_FULL + 1):
        assert Path(table['video_path'].iloc[t]).read_bytes() == last_cut   # copies of the last clip
    # A second call finds every clip and cuts nothing.
    benchmark._run_pool = lambda *a, **k: pytest.fail('cut again')
    assert len(benchmark._expand_to_per_TR_clips()) == n_TRs


def test_a_movie_that_could_not_be_cut_stops_the_run(tmp_path):
    """No clip file, no run: the table never points a model at a missing file."""
    import pandas as pd
    from unittest.mock import patch
    from brainscore.benchmarks.algonauts2025.benchmark import Algonauts2025Friends
    benchmark = Algonauts2025Friends(participant=1, form='video', assembly_root=tmp_path)
    (tmp_path / 'm1.mkv').write_bytes(b'')    # the movie exists, but nothing can cut it
    benchmark._stimulus_set = pd.DataFrame({'stimulus_id': ['m1'], 'video_path': [str(tmp_path / 'm1.mkv')]})
    with patch.object(Algonauts2025Friends, '_stim_id_to_n_TRs', return_value={'m1': 2}), \
            patch.object(Algonauts2025Friends, '_run_pool'), \
            patch.object(Algonauts2025Friends, '_fill_trailing_files', return_value=0), \
            pytest.raises(FileNotFoundError, match='were not cut'):
        benchmark._expand_to_per_TR_stimuli()
