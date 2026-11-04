"""Convert Courtois NeuroMod .h5 fMRI files into brainio-compatible
NeuroidAssembly netCDFs.

RUN ON EC2 ONLY. Reads the DataLad-installed Algonauts dataset; writes
~25 GB of per-(subject, split) .nc files plus stimulus metadata CSVs.

Verified .h5 layout (sub-01 inspection 2026-05-06):
    fmri/sub-0X/func/sub-0X_task-friends_*_desc-s123456_bold.h5
    fmri/sub-0X/func/sub-0X_task-movie10_*_bold.h5

Each .h5 contains many datasets keyed by movie segment, e.g.:
    ses-001_task-s01e02a   shape (482, 1000) float32
    ses-001_task-bourne01  shape (..., 1000) float32
    ses-006_task-life01_run-1  shape (..., 1000) float32

Stimulus layout:
    stimuli/movies/friends/s{1..7}/friends_s{SS}e{EE}{split}.mkv
    stimuli/movies/movie10/{bourne,wolf,life,figures}/{movie}{NN}.mkv
    stimuli/movies/ood/{movie}/task-{movie}{N}_video.mkv

Transcripts (.tsv, tab-separated):
    Columns: text_per_tr, words_per_tr, onsets_per_tr, durations_per_tr
    One row per TR (1.49 s). Empty rows for TRs with no spoken words.

Output:
    {output_root}/algonauts2025_friends_sub{NN}.nc      (training, w/ fMRI)
    {output_root}/algonauts2025_friends_s7_sub{NN}.nc   (held-out, NaN fMRI)
    {output_root}/algonauts2025_ood_sub{NN}.nc          (held-out, NaN fMRI)
    {output_root}/algonauts2025_stim_friends.csv
    {output_root}/algonauts2025_stim_friends_s7.csv
    {output_root}/algonauts2025_stim_ood.csv
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


ALGONAUTS_SUBJECTS = (1, 2, 3, 5)
ALGONAUTS_SPLITS = ('friends', 'friends_s7', 'ood')
SCHAEFER_N_PARCELS = 1000
TR_SEC = 1.49


# ── Stimulus path helpers ─────────────────────────────────────────


def _friends_video_path(stimuli_root: Path, season: str, episode: str,
                        split: str) -> Path:
    """Friends video: stimuli/movies/friends/s{N}/friends_s{SS}e{EE}{X}.mkv"""
    season_dir = f's{int(season)}'
    return (stimuli_root / 'movies' / 'friends' / season_dir
            / f'friends_s{season}e{episode}{split}.mkv')


def _friends_transcript_path(stimuli_root: Path, season: str,
                             episode: str, split: str) -> Path:
    season_dir = f's{int(season)}'
    return (stimuli_root / 'transcripts' / 'friends' / season_dir
            / f'friends_s{season}e{episode}{split}.tsv')


def _movie10_video_path(stimuli_root: Path, movie: str,
                        idx: str) -> Path:
    """Movie10 video: stimuli/movies/movie10/{movie}/{movie}{NN}.mkv"""
    return (stimuli_root / 'movies' / 'movie10' / movie
            / f'{movie}{idx}.mkv')


def _movie10_transcript_path(stimuli_root: Path, movie: str,
                             idx: str) -> Path:
    return (stimuli_root / 'transcripts' / 'movie10' / movie
            / f'movie10_{movie}{idx}.tsv')


def _ood_video_path(stimuli_root: Path, movie: str,
                    idx: str) -> Path:
    """OOD video: stimuli/movies/ood/{movie}/task-{movie}{N}_video.mkv"""
    return (stimuli_root / 'movies' / 'ood' / movie
            / f'task-{movie}{idx}_video.mkv')


def _ood_transcript_path(stimuli_root: Path, movie: str,
                         idx: str) -> Optional[Path]:
    """OOD transcript: ood_{movie}{N}.tsv. Chaplin has no transcript."""
    if movie == 'chaplin':
        return None
    return (stimuli_root / 'transcripts' / 'ood' / movie
            / f'ood_{movie}{idx}.tsv')


# ── Dataset-key parsers ───────────────────────────────────────────


_FRIENDS_KEY_RE = re.compile(
    r'^ses-(?P<session>\d+)_task-s(?P<season>\d+)e(?P<episode>\d+)'
    r'(?P<split>[a-z])$'
)
_MOVIE10_KEY_RE = re.compile(
    r'^ses-(?P<session>\d+)_task-(?P<movie>[a-z]+)(?P<idx>\d+)'
    r'(?:_run-(?P<run>\d+))?$'
)


def _parse_friends_key(key: str) -> Optional[dict]:
    m = _FRIENDS_KEY_RE.match(key)
    if not m:
        return None
    g = m.groupdict()
    return {
        'session': g['session'],
        'movie': 'friends',
        'season': g['season'],
        'episode': g['episode'],
        'split': g['split'],
        'run': '1',
    }


def _parse_movie10_key(key: str) -> Optional[dict]:
    m = _MOVIE10_KEY_RE.match(key)
    if not m:
        return None
    g = m.groupdict()
    return {
        'session': g['session'],
        'movie': g['movie'],
        'season': '',
        'episode': '',
        'split': g['idx'],   # 01, 02, … treated as split number
        'run': g['run'] or '1',
    }


# ── Stimulus_set builders ─────────────────────────────────────────


def _build_friends_stim_set(stimuli_root: Path) -> pd.DataFrame:
    """Walk stimuli/movies/friends/s{1..6} for training rows."""
    rows = []
    friends_dir = stimuli_root / 'movies' / 'friends'
    for season_dir in sorted(friends_dir.glob('s[1-6]')):
        for video_path in sorted(season_dir.glob('friends_s*.mkv')):
            stem = video_path.stem  # friends_s01e01a
            m = re.match(r'friends_s(\d+)e(\d+)([a-z])$', stem)
            if not m:
                continue
            season, episode, split = m.groups()
            tsv = _friends_transcript_path(
                stimuli_root, season, episode, split)
            rows.append({
                'stimulus_id': f'friends_s{season}e{episode}{split}',
                'movie': 'friends',
                'season': season,
                'episode': episode,
                'split': split,
                'run': '1',
                'video_path': str(video_path),
                'transcript_path': str(tsv) if tsv.exists() else '',
            })
    return pd.DataFrame(rows)


def _build_movie10_stim_set(stimuli_root: Path) -> pd.DataFrame:
    rows = []
    m10_dir = stimuli_root / 'movies' / 'movie10'
    for movie_dir in sorted(m10_dir.iterdir()):
        if not movie_dir.is_dir():
            continue
        movie = movie_dir.name
        for video_path in sorted(movie_dir.glob('*.mkv')):
            stem = video_path.stem  # bourne01
            m = re.match(rf'{movie}(\d+)$', stem)
            if not m:
                continue
            idx = m.group(1)
            tsv = _movie10_transcript_path(stimuli_root, movie, idx)
            rows.append({
                'stimulus_id': f'{movie}{idx}',
                'movie': movie,
                'season': '',
                'episode': '',
                'split': idx,
                'run': '1',  # life/figures shown twice but same .mkv
                'video_path': str(video_path),
                'transcript_path': str(tsv) if tsv.exists() else '',
            })
    return pd.DataFrame(rows)


def _build_friends_s7_stim_set(stimuli_root: Path) -> pd.DataFrame:
    rows = []
    s7_dir = stimuli_root / 'movies' / 'friends' / 's7'
    for video_path in sorted(s7_dir.glob('friends_s*.mkv')):
        stem = video_path.stem
        m = re.match(r'friends_s(\d+)e(\d+)([a-z])$', stem)
        if not m:
            continue
        season, episode, split = m.groups()
        tsv = _friends_transcript_path(
            stimuli_root, season, episode, split)
        rows.append({
            'stimulus_id': f'friends_s{season}e{episode}{split}',
            'movie': 'friends',
            'season': season,
            'episode': episode,
            'split': split,
            'run': '1',
            'video_path': str(video_path),
            'transcript_path': str(tsv) if tsv.exists() else '',
        })
    return pd.DataFrame(rows)


def _build_ood_stim_set(stimuli_root: Path) -> pd.DataFrame:
    rows = []
    ood_dir = stimuli_root / 'movies' / 'ood'
    for movie_dir in sorted(ood_dir.iterdir()):
        if not movie_dir.is_dir():
            continue
        movie = movie_dir.name
        # Per-split files have pattern task-{movie}{N}_video.mkv;
        # the unsplit task-{movie}_video.mkv exists too — skip it.
        for video_path in sorted(movie_dir.glob('task-*_video.mkv')):
            stem = video_path.stem  # task-chaplin1_video
            m = re.match(r'task-([a-z]+)(\d+)_video$', stem)
            if not m:
                continue
            mv, idx = m.groups()
            tsv = _ood_transcript_path(stimuli_root, mv, idx)
            tsv_str = str(tsv) if tsv and tsv.exists() else ''
            rows.append({
                'stimulus_id': f'{mv}{idx}',
                'movie': mv,
                'season': '',
                'episode': '',
                'split': idx,
                'run': '1',
                'video_path': str(video_path),
                'transcript_path': tsv_str,
            })
    return pd.DataFrame(rows)


def _build_stim_set_csv(algonauts_root: Path, split: str) -> pd.DataFrame:
    stimuli_root = algonauts_root / 'stimuli'
    if split == 'friends':
        # Training split bundles Friends S1-S6 + Movie10
        df_friends = _build_friends_stim_set(stimuli_root)
        df_movie10 = _build_movie10_stim_set(stimuli_root)
        return pd.concat([df_friends, df_movie10], ignore_index=True)
    elif split == 'friends_s7':
        return _build_friends_s7_stim_set(stimuli_root)
    elif split == 'ood':
        return _build_ood_stim_set(stimuli_root)
    raise ValueError(f"unknown split: {split!r}")


# ── Assembly builders ─────────────────────────────────────────────


def _stimulus_id_from_h5_key(key: str) -> Optional[str]:
    """Map an .h5 dataset key to its stimulus_id (or None to skip)."""
    f = _parse_friends_key(key)
    if f:
        return f"friends_s{f['season']}e{f['episode']}{f['split']}"
    m = _parse_movie10_key(key)
    if m:
        return f"{m['movie']}{m['split']}"
    return None


def _build_train_assembly(algonauts_root: Path, subject: int,
                          output_path: Path):
    """Merge Friends S1-S6 + Movie10 .h5 files for one subject into a
    single (presentation, neuroid) assembly. presentation tagged with
    (subject, movie, season, episode, split, run, t_within_run).
    """
    import h5py
    import xarray as xr
    from brainscore_core.supported_data_standards.brainio.assemblies import (
        NeuronRecordingAssembly)

    func_dir = algonauts_root / 'fmri' / f'sub-{subject:02d}' / 'func'
    h5_files = []
    for pat in (
        f'sub-{subject:02d}_task-friends_*_bold.h5',
        f'sub-{subject:02d}_task-movie10_*_bold.h5',
    ):
        h5_files.extend(sorted(func_dir.glob(pat)))
    if not h5_files:
        raise FileNotFoundError(
            f"No .h5 files found for sub-{subject:02d} in {func_dir}")

    # First pass: count total TRs
    total_TRs = 0
    keys_per_file: List[Tuple[Path, List[str]]] = []
    for path in h5_files:
        with h5py.File(path, 'r') as f:
            keys = sorted(f.keys())
            for k in keys:
                total_TRs += f[k].shape[0]
            keys_per_file.append((path, keys))

    print(f'  total TRs across {len(h5_files)} files: {total_TRs}')

    # Allocate
    Y = np.empty((total_TRs, SCHAEFER_N_PARCELS), dtype=np.float32)
    stim_ids: List[str] = []
    sessions: List[str] = []
    movies: List[str] = []
    seasons: List[str] = []
    episodes: List[str] = []
    splits: List[str] = []
    runs: List[str] = []
    t_within: List[int] = []

    cursor = 0
    for path, keys in keys_per_file:
        with h5py.File(path, 'r') as f:
            for k in keys:
                arr = f[k][...]  # (n_TR, 1000)
                friends_meta = _parse_friends_key(k)
                movie10_meta = _parse_movie10_key(k)
                meta = friends_meta or movie10_meta
                if meta is None:
                    print(f'  WARNING: unrecognized dataset key {k!r}; '
                          f'skipping')
                    continue
                stim_id = _stimulus_id_from_h5_key(k)
                n = arr.shape[0]
                Y[cursor:cursor+n] = arr
                stim_ids.extend([stim_id] * n)
                sessions.extend([meta['session']] * n)
                movies.extend([meta['movie']] * n)
                seasons.extend([meta['season']] * n)
                episodes.extend([meta['episode']] * n)
                splits.extend([meta['split']] * n)
                runs.extend([meta['run']] * n)
                t_within.extend(list(range(n)))
                cursor += n

    Y = Y[:cursor]   # trim if any datasets were skipped

    # Build xarray with a presentation dim carrying all the metadata
    coords = {
        'stimulus_id': ('presentation', stim_ids),
        'session': ('presentation', sessions),
        'movie': ('presentation', movies),
        'season': ('presentation', seasons),
        'episode': ('presentation', episodes),
        'split': ('presentation', splits),
        'run': ('presentation', runs),
        't_within_run': ('presentation', t_within),
        'subject': ('presentation', [str(subject)] * cursor),
    }
    da = xr.DataArray(
        Y, dims=('presentation', 'neuroid'),
        coords={**coords,
                'parcel_id': ('neuroid', list(range(SCHAEFER_N_PARCELS)))},
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Use plain DataArray (NeuronRecordingAssembly's gather_indexes can
    # be reapplied at load time by the benchmark).
    da.to_netcdf(str(output_path))
    print(f'  wrote {output_path}  shape {da.shape}')


def _build_held_out_stub(algonauts_root: Path, subject: int,
                         split: str, output_path: Path):
    """For Friends S7 and OOD: no fMRI; build a stub with NaN values
    at the expected (n_TR, 1000) shape per movie segment.
    """
    import xarray as xr

    target = (algonauts_root / 'fmri' / f'sub-{subject:02d}'
              / 'target_sample_number'
              / f'sub-{subject:02d}_'
                f'{"friends-s7" if split == "friends_s7" else "ood"}'
                f'_fmri_samples.npy')
    sample_counts = np.load(target, allow_pickle=True).item()
    # sample_counts: {'s07e01a': 460, ...} or {'chaplin1': 432, ...}

    total = sum(sample_counts.values())
    Y = np.full((total, SCHAEFER_N_PARCELS), np.nan, dtype=np.float32)

    stim_ids: List[str] = []
    t_within: List[int] = []
    for stim, n in sample_counts.items():
        if split == 'friends_s7':
            full_id = f'friends_{stim}'  # s07e01a -> friends_s07e01a
        else:
            full_id = stim                # chaplin1 stays chaplin1
        stim_ids.extend([full_id] * n)
        t_within.extend(list(range(n)))

    da = xr.DataArray(
        Y, dims=('presentation', 'neuroid'),
        coords={
            'stimulus_id': ('presentation', stim_ids),
            't_within_run': ('presentation', t_within),
            'subject': ('presentation', [str(subject)] * total),
            'parcel_id': ('neuroid', list(range(SCHAEFER_N_PARCELS))),
        },
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    da.to_netcdf(str(output_path))
    print(f'  wrote held-out stub {output_path}  shape {da.shape}')


# ── Main ──────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--algonauts-root', type=Path, required=True,
                        help='DataLad-installed dataset directory')
    parser.add_argument('--output-root', type=Path,
                        default=Path('~/.brainio/algonauts2025').expanduser())
    parser.add_argument('--subjects', type=int, nargs='+',
                        default=list(ALGONAUTS_SUBJECTS))
    parser.add_argument('--splits', type=str, nargs='+',
                        default=list(ALGONAUTS_SPLITS),
                        choices=ALGONAUTS_SPLITS)
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()

    args.output_root.mkdir(parents=True, exist_ok=True)
    print(f'Algonauts root:  {args.algonauts_root}')
    print(f'Output root:     {args.output_root}')
    print(f'Subjects:        {args.subjects}')
    print(f'Splits:          {args.splits}')

    # 1. Stimulus_set CSVs (one per split)
    for split in args.splits:
        out = args.output_root / f'algonauts2025_stim_{split}.csv'
        if out.exists() and not args.overwrite:
            print(f'\n[skip] stim_set exists: {out}')
            continue
        print(f'\n--- Building stim_set for {split} ---')
        df = _build_stim_set_csv(args.algonauts_root, split)
        df.to_csv(out, index=False)
        print(f'  wrote {len(df)} rows -> {out}')

    # 2. Per-(split, subject) assembly netCDF
    for split in args.splits:
        for subject in args.subjects:
            out = (args.output_root /
                   f'algonauts2025_{split}_sub{subject:02d}.nc')
            if out.exists() and not args.overwrite:
                print(f'\n[skip] assembly exists: {out}')
                continue
            print(f'\n--- Building {split} assembly for sub-{subject:02d} ---')
            try:
                if split == 'friends':
                    _build_train_assembly(
                        args.algonauts_root, subject, out)
                else:
                    _build_held_out_stub(
                        args.algonauts_root, subject, split, out)
            except FileNotFoundError as e:
                print(f'  [skip] {e}')
                continue
    print('\nDone.')


if __name__ == '__main__':
    main()
