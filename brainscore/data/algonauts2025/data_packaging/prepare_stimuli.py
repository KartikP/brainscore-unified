"""Cut the Friends and Movie10 films into one stimulus file per scan.

Reads the movie table and the assembly that ``prepare_assembly`` wrote, and the
native films they point to. Writes under the prepared root:

- ``frames/<movie>/<t + 1:04d>.jpg``: the frame at the middle of scan ``t``,
  224 x 224 (``vision:image`` form);
- ``clips/<movie>/<t:04d>.mp4``: the 1.49 s of scan ``t``, 224 x 224, 30 frames,
  no sound (``vision:video`` form);
- ``sound/<movie>/<t:04d>.wav``: the sound of the same 1.49 s, 16 kHz, mono,
  23840 samples (``audio``, with the clips).

One ffmpeg call per film and medium. A file that is already there is kept, so a
second run cuts only what is missing. Where a film ends before the last scan of
its run, the missing files are copies of the last whole file. Usage::

    python -m brainscore.data.algonauts2025.data_packaging.prepare_stimuli \\
        --split friends --participant 1 --media frames

The number of scans of each film comes from the participant's assembly, so the
participant names the assembly to read.
"""
import argparse
import logging
import shutil
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from brainscore.data.algonauts2025 import (
    CLIP_FPS,
    CLIP_FRAMES,
    MEDIA,
    PARTICIPANTS,
    SOUND_RATE_HZ,
    SOUND_SAMPLES,
    SPLITS,
    TR_SEC,
    scan_files,
)

logger = logging.getLogger(__name__)


def _extract_frames_one(args):
    """Worker for the frame-extraction pool. Tuple-args because Pool.imap
    doesn't take starargs."""
    video_path, out_dir, fps_offset, resize, n_TRs = args
    out_pat = f'{out_dir}/%04d.jpg'
    duration = TR_SEC * (n_TRs + 1)  # safety bound
    cmd = [
        'ffmpeg', '-y', '-loglevel', 'error',
        '-ss', f'{TR_SEC * fps_offset:.4f}',
        '-i', video_path,
        '-t', f'{duration:.4f}',
        '-vf', f'fps=1/{TR_SEC},scale={resize}:{resize}',
        '-q:v', '3',
        out_pat,
    ]
    subprocess.run(cmd, check=False, capture_output=True)
    return video_path


def _clip_ffmpeg_command(video_path, out_pattern, resize, n_TRs=None):
    """The one ffmpeg call that cuts a movie into one clip per TR.

    Frames are resampled to ``CLIP_FPS`` and a key frame is forced every
    ``CLIP_FRAMES`` frames (``-g`` with scene-cut detection off), so the segment
    muxer cuts exactly at clip borders. ``-segment_time_delta`` keeps the cut
    from missing a key frame that lands a rounding error before ``t x TR_SEC``.
    Clip ``t`` is the movie time [t x TR_SEC, (t+1) x TR_SEC). No sound.
    """
    cmd = ['ffmpeg', '-y', '-loglevel', 'error', '-i', str(video_path)]
    if n_TRs is not None:
        cmd += ['-t', f'{TR_SEC * n_TRs:.4f}']
    return cmd + [
        '-an', '-vf', f'fps={CLIP_FPS},scale={resize}:{resize}',
        '-c:v', 'libx264', '-preset', 'veryfast', '-crf', '23', '-pix_fmt', 'yuv420p',
        '-g', str(CLIP_FRAMES), '-keyint_min', str(CLIP_FRAMES), '-sc_threshold', '0',
        '-f', 'segment', '-segment_time', f'{TR_SEC}', '-segment_time_delta', '0.01',
        '-segment_start_number', '0', '-reset_timestamps', '1', '-segment_format', 'mp4',
        str(out_pattern),
    ]


def _clip_frame_count(path) -> int:
    """Frames in a clip, from the mp4 header (0 when the file cannot be read)."""
    out = subprocess.run(
        ['ffprobe', '-v', 'error', '-select_streams', 'v:0',
         '-show_entries', 'stream=nb_frames', '-of', 'csv=p=0', str(path)],
        check=False, capture_output=True, text=True).stdout.strip()
    return int(out) if out.isdigit() else 0


def _cut_clips_one(args):
    """Worker for the clip-cutting pool: one ffmpeg call per movie.

    The call writes into a work folder. A clip moves to ``{out_dir}/{t:04d}.mp4``
    only when it has exactly ``CLIP_FRAMES`` frames and ``t < n_TRs``. A short
    clip at the end of the movie is dropped, so a clip file is always complete;
    the caller fills what is missing.
    """
    video_path, out_dir, resize, n_TRs = args
    out_dir = Path(out_dir)
    work = out_dir / '_cut'
    work.mkdir(parents=True, exist_ok=True)
    subprocess.run(_clip_ffmpeg_command(video_path, work / '%04d.mp4', resize, n_TRs),
                   check=False, capture_output=True)
    for cut in sorted(work.glob('*.mp4')):
        t = int(cut.stem)
        if t < n_TRs and _clip_frame_count(cut) == CLIP_FRAMES:
            shutil.move(str(cut), str(out_dir / f'{t:04d}.mp4'))
    shutil.rmtree(work, ignore_errors=True)
    return video_path


def _sound_ffmpeg_command(video_path, out_pattern, n_TRs=None):
    """The one ffmpeg call that cuts the sound of a movie into one WAV per TR.

    The filter resamples to 16 kHz mono, groups the samples into packets of
    exactly ``SOUND_SAMPLES`` and numbers them from 0 (``asetpts``), so the
    segment muxer cuts at every packet. Chunk ``t`` is the sound of the movie
    time [t x TR_SEC, (t+1) x TR_SEC), counted from the start of the sound.
    No picture.
    """
    cmd = ['ffmpeg', '-y', '-loglevel', 'error', '-i', str(video_path)]
    if n_TRs is not None:
        cmd += ['-t', f'{TR_SEC * n_TRs:.4f}']
    return cmd + [
        '-vn', '-af',
        (f'aresample={SOUND_RATE_HZ},aformat=sample_fmts=s16:channel_layouts=mono,'
         f'asetnsamples=n={SOUND_SAMPLES}:p=0,asetpts=N/SR/TB'),
        '-c:a', 'pcm_s16le',
        '-f', 'segment', '-segment_time', f'{TR_SEC}', '-segment_time_delta', '0.01',
        '-segment_start_number', '0', '-reset_timestamps', '1', '-segment_format', 'wav',
        str(out_pattern),
    ]


def _wav_sample_count(path) -> int:
    """Samples in a 16 kHz mono WAV (0 when the file cannot be read or has another form)."""
    import wave
    try:
        with wave.open(str(path)) as wav:
            if wav.getframerate() != SOUND_RATE_HZ or wav.getnchannels() != 1:
                return 0
            return wav.getnframes()
    except (OSError, EOFError, wave.Error):
        return 0


def _cut_sound_one(args):
    """Worker for the sound-cutting pool: one ffmpeg call per movie.

    Like ``_cut_clips_one``: a chunk moves to ``{out_dir}/{t:04d}.wav`` only
    when it has exactly ``SOUND_SAMPLES`` samples and ``t < n_TRs``. A short
    chunk at the end of the movie is dropped; the caller fills what is missing.
    """
    video_path, out_dir, n_TRs = args
    out_dir = Path(out_dir)
    work = out_dir / '_cut'
    work.mkdir(parents=True, exist_ok=True)
    subprocess.run(_sound_ffmpeg_command(video_path, work / '%04d.wav', n_TRs),
                   check=False, capture_output=True)
    for cut in sorted(work.glob('*.wav')):
        t = int(cut.stem)
        if t < n_TRs and _wav_sample_count(cut) == SOUND_SAMPLES:
            shutil.move(str(cut), str(out_dir / f'{t:04d}.wav'))
    shutil.rmtree(work, ignore_errors=True)
    return video_path


def _run_pool(worker, jobs, n_workers: int) -> None:
    import multiprocessing as mp
    # 'fork' avoids the re-import that 'spawn' triggers (which makes the worker
    # re-execute the driver script's top-level code). ffmpeg is just a
    # subprocess, so fork is safe here.
    ctx = mp.get_context('fork')
    with ctx.Pool(processes=n_workers) as pool:
        for done, _ in enumerate(pool.imap_unordered(worker, jobs), 1):
            if done % 25 == 0 or done == len(jobs):
                logger.debug('%d/%d videos done', done, len(jobs))


def _fill_trailing_files(stimulus_set, n_scans: Dict[str, int], root: Path, kind: str) -> int:
    """Copy the last existing file over every missing one; returns the count."""
    folder_name, suffix, first, _ = MEDIA[kind]
    n_filled = 0
    for _, srow in stimulus_set.iterrows():
        stim_id = srow['stimulus_id']
        n_TRs = n_scans.get(stim_id)
        if n_TRs is None:
            continue
        target_paths = [Path(root) / folder_name / stim_id / f'{t + first:04d}{suffix}'
                        for t in range(n_TRs)]
        existing = [p for p in target_paths if p.exists()]
        if not existing:
            continue
        last_existing = existing[-1]
        for p in target_paths:
            if not p.exists():
                shutil.copy(last_existing, p)
                n_filled += 1
    return n_filled


# What could not be made, and what to check, for each medium.
_NOT_CUT = {'frames': ('frames', 'the movie files'), 'clips': ('clips', 'the movie files'),
            'sound': ('sound chunks', 'the sound of the movie files')}


def prepare_stimuli(stimulus_set, n_scans: Dict[str, int], media: Sequence[str], root: Path,
                    n_workers: int = 8, resize: int = 224, fps_offset: float = 0.5) -> None:
    """Write the files of ``media`` for every scan of ``n_scans`` under ``root``.

    ``stimulus_set`` has one row per film (``stimulus_id``, ``video_path``).
    ``n_scans`` gives the scans of each film (``scans_per_stimulus`` of the
    assembly). A film whose files are all there is not cut again. Raises
    ``FileNotFoundError`` when a file could not be made.
    """
    root = Path(root)
    for kind in media:
        (root / MEDIA[kind][0]).mkdir(parents=True, exist_ok=True)
        movies = scan_files(stimulus_set, n_scans, root, kind)
        planned = []
        for _, video_path, folder, paths in movies:
            folder.mkdir(parents=True, exist_ok=True)
            if not all(path.exists() for path in paths):
                planned.append((video_path, str(folder), len(paths)))
        if kind == 'frames':
            worker = _extract_frames_one
            jobs = [(video, out_dir, fps_offset, resize, n) for video, out_dir, n in planned]
        elif kind == 'clips':
            worker = _cut_clips_one
            jobs = [(video, out_dir, resize, n) for video, out_dir, n in planned]
        else:
            worker, jobs = _cut_sound_one, planned
        if jobs:
            logger.debug('cutting %s for %d videos via ffmpeg (n_workers=%d)',
                         kind, len(jobs), n_workers)
            _run_pool(worker, jobs, n_workers)
        n_filled = _fill_trailing_files(stimulus_set, n_scans, root, kind)
        if n_filled > 0:
            logger.debug('filled %d trailing %s by duplicating the last one', n_filled, kind)
        files = [path for *_, paths in movies for path in paths]
        missing = [str(path) for path in files if not path.exists()]
        if missing:
            noun, source = _NOT_CUT[kind]
            raise FileNotFoundError(
                f"{len(missing)} of {len(files)} {noun} were not cut; the first is {missing[0]}. "
                f"Check that ffmpeg can read {source}.")


def main(argv: Optional[List[str]] = None) -> None:
    from brainscore import load_dataset, load_stimulus_set
    from brainscore.data.algonauts2025 import (
        assembly_identifier, default_data_root, scans_per_stimulus, stimulus_set_identifier)

    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('--split', choices=SPLITS, default='friends')
    parser.add_argument('--participant', type=int, choices=PARTICIPANTS, required=True,
                        help='the participant whose assembly gives the scans of each film')
    parser.add_argument('--media', nargs='+', choices=tuple(MEDIA), required=True,
                        help='frames (vision:image), clips (vision:video), sound (audio)')
    parser.add_argument('--movies', nargs='+', default=None,
                        help='only these films (stimulus ids), for a short run')
    parser.add_argument('--prepared-root', type=Path, default=None,
                        help='the prepared files (default: BRAINSCORE_ALGONAUTS_PREPARED)')
    parser.add_argument('--workers', type=int, default=8)
    args = parser.parse_args(argv)

    root = args.prepared_root or default_data_root()
    assembly = load_dataset(assembly_identifier(args.split, args.participant), root=root)
    stimulus_set = load_stimulus_set(stimulus_set_identifier(args.split), root=root)
    n_scans = scans_per_stimulus(assembly)
    if args.movies is not None:
        unknown = sorted(set(args.movies) - set(n_scans))
        if unknown:
            raise SystemExit(f'movies {unknown} are not in the assembly.')
        n_scans = {movie: n for movie, n in n_scans.items() if movie in args.movies}
    prepare_stimuli(stimulus_set, n_scans, args.media, root, n_workers=args.workers)


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    main()
