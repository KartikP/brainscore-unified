"""What the Lahner2024 benchmarks show a subject, and how they read its answer.

Both Lahner2024 benchmarks (GLM betas and TR time series) build their stimulus
tables and record the subject here. A benchmark reads a subject only through
``participate`` (the core helper ``neural_response``) and routes only on the
subject's declared channels (Contract Rules 1.1, 2.1, 2.2, 5.9).
"""
import os
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

# The channels a Lahner2024 benchmark sends, and the regions it then asks for.
# The visual stream goes as clips on `vision:video`, or as frames sampled from the
# clips on `vision:image`, by the form the subject declares; `neural:IT` records
# it. The sound track goes on `audio`; `neural:A1` records it.
CHANNEL_REGIONS = {
    frozenset({'vision:image', 'vision:video'}): ('IT',),
    frozenset({'vision:image', 'vision:video', 'audio'}): ('IT', 'A1'),
}

# Videos are 3 seconds long; by default a stills-only subject sees three
# frames per clip (start, middle, end).
VIDEO_DURATION_MS = 3000
DEFAULT_SAMPLE_TIMES_MS = (500, 1500, 2500)
TIME_BINS = [(0, VIDEO_DURATION_MS)]
# Default location of the sound tracks (16 kHz mono WAV, one per clip).
DEFAULT_AUDIO_DIR = '~/.brainio/lahner2024_audio_16k'


def channel_regions(channels: Sequence[str]):
    """``(channels, regions)`` for a channel set of ``CHANNEL_REGIONS``; raises otherwise."""
    channels = frozenset(channels)
    if channels not in CHANNEL_REGIONS:
        raise ValueError(f"channels must be one of {[sorted(c) for c in CHANNEL_REGIONS]}; "
                         f"got {sorted(channels)}.")
    return channels, CHANNEL_REGIONS[channels]


def read_stimulus_ids(assembly):
    """Read stimulus ids from an assembly's presentation axis, whether they
    live as a MultiIndex level or a plain coord, and whether the assembly came
    from native-video extraction (uses 'stimulus_id') or frame-aggregation via
    ``temporal_bin`` (uses 'clip_id').

    Tried in order: MultiIndex level 'stimulus_id', MultiIndex level 'clip_id',
    then a top-level 'stimulus_id'/'clip_id' coord.
    """
    if 'presentation' in assembly.indexes:
        idx = assembly.indexes['presentation']
        if hasattr(idx, 'get_level_values'):
            names = list(idx.names) if hasattr(idx, 'names') else []
            if 'stimulus_id' in names:
                return list(idx.get_level_values('stimulus_id'))
            if 'clip_id' in names:
                return list(idx.get_level_values('clip_id'))
    for col in ('stimulus_id', 'clip_id'):
        if col in assembly.coords:
            return list(assembly[col].values)
    raise KeyError(
        f"assembly has neither 'stimulus_id' nor 'clip_id' on its "
        f"presentation axis; coords={list(assembly.coords)}")


def taken_inputs(subject, channels) -> set:
    """The benchmark inputs ``channels`` that the subject takes in, matched as
    pre-flight matches them (``channel_declared``, rule 5.2)."""
    from brainscore_core.streaming import channel_declared
    return {channel for channel in channels if channel_declared(channel, subject.in_channels)}


# The stream that records each region when the subject can answer it from there.
_OWN_STREAM = {'IT': 'video', 'A1': 'audio'}


def stream_regions(subject, taken, regions) -> List[Tuple[str, str, List[str]]]:
    """``[(stream, input channel, regions), ...]``: the streams shown to ``subject``
    and the regions the session of each requests, visual stream first.

    Every region in ``regions`` is requested once (rule 5.1a). The visual stream
    is the clips when the subject takes ``vision:video``, else the frames on
    ``vision:image``; the sound is ``audio``. A stream is shown only when the
    subject takes it (``taken``). A region is recorded from its own stream (IT
    from the visual one, A1 from the sound) when the subject can answer it from
    there (``output_inputs``, rule 2.5), else from the other stream it can
    answer it from. A region that no shown stream can answer raises
    ``CompatibilityError`` before any compute; pre-flight gives N/A for it.
    """
    from brainscore_core.compatibility import CompatibilityError
    visual = next((c for c in ('vision:video', 'vision:image') if c in taken), None)
    channels = {'video': visual, 'audio': 'audio' if 'audio' in taken else None}
    plan = {'video': [], 'audio': []}
    for region in regions:
        listed = subject.output_inputs.get(f'neural:{region}')
        usable = [stream for stream, channel in channels.items()
                  if channel is not None and (listed is None or channel in listed)]
        if not usable:
            raise CompatibilityError(
                f'{subject.identifier!r} answers neural:{region} only from {sorted(listed or ())}, '
                f'and takes none of the streams shown ({sorted(c for c in channels.values() if c)}).')
        own = _OWN_STREAM[region]
        plan[own if own in usable else usable[0]].append(region)
    return [(stream, channels[stream], requested) for stream, requested in plan.items() if requested]


# ── Stimulus tables ───────────────────────────────────────────────────────


def clip_table(stimuli, identifier: str, ids: Optional[Sequence[str]] = None,
               audio_dir: Optional[Path] = None):
    """One row per clip of ``stimuli``: ``stimulus_id`` and the clip's file.

    The file is the video (``video_path``). With ``audio_dir`` it is the sound
    track demuxed from the video (``audio_path``, ``<audio_dir>/<video stem>.wav``),
    and every file must exist. ``ids`` picks and orders the clips; the default
    is every row of ``stimuli`` in table order.
    """
    import pandas as pd
    from brainscore_core.supported_data_standards.brainio.stimuli import StimulusSet

    ids = list(stimuli['stimulus_id']) if ids is None else list(ids)
    videos = [stimuli.get_stimulus(stimulus_id) for stimulus_id in ids]
    if audio_dir is None:
        column, files = 'video_path', [str(video) for video in videos]
    else:
        column = 'audio_path'
        files = [str(audio_dir / f'{Path(video).stem}.wav') for video in videos]
        missing = [path for path in files if not Path(path).exists()]
        if missing:
            raise FileNotFoundError(
                f"{len(missing)} audio files missing under {audio_dir}. Run `python -m "
                f"brainscore.data.lahner2024.data_packaging.prepare_audio_tracks "
                f"--audio-dir {audio_dir}` first. First missing: {missing[0]}")
    frame = pd.DataFrame({'stimulus_id': ids, column: files})
    table = StimulusSet(frame)
    table.identifier = identifier
    table.stimulus_paths = dict(zip(frame['stimulus_id'], frame[column]))
    return table


def resolve_frames_dir(frames_dir=None) -> Path:
    """Where extracted frames are kept: ``frames_dir``, else
    ``BRAINSCORE_LAHNER_FRAMES_DIR``, else ``~/brainscore-data/lahner2024/frames``."""
    if frames_dir:
        return Path(frames_dir)
    configured = os.environ.get('BRAINSCORE_LAHNER_FRAMES_DIR')
    return Path(configured) if configured else Path.home() / 'brainscore-data' / 'lahner2024' / 'frames'


def frame_table(stimuli, sample_times_ms: Sequence[float], frames_dir: Path):
    """One row per (clip, sample time): a still extracted from every clip of ``stimuli``.

    Rows carry ``clip_id`` and ``frame_time_ms`` (see ``expand_clip_to_frames``).
    Frames are extracted once with OpenCV and kept under ``frames_dir``.
    """
    from brainscore.temporal import expand_clip_to_frames

    if not frames_dir.exists():
        try:
            frames_dir.mkdir(parents=True, exist_ok=True)
        except OSError as error:
            raise FileNotFoundError(
                f"Lahner2024 frame cache directory {frames_dir} does not "
                f"exist and could not be created ({error}). Set "
                f"BRAINSCORE_LAHNER_FRAMES_DIR to a writable path, or pass "
                f"frames_dir=... when constructing the benchmark.") from error
    return expand_clip_to_frames(
        clip_table(stimuli, stimuli.identifier),
        sample_times_ms=sample_times_ms,
        frame_extractor=lambda video_path, t_ms: _extract_frame_with_cv2(
            video_path, t_ms, frames_dir=frames_dir),
        clip_id_col='stimulus_id',
        video_col='video_path',
    )


def _extract_frame_with_cv2(video_path, time_ms: float, frames_dir: Path) -> str:
    """Extract one frame of ``video_path`` at ``time_ms`` as a PNG under ``frames_dir``.

    Returns the path. Idempotent: an existing frame is returned without
    extracting it again.
    """
    import cv2  # imported here to avoid making it a brainscore_core dep

    video_path = Path(video_path)
    frames_dir.mkdir(parents=True, exist_ok=True)
    out_path = frames_dir / f'{video_path.stem}_t{int(time_ms)}.png'
    if out_path.exists():
        return str(out_path)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise IOError(f"cv2 could not open {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    target_frame_idx = int(round(time_ms / 1000.0 * fps))
    cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame_idx)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise IOError(
            f"cv2 failed to read frame at {time_ms}ms "
            f"(target idx {target_frame_idx}, fps {fps}) from {video_path}")
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)   # cv2 reads BGR
    from PIL import Image
    Image.fromarray(frame).save(out_path)
    return str(out_path)


# ── Recording ─────────────────────────────────────────────────────────────


def record_visual(subject, clips, frames, regions=('IT',)):
    """The responses for ``regions`` (one session) to every clip: whole clips
    (``clips()``) when the subject declares ``vision:video``, else frames
    (``frames()``) pooled per clip. A list in the order of ``regions``.

    The tables are built only for the route taken.
    """
    from brainscore.presentation import takes_clips
    if takes_clips(subject):
        return record_clips(subject, clips(), regions)
    return record_frames_per_clip(subject, frames(), regions)


def _record(subject, stimuli, regions):
    """One session on ``stimuli`` requesting ``neural:<region>`` for every region."""
    from brainscore_core.streaming_helpers import stimulus_session
    outputs = [f'neural:{region}' for region in regions]
    session = stimulus_session(stimuli, outputs, time_bins=TIME_BINS)
    subject.participate(session)
    return [session.collect(output) for output in outputs]


def record_clips(subject, clips, regions):
    """Show whole clips in one session; return the responses for ``regions``.

    Each response must answer each clip exactly once (``response_stimulus_ids``).
    A response whose only presentation coordinate is ``stimulus_id`` gets that
    coordinate back under its name.
    """
    from brainscore.presentation import response_stimulus_ids
    results = []
    for result in _record(subject, clips, regions):
        ids = response_stimulus_ids(result, clips['stimulus_id'])
        try:
            result['stimulus_id']
        except KeyError:
            result = result.assign_coords(stimulus_id=('presentation', ids))
        results.append(result)
    return results


def record_frames_per_clip(subject, frames, regions):
    """Show frames sampled from each clip in one session; pool each response for
    ``regions`` per clip and time bin.

    ``frames`` is the benchmark's own frame table (``stimulus_id``,
    ``clip_id``, ``frame_time_ms``). The subject may return only
    ``stimulus_id`` on the presentation axis, so the clip and the frame time
    of each response row are joined from that table by ``stimulus_id``.
    """
    from brainscore.temporal import temporal_bin
    return [temporal_bin(with_frame_columns(per_frame, frames), time_bins=TIME_BINS)
            for per_frame in _record(subject, frames, regions)]


def with_frame_columns(per_frame, frames):
    """``per_frame`` with ``clip_id`` and ``frame_time_ms`` from ``frames``.

    Rebuilds the assembly so the benchmark's table is the one source of the
    clip and the frame time; values and neuroid coordinates are kept as given.
    """
    from brainscore_core.supported_data_standards.brainio.assemblies import (
        NeuroidAssembly, walk_coords)
    from brainscore.presentation import response_stimulus_ids
    per_frame = per_frame.transpose('presentation', 'neuroid')
    ids = response_stimulus_ids(per_frame, frames['stimulus_id'])
    rows = frames.set_index('stimulus_id').loc[ids]
    coords = {
        'stimulus_id': ('presentation', ids),
        'clip_id': ('presentation', rows['clip_id'].values),
        'frame_time_ms': ('presentation', rows['frame_time_ms'].values),
    }
    for name, dims, values in walk_coords(per_frame):
        if set(dims) == {'neuroid'}:
            coords[name] = ('neuroid', values)
    return NeuroidAssembly(per_frame.values, coords=coords,
                           dims=['presentation', 'neuroid'])


def time_mean(assembly):
    """``(n_clips, n_features)``: the response averaged over its time bins, if it has any."""
    data = assembly.values
    return data.mean(axis=1) if data.ndim == 3 else data


def align(features, ids, to_ids):
    """The rows of ``features`` (in the order of ``ids``) put in the order of ``to_ids``."""
    if list(ids) == list(to_ids):
        return features
    order = {stimulus_id: i for i, stimulus_id in enumerate(ids)}
    return features[[order[stimulus_id] for stimulus_id in to_ids]]
