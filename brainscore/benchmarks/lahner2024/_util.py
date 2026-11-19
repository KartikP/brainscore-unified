"""What the Lahner2024 benchmarks show a subject, and how they read its answer.

Both Lahner2024 benchmarks (GLM betas and TR time series) build their stimulus
tables and record the subject here. A benchmark reads a subject only through
``participate`` (the core helper ``neural_response``) and routes only on the
subject's declared channels (Contract Rules 1.1, 2.1, 2.2, 5.9).
"""
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
# frames per clip (``DEFAULT_SAMPLE_TIMES_MS`` of the data plugin).
VIDEO_DURATION_MS = 3000
TIME_BINS = [(0, VIDEO_DURATION_MS)]


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
    ``ExecutionNotFeasibleError`` before any compute: pre-flight accepts a
    region answered from ``vision:image`` while only the clips are shown.
    """
    from brainscore_core.compatibility import ExecutionNotFeasibleError
    visual = next((c for c in ('vision:video', 'vision:image') if c in taken), None)
    channels = {'video': visual, 'audio': 'audio' if 'audio' in taken else None}
    plan = {'video': [], 'audio': []}
    for region in regions:
        listed = subject.output_inputs.get(f'neural:{region}')
        usable = [stream for stream, channel in channels.items()
                  if channel is not None and (listed is None or channel in listed)]
        if not usable:
            raise ExecutionNotFeasibleError(
                f'{subject.identifier!r} answers neural:{region} only from {sorted(listed or ())}, '
                f'and takes none of the streams shown ({sorted(c for c in channels.values() if c)}).',
                channels={f'neural:{region}'})
        own = _OWN_STREAM[region]
        plan[own if own in usable else usable[0]].append(region)
    return [(stream, channels[stream], requested) for stream, requested in plan.items() if requested]


# ── Stimulus tables ───────────────────────────────────────────────────────


def clip_table(stimuli, identifier: str, ids: Optional[Sequence[str]] = None,
               sound: bool = False):
    """One row per clip of ``stimuli``: ``stimulus_id`` and the clip's file.

    The file is the video (``video_path``). With ``sound`` it is the sound track
    demuxed from the video (``audio_path``, ``<video stem>.wav`` in the local asset
    ``lahner2024-sound``), and every file must exist. ``ids`` picks and orders the
    clips; the default is every row of ``stimuli`` in table order. The table is named
    ``identifier`` and the signature of its files (``input_identifier``).
    """
    import pandas as pd
    from brainscore_core.supported_data_standards.brainio.stimuli import StimulusSet
    from brainscore.presentation import input_identifier

    ids = list(stimuli['stimulus_id']) if ids is None else list(ids)
    videos = [stimuli.get_stimulus(stimulus_id) for stimulus_id in ids]
    if not sound:
        column, files = 'video_path', [str(video) for video in videos]
    else:
        from brainscore.data import local
        from brainscore.data.lahner2024 import LAHNER_SOUND
        audio_dir = local.path(LAHNER_SOUND.name)
        column = 'audio_path'
        files = [str(audio_dir / f'{Path(video).stem}.wav') for video in videos]
        _require_files(files, audio_dir, LAHNER_SOUND)
    frame = pd.DataFrame({'stimulus_id': ids, column: files})
    table = StimulusSet(frame)
    table.identifier = input_identifier(identifier, frame, [column])
    table.stimulus_paths = dict(zip(frame['stimulus_id'], frame[column]))
    return table


def frame_table(stimuli, sample_times_ms: Sequence[float]):
    """One row per (clip, sample time): a prepared still of every clip of ``stimuli``.

    Rows carry ``clip_id`` and ``frame_time_ms`` (see ``expand_clip_to_frames``).
    The frames are the local asset ``lahner2024-frames``; every one must exist. The
    table is named by its frame files (``input_identifier``).
    """
    import pandas as pd
    from brainscore_core.supported_data_standards.brainio.stimuli import StimulusSet
    from brainscore.data import local
    from brainscore.data.lahner2024 import LAHNER_FRAMES, frame_file
    from brainscore.presentation import input_identifier
    from brainscore.temporal import expand_clip_to_frames

    frames_dir = local.path(LAHNER_FRAMES.name)
    # The clip paths only: the frames are named by their own files, not the clips'.
    ids = list(stimuli['stimulus_id'])
    clips = StimulusSet(pd.DataFrame({'stimulus_id': ids,
                                      'video_path': [str(stimuli.get_stimulus(i)) for i in ids]}))
    clips.identifier = stimuli.identifier
    _require_files([str(frame_file(frames_dir, video, time_ms))
                    for video in clips['video_path'] for time_ms in sample_times_ms],
                   frames_dir, LAHNER_FRAMES,
                   options=' --sample-times-ms ' + ' '.join(f'{t:g}' for t in sample_times_ms))
    frames = expand_clip_to_frames(
        clips,
        sample_times_ms=sample_times_ms,
        frame_extractor=lambda video_path, t_ms: str(frame_file(frames_dir, video_path, t_ms)),
        clip_id_col='stimulus_id',
        video_col='video_path',
    )
    frames.identifier = input_identifier(f'{stimuli.identifier}-frames', frames, ['image_file_name'])
    return frames


def _require_files(files, folder, asset, options=''):
    """Raise ``LocalDataMissing`` naming the preparation command when a file is absent."""
    from brainscore.data.local import LocalDataMissing
    missing = [path for path in files if not Path(path).exists()]
    if missing:
        raise LocalDataMissing(
            f"{len(missing)} of {len(files)} files of {asset.name} are missing under {folder} "
            f"(first: {missing[0]}). Prepare them:\n    {asset.prepare}{options}")


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
