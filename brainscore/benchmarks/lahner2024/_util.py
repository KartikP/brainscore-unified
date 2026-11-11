"""Shared helpers for the Lahner2024 benchmark variants."""
import numpy as np


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


# ── Presenting stimuli to a subject (Contract Rules 1.1, 2.1, 2.2) ──────────
#
# Every Lahner variant reads a subject's activity only through ``participate``
# (via the core typed helper ``neural_response``) and routes only on the
# subject's declared channels. Nothing here calls ``process`` or reads a
# private attribute, so any contract subject can be scored.


def _families(channels):
    from brainscore_core.streaming import parse_channel
    return {parse_channel(channel)[0] for channel in channels}


def takes_vision(subject) -> bool:
    """Whether the subject declares any visual input (stills, clips or both)."""
    return 'vision' in _families(subject.in_channels)


def takes_audio(subject) -> bool:
    """Whether the subject declares audio input."""
    return 'audio' in _families(subject.in_channels)


def records(subject, region: str) -> bool:
    """Whether the subject declares the neural output ``neural:<region>``."""
    return f'neural:{region}' in subject.out_channels


def takes_audiovisual(subject) -> bool:
    """Whether the subject can take part in the audio + video variants.

    It must take a visual input and audio, and record ``neural:IT`` (the
    visual tower) and ``neural:A1`` (the audio tower).
    """
    return (takes_vision(subject) and takes_audio(subject)
            and records(subject, 'IT') and records(subject, 'A1'))


def record_clips(subject, clips, region: str, time_bins):
    """Show whole clips; return the subject's response for ``region``.

    The response must answer each clip exactly once (``response_stimulus_ids``).
    A response whose only presentation coordinate is ``stimulus_id`` gets that
    coordinate back under its name.
    """
    from brainscore_core.streaming_helpers import neural_response
    from brainscore.presentation import response_stimulus_ids
    result = neural_response(subject, clips, record=region, time_bins=time_bins)
    ids = response_stimulus_ids(result, clips['stimulus_id'])
    try:
        result['stimulus_id']
    except KeyError:
        result = result.assign_coords(stimulus_id=('presentation', ids))
    return result


def record_frames_per_clip(subject, frames, region: str, time_bins):
    """Show frames sampled from each clip; pool them per clip and time bin.

    ``frames`` is the benchmark's own frame table (``stimulus_id``,
    ``clip_id``, ``frame_time_ms``). The subject may return only
    ``stimulus_id`` on the presentation axis, so the clip and the frame time
    of each response row are joined from that table by ``stimulus_id``.
    """
    from brainscore_core.streaming_helpers import neural_response
    from brainscore.temporal import temporal_bin
    per_frame = neural_response(subject, frames, record=region, time_bins=time_bins)
    return temporal_bin(with_frame_columns(per_frame, frames), time_bins=time_bins)


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
