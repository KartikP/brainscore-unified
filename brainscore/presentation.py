"""Show a subject the stimulus people saw, in the form the subject takes natively.

A subject declares the visual forms it has (``vision:image``, ``vision:video``).
A benchmark presents the stimulus in its own form when the subject has it, and
adapts it only when the form is missing, recording the adaptation on the score:

    adaptation = None if takes_clips(subject) else frames_from_clip(sample_times_ms)
    score.attrs['input_adaptation'] = adaptation
"""


def takes_clips(subject) -> bool:
    """Whether the subject takes video clips natively."""
    return 'vision:video' in subject.in_channels


def takes_stills(subject) -> bool:
    """Whether the subject takes still images natively."""
    return 'vision:image' in subject.in_channels


def frames_from_clip(sample_times_ms, aggregate='mean'):
    """The record of showing a stills-only subject frames sampled from each clip."""
    return {'kind': 'frames_from_clip', 'sample_times_ms': [float(t) for t in sample_times_ms],
            'aggregate': aggregate}


def response_stimulus_ids(response, requested):
    """The ``stimulus_id`` of each row of a subject's response, checked.

    ``requested`` is the list of stimulus ids the benchmark presented. The
    response must answer each of them exactly once (in any order); anything
    else raises ``ValueError``, so a response can never be scored against the
    wrong stimuli.

    The ids are read from a ``stimulus_id`` coordinate on the ``presentation``
    axis (or a level of its MultiIndex). When ``stimulus_id`` is the only
    presentation coordinate, brainio keeps it as the plain, unnamed
    ``presentation`` index; that index is accepted only because its values
    must then equal the requested ids. Any other shape raises ``ValueError``.
    """
    from collections import Counter
    import numpy as np
    import pandas as pd

    dims = getattr(response, 'dims', ())
    if 'presentation' not in dims:
        raise ValueError(f"the response has no 'presentation' axis (dims {tuple(dims)})")
    n_rows = response.sizes['presentation']
    try:
        coord = response['stimulus_id']
    except KeyError:
        coord = None
    if coord is not None:
        if coord.dims != ('presentation',):
            raise ValueError(f"the response's stimulus_id has dims {coord.dims}; "
                             "expected ('presentation',)")
        ids = np.asarray(coord.values)
    else:
        index = response.indexes.get('presentation')
        if index is None or isinstance(index, pd.MultiIndex):
            raise ValueError('the response carries no stimulus_id on its presentation axis')
        ids = np.asarray(index.values)
    if ids.shape != (n_rows,):
        raise ValueError(f'the response has {n_rows} rows but stimulus ids of shape {ids.shape}')
    answered = [str(value) for value in ids]
    wanted = [str(value) for value in requested]
    if Counter(answered) != Counter(wanted):
        missing = sorted((Counter(wanted) - Counter(answered)).elements())[:3]
        extra = sorted((Counter(answered) - Counter(wanted)).elements())[:3]
        raise ValueError(f'the response does not answer each presented stimulus exactly once '
                         f'(missing {missing}, unexpected {extra})')
    return ids
