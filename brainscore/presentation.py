"""Show a subject the stimulus people saw, in the form the subject takes natively.

A subject declares the visual forms it has (``vision:still``, ``vision:clip``).
A benchmark presents the stimulus in its own form when the subject has it, and
adapts it only when the form is missing, recording the adaptation on the score:

    adaptation = None if takes_clips(subject) else frames_from_clip(sample_times_ms)
    score.attrs['input_adaptation'] = adaptation
"""


def takes_clips(subject) -> bool:
    """Whether the subject takes video clips natively."""
    return 'vision:clip' in subject.in_channels


def takes_stills(subject) -> bool:
    """Whether the subject takes still images natively."""
    return 'vision:still' in subject.in_channels


def frames_from_clip(sample_times_ms, aggregate='mean'):
    """The record of showing a stills-only subject frames sampled from each clip."""
    return {'kind': 'frames_from_clip', 'sample_times_ms': [float(t) for t in sample_times_ms],
            'aggregate': aggregate}


def response_stimulus_ids(response):
    """The ``stimulus_id`` of each row of a subject's response assembly.

    Reads the ``stimulus_id`` coordinate (or MultiIndex level). When
    ``stimulus_id`` is the only presentation coordinate, brainio keeps it as
    the plain ``presentation`` index without its name, so that index is read.
    A benchmark joins its own stimulus metadata to these ids; it never relies
    on the subject to return other columns.
    """
    import numpy as np
    import pandas as pd
    try:
        return np.asarray(response['stimulus_id'].values)
    except KeyError:
        pass
    index = response.indexes.get('presentation')
    if index is None or isinstance(index, pd.MultiIndex):
        raise KeyError('the response carries no stimulus_id on its presentation axis')
    return np.asarray(index.values)
