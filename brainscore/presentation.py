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
