"""Read a motion report and compare unflipped model vectors with the three participants.

This metric stays inside the benchmark: the flip reversal and the four-flip mean
belong to this study's design, and no other benchmark uses them.
"""
import json
import math
import re

import numpy as np

_FENCE = chr(96) * 3


def parse_report(reply) -> tuple[dict | None, str | None]:
    """One JSON object (plain, or a whole-reply json code fence) with the two numbers.

    Returns the report and None, or None and the reason the reply is not usable.
    No value is inferred or changed.
    """
    def unique_fields(pairs):
        if len({name for name, _ in pairs}) != len(pairs):
            raise ValueError('duplicate JSON field')
        return dict(pairs)

    if not isinstance(reply, str):
        return None, 'the reply is not text'
    fenced = re.fullmatch(r'\s*' + _FENCE + r'json\s*\n(.*?)\n' + _FENCE + r'\s*', reply, re.DOTALL)
    try:
        value = json.loads(fenced.group(1) if fenced else reply, object_pairs_hook=unique_fields)
    except ValueError:
        return None, 'expected a JSON object'
    if not isinstance(value, dict) or set(value) != {'direction_deg', 'speed_px_per_frame'}:
        return None, 'expected exactly direction_deg and speed_px_per_frame'
    for name, number in value.items():
        if isinstance(number, bool) or not isinstance(number, (int, float)) or not math.isfinite(number):
            return None, f'{name} must be a finite number'
    if not 0 <= value['direction_deg'] < 360:
        return None, 'direction_deg must be in [0, 360)'
    if value['speed_px_per_frame'] < 0:
        return None, 'speed_px_per_frame must not be negative'
    return value, None


def unflip_vector(direction: float, speed: float, flip: int) -> tuple[float, float]:
    """A report (counterclockwise from right, y up) as an image vector (x right, y down), unflipped.

    Flip 2 mirrors x, flip 3 mirrors y, flip 4 mirrors both (the study's flip types).
    """
    if flip not in (1, 2, 3, 4):
        raise ValueError('flip must be 1, 2, 3 or 4')
    angle = math.radians(direction)
    x, y = speed * math.cos(angle), -speed * math.sin(angle)
    return (-x if flip in (2, 4) else x), (-y if flip in (3, 4) else y)


def endpoint_errors(vectors: dict, groups: dict, reference) -> np.ndarray:
    """Endpoint error of each group mean against each participant (groups x participants).

    ``vectors`` maps a stimulus id to its unflipped (x, y); ``groups`` maps a reference
    stimulus id (a scene and modulation) to its four flip stimulus ids.
    """
    ids = [str(i) for i in reference['stimulus_id'].values]
    human = reference.transpose('presentation', 'participant', 'axis').values
    model = np.array([np.mean([vectors[s] for s in groups[i]], axis=0) for i in ids])
    return np.linalg.norm(model[:, None, :] - human, axis=-1)
