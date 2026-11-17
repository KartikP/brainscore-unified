"""Read a motion report and compare unflipped model vectors with the three participants.

This metric stays inside the benchmark: the flip reversal and the four-flip mean
belong to this study's design, and no other benchmark uses them.
"""
import json
import math
import re

import numpy as np

_FENCE = chr(96) * 3
# The largest speed the metric takes. The clip is 600 pixels wide, so a speed above this is not
# a report of the clip; the limit also keeps every mean and norm far from overflow.
MAX_SPEED_PX_PER_FRAME = 1e6


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
        if isinstance(number, bool) or not isinstance(number, (int, float)):
            return None, f'{name} must be a number'
        try:
            if not math.isfinite(number):
                return None, f'{name} must be a finite number'
        except OverflowError:       # an integer too large for a float
            return None, f'{name} is too large'
    if not 0 <= value['direction_deg'] < 360:
        return None, 'direction_deg must be in [0, 360)'
    if not 0 <= value['speed_px_per_frame'] <= MAX_SPEED_PX_PER_FRAME:
        return None, f'speed_px_per_frame must be in [0, {MAX_SPEED_PX_PER_FRAME:g}]'
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


def endpoint_errors(vectors: dict, groups: dict, reference) -> list[tuple[str, int, float]]:
    """Endpoint error of each group mean against each participant, as (group, participant, error).

    ``vectors`` maps a stimulus id to its unflipped (x, y); ``groups`` maps a reference
    stimulus id (a scene and modulation) to its four flip stimulus ids, in trial order.
    The rows are in sorted group order, then participant order. Each group mean and each
    norm is computed as in the original implementation (pull request #63), so the values
    are equal to the last digit.
    """
    import pandas as pd
    ids = [str(i) for i in reference['stimulus_id'].values]
    human = reference.transpose('presentation', 'participant', 'axis')
    participants = [int(p) for p in human['participant'].values]
    rows = []
    for group in sorted(groups):
        model = pd.DataFrame([vectors[s] for s in groups[group]], columns=['x', 'y']).mean().to_numpy()
        for index, participant in enumerate(participants):
            rows.append((group, participant,
                         float(np.linalg.norm(model - human.values[ids.index(group), index]))))
    return rows
