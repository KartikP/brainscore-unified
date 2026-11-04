"""Compare unflipped model motion vectors with the three human participants."""
import math

import numpy as np
import pandas as pd

from .experiment import REPORT_PARSER_VERSION, parse_report, report_format


def unflip_vector(direction: float, speed: float, flip: int) -> tuple[float, float]:
    if flip not in (1, 2, 3, 4):
        raise ValueError('flip must be 1, 2, 3 or 4')
    angle = math.radians(direction)
    x, y = speed * math.cos(angle), -speed * math.sin(angle)
    return (-x if flip in (2, 4) else x), (-y if flip in (3, 4) else y)


def analyze(behavior, stimuli, reference) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Align by identity; a comparison needs all four valid flips, with no imputation."""
    ids = list(behavior['stimulus_id'].values.astype(str))
    expected = list(stimuli.stimulus_id.astype(str))
    if len(ids) != len(set(ids)) or len(expected) != len(set(expected)) or set(ids) != set(expected):
        raise ValueError('Behavior must contain every selected stimulus exactly once')
    raw = list(behavior['raw_answer'].values) if 'raw_answer' in behavior.coords else list(behavior.values)
    supplied_valid = list(behavior['valid'].values) if 'valid' in behavior.coords else [True] * len(ids)
    replies = dict(zip(ids, zip(raw, supplied_valid)))
    trials = []
    for row in stimuli.itertuples(index=False):
        answer, valid = replies[row.stimulus_id]
        parsed, error = parse_report(answer)
        if not bool(valid) and error is None:
            error = 'response was marked invalid by subject'
        x, y = (unflip_vector(parsed['direction_deg'], parsed['speed_px_per_frame'], int(row.flip))
                if error is None else (np.nan, np.nan))
        trials.append(dict(stimulus_id=row.stimulus_id, base_id=row.base_id, scene=int(row.scene),
                           modulation=int(row.modulation), flip=int(row.flip), raw_answer=answer,
                           valid=error is None, parse_error=error or '',
                           response_format=report_format(answer)[1] if error is None else 'invalid',
                           x=x, y=y))
    trials = pd.DataFrame(trials)
    ref_ids = list(reference['stimulus_id'].values.astype(str))
    if len(ref_ids) != len(set(ref_ids)) or set(ref_ids) != set(stimuli.base_id):
        raise ValueError('Human reference identities do not match the selected scene/modulation groups')
    reference_values = reference.transpose('presentation', 'participant', 'axis')
    if list(reference_values['axis'].values) != ['x', 'y']:
        raise ValueError('Reference axes must be x, y')
    lookup = {sid: index for index, sid in enumerate(ref_ids)}
    comparisons, complete = [], 0
    for base_id, group in trials.groupby('base_id', sort=True):
        if sorted(group.flip) != [1, 2, 3, 4]:
            raise ValueError('Each comparison needs the four distinct flip conditions')
        valid = bool(group.valid.all())
        complete += int(valid)
        model = group[['x', 'y']].mean().to_numpy() if valid else np.array([np.nan, np.nan])
        for participant_index, participant in enumerate(reference_values['participant'].values):
            human = reference_values.values[lookup[base_id], participant_index]
            model_speed, human_speed = np.linalg.norm(model), np.linalg.norm(human)
            angle_error = np.nan
            if valid and model_speed > 1e-12 and human_speed > 1e-12:
                cosine = np.dot(model, human) / (model_speed * human_speed)
                angle_error = math.degrees(math.acos(float(np.clip(cosine, -1, 1))))
            comparisons.append(dict(base_id=base_id, scene=int(group.scene.iloc[0]),
                                    modulation=int(group.modulation.iloc[0]), participant=int(participant),
                                    valid=valid, model_x=model[0], model_y=model[1],
                                    human_x=human[0], human_y=human[1],
                                    angular_error_deg=angle_error,
                                    speed_error_px_per_frame=abs(model_speed - human_speed),
                                    endpoint_error_px_per_frame=np.linalg.norm(model - human)))
    comparisons = pd.DataFrame(comparisons)

    def mean(column):
        values = comparisons.loc[comparisons.valid, column].dropna()
        return float(values.mean()) if len(values) else None

    summary = {'status': 'complete' if bool(trials.valid.all()) else 'incomplete_answers',
               'trials': len(trials), 'valid_answers': int(trials.valid.sum()),
               'comparison_groups': len(ref_ids), 'complete_groups': complete,
               'participants': int(reference.sizes['participant']),
               'mean_angular_error_deg': mean('angular_error_deg'),
               'angular_comparisons': int(comparisons.angular_error_deg.notna().sum()),
               'mean_speed_error_px_per_frame': mean('speed_error_px_per_frame'),
               'mean_endpoint_error_px_per_frame': mean('endpoint_error_px_per_frame'),
               'aggregation': 'mean over participants and complete scene/modulation groups',
               'normalization': 'none; descriptive comparison',
               'report_parser': REPORT_PARSER_VERSION,
               'json_code_fence_answers': int((trials.response_format == 'json_code_fence').sum()),
               'reference_build': reference.attrs.get('identifier'),
               'full_study': bool(reference.attrs.get('provenance', {}).get('complete', False))}
    return trials, comparisons, summary
