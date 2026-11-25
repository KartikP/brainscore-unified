"""A face localizer for a study model: its face-selective units in one region, as an fMRI localizer finds
them. ``FunctionalSelection`` keeps the units with the highest Cohen's d of faces against the other
categories on one half of the localizer pictures; the other half tests them. ``RandomSelection.matched_to``
gives the matched control: as many units, drawn at random from the same layer.

The rule (one rule, set before any look at a task): the top ``N_UNITS`` units by d; the select half is
the pictures with an odd ``number``, the test half those with an even one (half of each subcategory in
each). This is a new named selection, beside the study's depth table; the table does not change.

    from brainscore import load_stimulus_set
    result = face_localizer(subject, load_stimulus_set('fLoc'))
    result['units'].indices, result['d_test'][result['units'].indices].mean()
"""
from typing import Optional

import numpy as np

N_UNITS = 50          # the selected units of the region's layer
SEED = 0              # the seed of the matched random units


def unit_d(subject, stimuli, region: str, positive, column: str = 'category') -> np.ndarray:
    """Cohen's d of each recorded unit of ``region``: pictures whose ``column`` is in ``positive`` against
    all others (the pooled standard deviation, as ``FunctionalSelection``). One recording, one layer."""
    from brainscore_core.streaming_helpers import neural_response
    recorded = neural_response(subject, stimuli, [f'neural:{region}'])
    extra = [dim for dim in recorded.dims if dim not in ('presentation', 'neuroid')]
    recorded = (recorded.mean(extra) if extra else recorded).transpose('presentation', 'neuroid')
    label_of = dict(zip(stimuli['stimulus_id'].values, stimuli[column].values))
    faces = np.isin([label_of[sid] for sid in recorded['stimulus_id'].values], list(positive))
    if faces.all() or not faces.any():
        raise ValueError(f'{column} {list(positive)} must match some pictures and not all of them.')
    values = recorded.values
    pos, neg = values[faces], values[~faces]
    pooled = np.sqrt((pos.std(0, ddof=1) ** 2 + neg.std(0, ddof=1) ** 2) / 2)
    return np.divide(pos.mean(0) - neg.mean(0), pooled, out=np.zeros(values.shape[1]), where=pooled > 0)


def face_localizer(subject, stimuli, region: str = 'FFA', faces=('faces',), column: str = 'category',
                   n_units: int = N_UNITS, seed: int = SEED, cross_check: Optional[tuple] = None) -> dict:
    """The face-selective units of ``region``: ``units`` (top ``n_units`` d on the select half), ``control``
    (the matched random units), ``d_select`` and ``d_test`` (each unit's d on each half). With
    ``cross_check=(other stimuli, its face labels, its column)``, also ``d_cross``: each unit's d there."""
    from brainscore import FunctionalSelection, RandomSelection
    odd = stimuli['number'].values % 2 == 1
    select, test = stimuli[odd], stimuli[~odd]
    for half, name in ((select, 'select'), (test, 'test')):
        half.stimulus_paths = stimuli.stimulus_paths
        half.identifier = f'{stimuli.identifier}-{name}'
    negative = sorted(set(stimuli[column]) - set(faces))
    units = FunctionalSelection(region, select, contrast=(list(faces), negative), contrast_column=column,
                                n_units=n_units).resolve(subject)
    result = {'units': units, 'control': RandomSelection.matched_to(units, seed=seed).resolve(subject),
              'd_select': unit_d(subject, select, region, faces, column),
              'd_test': unit_d(subject, test, region, faces, column)}
    if cross_check is not None:
        other, other_faces, other_column = cross_check
        result['d_cross'] = unit_d(subject, other, region, other_faces, other_column)
    return result
