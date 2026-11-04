"""Compare measured motion reports with human judgments, after the experiment."""
from brainscore_core import Score
from brainscore_core.supported_data_standards.brainio.assemblies import BehavioralAssembly
from brainscore.experiments.sun2025.analysis import analyze


def motion_vector_error(behavior: BehavioralAssembly, stimuli,
                        reference: BehavioralAssembly) -> Score:
    """Mean endpoint error, requiring every trial; lower is closer to humans.

    Direction and speed are converted to a vector, spatial flips are reversed,
    and the four vectors in each scene/modulation group are averaged. The error
    is then averaged equally over groups and human participants. Invalid answers
    never improve the headline score by removing a difficult comparison group.
    """
    expected = set(stimuli.stimulus_id.astype(str))
    received = list(behavior['stimulus_id'].values.astype(str))
    if len(received) != len(set(received)) or set(received) - expected:
        raise ValueError('Measurements contain duplicate or unexpected stimulus identities')
    missing = sorted(expected - set(received))
    score = Score(float('nan'))
    score.attrs.update(metric='mean_vector_endpoint_error', units='displayed pixels per frame',
                       higher_is_better=False, normalization='raw', normalized=False,
                       ceiling_status='not_estimated', measurements=behavior,
                       expected_trials=len(stimuli), received_trials=len(received),
                       missing_stimulus_ids=missing)
    if missing:
        score.attrs.update(status='failed', reason='missing_reports')
        return score

    trials, comparisons, summary = analyze(behavior, stimuli, reference)
    complete = bool(trials.valid.all())
    value = summary['mean_endpoint_error_px_per_frame'] if complete else float('nan')
    score = Score(value, attrs=score.attrs)
    score.attrs.update(
        status='scored' if complete else 'failed',
        reason=None if complete else 'invalid_motion_reports',
        valid_answers=summary['valid_answers'], summary=summary,
        raw=Score(value),
        reports=BehavioralAssembly(
            trials.raw_answer.astype(str).to_numpy(), dims=['presentation'],
            coords={name: ('presentation', trials[name].to_numpy()) for name in
                    ('stimulus_id', 'base_id', 'scene', 'modulation', 'flip',
                     'valid', 'parse_error', 'response_format')}),
        participant_errors=Score(
            comparisons.endpoint_error_px_per_frame.to_numpy(), dims=['comparison'],
            coords={name: ('comparison', comparisons[name].to_numpy()) for name in
                    ('base_id', 'scene', 'modulation', 'participant', 'valid',
                     'angular_error_deg', 'speed_error_px_per_frame')}),
    )
    return score
