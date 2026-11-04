"""Score the same laboratory procedure through the registered benchmark interface."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import brainscore
from brainscore_core import BrainScoreModel
from brainscore_core.compatibility import CompatibilityError
from brainscore_core.score_status import score_or_na
from brainscore_core.supported_data_standards.brainio.assemblies import BehavioralAssembly
from brainscore_core.supported_data_standards.brainio.stimuli import StimulusSet
from brainscore.experiments import Experiment, ExperimentResult, RecordBehavior, RecordNeural
from brainscore.experiments.sun2025 import analyze, make_protocol
from brainscore.benchmarks.sun2025 import IDENTIFIER
from brainscore.benchmarks.sun2025.benchmark import Sun2025MotionBenchmark
from brainscore.benchmarks.sun2025.metric import motion_vector_error

pytestmark = pytest.mark.unit


def study(scenes=(1,), modulations=(1,)):
    rows, groups = [], []
    for scene in scenes:
        for modulation in modulations:
            base = f's{scene}-m{modulation}'
            groups.append((base, scene, modulation))
            for flip in (1, 2, 3, 4):
                rows.append(dict(stimulus_id=f'{base}-f{flip}', base_id=base,
                                 scene=scene, modulation=modulation, flip=flip,
                                 video_path=f'{base}-f{flip}.avi'))
    stimuli = StimulusSet(rows)
    stimuli.identifier = 'synthetic-sun'
    stimuli.stimulus_paths = dict(zip(stimuli.stimulus_id, stimuli.video_path))
    reference = BehavioralAssembly(
        np.tile([3., -4.], (len(groups), 3, 1)), dims=['presentation', 'participant', 'axis'],
        coords={'stimulus_id': ('presentation', [group[0] for group in groups]),
                'scene': ('presentation', [group[1] for group in groups]),
                'modulation': ('presentation', [group[2] for group in groups]),
                'participant': [1, 2, 3], 'axis': ['x', 'y']},
        attrs={'identifier': 'synthetic-human', 'provenance': {'complete': len(rows) == 1120}})
    return stimuli, reference


def subject(reply=None, *, modality='video'):
    # A deliberately blind control: declaring clip input never implies using its pixels.
    return BrainScoreModel(identifier='sun-no-motion-control', supported_modalities={modality},
                           reply=reply or (lambda prompt: json.dumps(
                               {'direction_deg': 0, 'speed_px_per_frame': 0})))


def measured(stimuli, *, bad_id=None, invalid_id=None):
    replies = [json.dumps({'direction_deg': 0, 'speed_px_per_frame': 0})
               if sid != bad_id else 'no numerical report' for sid in stimuli.stimulus_id]
    return BehavioralAssembly(replies, dims=['presentation'], coords={
        'stimulus_id': ('presentation', list(stimuli.stimulus_id)),
        'scene': ('presentation', list(stimuli.scene)),
        'raw_answer': ('presentation', replies),
        'answer_valid': ('presentation', [sid != invalid_id for sid in stimuli.stimulus_id])})


def test_registered_benchmark_runs_full_design_through_score(monkeypatch, tmp_path):
    stimuli, reference = study(range(1, 41), range(1, 8))
    calls = []

    def load_stimuli(identifier, **kwargs):
        calls.append(('stimuli', identifier))
        return stimuli

    def load_reference(identifier, **kwargs):
        calls.append(('reference', identifier))
        return reference

    monkeypatch.setattr(brainscore, 'load_stimulus_set', load_stimuli)
    monkeypatch.setattr(brainscore, 'load_dataset', load_reference)
    benchmark = brainscore.load_benchmark(IDENTIFIER, output_dir=tmp_path / 'run')
    control_score = brainscore.score(subject(), benchmark, check_mem=False)
    assert benchmark.identifier == IDENTIFIER and benchmark.version == 1
    assert float(control_score) == 5
    assert calls == [('stimuli', 'Sun2025-second-order-motion'),
                     ('reference', 'Sun2025-second-order-motion')]
    assert control_score.attrs['full_study'] is True
    assert control_score.attrs['higher_is_better'] is False
    assert control_score.attrs['normalized'] is False
    assert control_score.attrs['summary']['angular_comparisons'] == 0
    assert control_score.attrs['compatibility']['match'] == 'full'
    assert control_score.attrs['received_trials'] == 1120
    assert control_score.attrs['records_retained'] is True
    assert (tmp_path / 'run' / 'experiment.json').exists()


def test_same_protocol_and_recorders_match_direct_experiment(tmp_path):
    stimuli, reference = study(modulations=(1, 2))
    protocol = make_protocol(stimuli)
    control = subject()
    experiment = Experiment(subject=control, protocol=protocol, tools=[RecordBehavior()],
                            output_dir=tmp_path / 'direct').run()
    measurements = experiment.behavior(measure='text')
    expected = analyze(measurements, stimuli, reference)[2]['mean_endpoint_error_px_per_frame']
    provenance = {'revision': 'synthetic-test-revision', 'dtype': 'float64'}
    benchmark = Sun2025MotionBenchmark(protocol, reference,
                                       metadata={'execution_provenance': provenance})
    assert benchmark.protocol is protocol
    assert benchmark.reference is reference
    assert benchmark.metric is motion_vector_error
    assert benchmark.requested_output_channels == frozenset({'behavior:text'})
    assert float(benchmark.score_measurements(measurements)) == expected
    score = brainscore.score(control, benchmark, check_mem=False)
    assert float(score) == expected
    assert score.attrs['experiment_directory'] is None and not score.attrs['records_retained']
    assert score.attrs['execution_provenance'] == provenance
    assert score.attrs['experiment_provenance']['plan']['metadata']['execution_provenance'] == provenance
    assert benchmark.identifier.startswith(IDENTIFIER + '-custom-')
    assert not score.attrs['full_study']


def test_reference_and_measurement_order_do_not_change_score():
    stimuli, reference = study(modulations=(1, 2))
    protocol = make_protocol(stimuli)
    a = Sun2025MotionBenchmark(protocol, reference)
    b = Sun2025MotionBenchmark(protocol, reference.isel(presentation=[1, 0]))
    assert a.identifier == b.identifier
    behavior = measured(stimuli)
    assert float(a.score_measurements(behavior)) == float(
        b.score_measurements(behavior.isel(presentation=slice(None, None, -1))))


def test_optional_neural_instrument_records_the_same_presentations(tmp_path):
    from brainscore_core.selection import LayerSelector
    from tests.test_sun2025 import Net, Extractor
    stimuli, reference = study()
    net = Net()

    def reply(prompt):
        net.forward(prompt.text())
        return '{"direction_deg": 0, "speed_px_per_frame": 0}'

    candidate = BrainScoreModel(identifier='recorded-video-control',
                               extractor={'video': Extractor(net)}, reply=reply,
                               region_layer_map={'V1': LayerSelector('early', modality='video')})
    instruments = [RecordBehavior(), RecordNeural(['V1'])]
    benchmark = Sun2025MotionBenchmark(make_protocol(stimuli), reference, recorders=instruments,
                                       output_dir=tmp_path / 'joint')
    score = brainscore.score(candidate, benchmark, check_mem=False)
    recording = ExperimentResult(tmp_path / 'joint', None)
    assert float(score) == 5 and net.passes == 4
    assert recording.neural('V1').sizes['presentation'] == 4
    assert list(recording.neural('V1')['stimulus_id'].values) == list(
        recording.behavior(measure='text')['stimulus_id'].values)


@pytest.mark.parametrize('invalid_kind', ['parser', 'subject'])
def test_invalid_report_cannot_improve_score_by_dropping_a_group(invalid_kind):
    stimuli, reference = study(modulations=(1, 2))
    identifier = stimuli.stimulus_id.iloc[0]
    behavior = measured(stimuli, **{('bad_id' if invalid_kind == 'parser' else 'invalid_id'): identifier})
    score = Sun2025MotionBenchmark(make_protocol(stimuli), reference).score_measurements(behavior)
    assert np.isnan(float(score))
    assert score.attrs['status'] == 'failed' and score.attrs['reason'] == 'invalid_motion_reports'
    assert score.attrs['valid_answers'] == 7
    assert score.attrs['summary']['complete_groups'] == 1
    assert int(score.attrs['reports']['valid'].values.sum()) == 7
    assert len(score.attrs['measurements']) == 8


def test_missing_reports_return_failure_not_a_partial_score():
    stimuli, reference = study()
    behavior = measured(stimuli).isel(presentation=[0, 1, 2])
    score = Sun2025MotionBenchmark(make_protocol(stimuli), reference).score_measurements(behavior)
    assert np.isnan(float(score)) and score.attrs['reason'] == 'missing_reports'
    assert score.attrs['missing_stimulus_ids'] == [stimuli.stimulus_id.iloc[3]]


def test_incompatible_subject_is_na_without_inference_or_output(tmp_path):
    stimuli, reference = study()
    seen = []
    candidate = subject(lambda prompt: seen.append(prompt), modality='vision')
    benchmark = Sun2025MotionBenchmark(make_protocol(stimuli), reference, output_dir=tmp_path / 'run')
    with pytest.raises(CompatibilityError):
        benchmark(candidate)
    score = score_or_na(brainscore.score, candidate, benchmark, check_mem=False)
    assert np.isnan(float(score)) and score.attrs['status'] == 'N/A'
    assert not seen and not (tmp_path / 'run').exists()


def test_full_registry_rejects_pilot_data_before_inference(monkeypatch):
    stimuli, reference = study()
    monkeypatch.setattr(brainscore, 'load_stimulus_set', lambda *args, **kwargs: stimuli)
    monkeypatch.setattr(brainscore, 'load_dataset', lambda *args, **kwargs: reference)
    with pytest.raises(ValueError, match='complete 1120-trial study'):
        brainscore.load_benchmark(IDENTIFIER)


def test_reference_mismatch_and_ambiguous_design_fail_before_compute():
    stimuli, reference = study()
    wrong = reference.assign_coords(scene=('presentation', [99]))
    with pytest.raises(ValueError, match='scene coordinates'):
        Sun2025MotionBenchmark(make_protocol(stimuli), wrong)
    with pytest.raises(ValueError, match='four distinct flips'):
        Sun2025MotionBenchmark(make_protocol(stimuli.iloc[:3].copy()), reference)
    with pytest.raises(ValueError, match='one condition and one repetition'):
        Sun2025MotionBenchmark(make_protocol(stimuli, repetitions=(0, 1)), reference)
    with pytest.raises(ValueError, match='Attach RecordBehavior'):
        Sun2025MotionBenchmark(make_protocol(stimuli), reference, recorders=[RecordNeural(['V1'])])


def test_exact_human_vectors_score_zero_and_reset_lifecycle_is_unchanged(tmp_path):
    stimuli, reference = study()
    calls = []
    angle = np.degrees(np.arctan2(4, 3))
    directions = {1: angle, 2: 180-angle, 3: 360-angle, 4: 180+angle}

    def oracle(prompt):
        calls.append(prompt.stimulus_id)
        return json.dumps({'direction_deg': directions[int(prompt.stimulus_id[-1])],
                           'speed_px_per_frame': 5})

    candidate = subject(oracle)
    reset = candidate.reset
    resets = []

    def count_reset():
        resets.append(True)
        reset()

    candidate.reset = count_reset
    benchmark = Sun2025MotionBenchmark(make_protocol(stimuli), reference, output_dir=tmp_path / 'oracle')
    score = brainscore.score(candidate, benchmark, check_mem=False)
    assert float(score) < 1e-12 and len(calls) == 4 and len(resets) == 2


def test_saved_real_pilot_matches_previous_analysis():
    # Recorded model reports, not a new model inference or synthetic scientific result.
    directory = Path(__file__).resolve().parents[1] / 'examples/sun2025/validation'
    reports = pd.read_csv(directory / 'motion-reports.csv', keep_default_na=False)
    comparison = pd.read_csv(directory / 'human-comparison.csv')
    stimuli = StimulusSet(reports[['stimulus_id', 'base_id', 'scene', 'modulation', 'flip']].copy())
    stimuli['video_path'] = stimuli.stimulus_id + '.avi'
    stimuli.identifier = 'recorded-pilot-validation'
    stimuli.stimulus_paths = dict(zip(stimuli.stimulus_id, stimuli.video_path))
    reference_rows = comparison.sort_values(['base_id', 'participant'])
    groups = reference_rows.drop_duplicates('base_id')
    reference = BehavioralAssembly(
        reference_rows[['human_x', 'human_y']].to_numpy().reshape(7, 3, 2),
        dims=['presentation', 'participant', 'axis'],
        coords={'stimulus_id': ('presentation', groups.base_id.to_numpy()),
                'scene': ('presentation', groups.scene.to_numpy()),
                'modulation': ('presentation', groups.modulation.to_numpy()),
                'participant': [1, 2, 3], 'axis': ['x', 'y']})
    behavior = BehavioralAssembly(reports.raw_answer.to_numpy(), dims=['presentation'], coords={
        'stimulus_id': ('presentation', reports.stimulus_id.to_numpy()),
        'scene': ('presentation', reports.scene.to_numpy())})
    benchmark = Sun2025MotionBenchmark(make_protocol(stimuli), reference)
    score = benchmark.score_measurements(behavior)
    saved = json.loads((directory / 'summary.json').read_text())
    assert float(score) == pytest.approx(saved['mean_endpoint_error_px_per_frame'], rel=0, abs=1e-12)
    assert score.attrs['valid_answers'] == 28
