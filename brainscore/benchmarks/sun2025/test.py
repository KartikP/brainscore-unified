"""Sun2025 motion benchmark through participate, on a made-up prepared build."""
import json
import math

import numpy as np
import pytest

import brainscore
from brainscore_core.contract import Subject
from brainscore_core.streaming import StreamEvent
from brainscore.benchmarks.sun2025.metric import parse_report, unflip_vector
from tests.test_data.test_sun2025_data import HUMAN, build  # noqa: F401  (the made-up build fixture)

pytestmark = pytest.mark.unit

NO_MOTION = '{"direction_deg": 0, "speed_px_per_frame": 0}'


def _report(x, y, flip):
    """The report whose unflipped vector is (x, y)."""
    fx, fy = (-x if flip in (2, 4) else x), (-y if flip in (3, 4) else y)
    return json.dumps({'direction_deg': math.degrees(math.atan2(-fy, fx)) % 360,
                       'speed_px_per_frame': math.hypot(fx, fy)})


class Reporter(Subject):
    """Answers each clip with ``answer(stimulus_id)`` as text."""
    identifier = 'made-up-reporter'
    in_channels = {'vision:video', 'instruction'}
    out_channels = {'behavior:text'}

    def __init__(self, answer):
        self.answer = answer

    def participate(self, session):
        while (event := session.next_input()) is not None:
            if event.channel == 'vision:video':
                session.respond(StreamEvent('behavior:text', self.answer(event.meta['stimulus_id']),
                                            t_ms=event.t_ms, meta=dict(event.meta)))

    def reset(self):
        pass


def _exact(stimulus_id):
    modulation, flip = int(stimulus_id.split('-m')[1][0]), int(stimulus_id[-1])
    return _report(*HUMAN[modulation], flip)


def _benchmark(build):
    return brainscore.load_benchmark('Sun2025-motion-vector-error', root=build, scenes=[1])


def test_declarations(build):
    benchmark = _benchmark(build)
    assert {'vision:video', 'instruction'} <= set(benchmark.required_input_channels)
    assert 'behavior:text' in benchmark.requested_output_channels
    assert benchmark.trial_memory == 'none' and benchmark.task.trial_memory == 'none'


def test_a_full_run_scores_against_the_participants_with_the_floor(build):
    benchmark = _benchmark(build)
    exact = brainscore.score(Reporter(_exact), benchmark)
    still = brainscore.score(Reporter(lambda _: NO_MOTION), benchmark)
    floor = np.mean([math.hypot(*vector) for vector in HUMAN.values()])
    assert float(exact) == pytest.approx(0, abs=1e-9)
    assert float(still) == pytest.approx(floor)
    assert float(exact.attrs['floor']) == pytest.approx(floor)
    assert exact.attrs['n_trials'] == 8 and exact.attrs['n_valid'] == 8
    assert exact.attrs['normalized'] is False and exact.attrs['higher_is_better'] is False
    assert exact.attrs['leaderboard_eligible'] is False      # storage refuses the score
    # A selection of scenes is named as a subset and has its own identity.
    assert exact.attrs['full_study'] is False and exact.attrs['subset']
    assert exact.attrs['benchmark_identifier'].startswith('Sun2025-motion-vector-error-subset-')
    assert any('motion matching' in line for line in exact.attrs['adaptation'])


def test_an_unusable_answer_is_kept_with_its_raw_text_and_counted(build):
    bad = 'sun2025-s01-m2-v1-f3'
    score = brainscore.score(Reporter(lambda sid: 'It moves left.' if sid == bad else _exact(sid)),
                             _benchmark(build))
    reports = score.attrs['reports']
    row = list(reports['stimulus_id'].values).index(bad)
    assert reports['response_state'].values[row] == 'invalid'
    assert reports.values[row] == 'It moves left.'
    assert score.attrs['n_invalid'] == 1 and score.attrs['n_valid'] == 7
    assert np.isnan(float(score)) and score.attrs['result_validity'] == 'invalid'


def test_a_subject_with_no_video_input_is_na_before_compute(build):
    class ImageOnly(Reporter):
        in_channels = {'vision:image', 'instruction'}

        def participate(self, session):
            raise AssertionError('participate must not run for an N/A pair')

    score = brainscore.score(ImageOnly(_exact), _benchmark(build))
    assert score.attrs['status'] == 'N/A'


@pytest.mark.parametrize('reply, usable', [
    ('{"direction_deg": 90, "speed_px_per_frame": 2.5}', True),
    ('```json\n{"direction_deg": 90, "speed_px_per_frame": 2.5}\n```', True),
    ('{"direction_deg": 360, "speed_px_per_frame": 1}', False),
    ('{"direction_deg": 10, "speed_px_per_frame": -1}', False),
    ('{"direction_deg": 10, "speed_px_per_frame": 1, "note": 1}', False),
    ('{"direction_deg": true, "speed_px_per_frame": 1}', False),
    ('Answer: {"direction_deg": 10, "speed_px_per_frame": 1}', False),
])
def test_the_parser_takes_only_the_two_numbers(reply, usable):
    assert (parse_report(reply)[1] is None) == usable


def test_unflip_reverses_each_flip():
    # Up (90 degrees) is -y in image coordinates; flips 3 and 4 mirror y, flips 2 and 4 mirror x.
    assert unflip_vector(90, 1, 1) == pytest.approx((0, -1))
    assert unflip_vector(90, 1, 3) == pytest.approx((0, 1))
    assert unflip_vector(0, 1, 2) == pytest.approx((-1, 0))


def _design(human):
    """An in-memory benchmark: group g has scene 1, modulation g + 1 and four flips; human is groups x 3 x 2."""
    from brainscore.benchmarks.sun2025.benchmark import Sun2025MotionBenchmark
    return Sun2025MotionBenchmark(*_parts(human))


def _parts(human):
    """The stimuli and the reference of ``_design``."""
    from brainscore_core.supported_data_standards.brainio.assemblies import BehavioralAssembly
    from brainscore_core.supported_data_standards.brainio.stimuli import StimulusSet
    groups = range(len(human))
    stimuli = StimulusSet([dict(stimulus_id=f'g{g}-f{f}', base_id=f'g{g}', scene=1, modulation=g + 1,
                                flip=f, video_path=f'g{g}-f{f}.avi') for g in groups for f in (1, 2, 3, 4)])
    stimuli.identifier = 'made-up-sun'
    reference = BehavioralAssembly(
        np.array(human, dtype=float), dims=['presentation', 'participant', 'axis'],
        coords={'stimulus_id': ('presentation', [f'g{g}' for g in groups]),
                'scene': ('presentation', [1] * len(human)),
                'modulation': ('presentation', [g + 1 for g in groups]),
                'participant': [1, 2, 3], 'axis': ['x', 'y']})
    return stimuli, reference


def test_the_score_equals_the_old_implementation():
    # Case 15 of np.random.default_rng(2025) (uniform(0, 360, 8), uniform(0, 10, 8), normal(0, 3, (2, 3, 2)))
    # in the review of this port. 4.255378301966291 is the value of the old implementation
    # (pull request #63, head 1ae58a1a, experiments/sun2025/analysis.py::analyze) on these answers.
    directions = [147.55655232248634, 10.054686038515715, 131.89316696603308, 14.840835607808206,
                  317.75805914843266, 290.6338338446624, 162.41275523165345, 266.2633233415828]
    speeds = [9.945997580525798, 2.2990268592918985, 8.697693238582275, 3.2021590327784,
              4.59102398750537, 8.86262504666291, 3.7265614455438656, 9.705307474917234]
    human = [[[2.1094195984569146, -1.5147653974728494], [0.12047963306372184, -4.029855323217056],
              [-1.243139986837792, 1.3765951199568947]],
             [[-3.964102084273672, -0.8564605166536496], [-0.5824091662970251, 1.891242993019418],
              [0.1751960004616196, -2.1516355477717526]]]
    answers = {f'g{g}-f{f}': json.dumps({'direction_deg': directions[4 * g + f - 1],
                                         'speed_px_per_frame': speeds[4 * g + f - 1]})
               for g in (0, 1) for f in (1, 2, 3, 4)}
    score = brainscore.score(Reporter(answers.__getitem__), _design(human))
    # The last digit depends on the processor's order of sums (seen on the hosted checks: ...291 and ...292).
    assert float(score) == pytest.approx(4.255378301966291, rel=1e-12)


def test_a_number_outside_the_supported_range_is_an_invalid_answer():
    assert parse_report('{"direction_deg": 10, "speed_px_per_frame": 1' + '0' * 400 + '}')[1]
    assert parse_report('{"direction_deg": 10, "speed_px_per_frame": 1e400}')[1]
    huge = '{"direction_deg": 0, "speed_px_per_frame": 1e308}'
    benchmark = _design([[[1.0, 0.0]] * 3])
    score = brainscore.score(Reporter(lambda sid: huge if sid == 'g0-f1' else NO_MOTION), benchmark)
    reports = score.attrs['reports']
    row = list(reports['stimulus_id'].values).index('g0-f1')
    assert reports['response_state'].values[row] == 'invalid' and reports.values[row] == huge
    assert score.attrs['n_invalid'] == 1 and score.attrs['result_validity'] == 'invalid'


TWO_GROUPS = [[[0.0, 0.0]] * 3, [[10.0, 0.0], [9.0, 1.0], [11.0, -1.0]]]


def test_a_reference_that_does_not_match_the_stimuli_is_refused():
    from brainscore.benchmarks.sun2025.benchmark import Sun2025MotionBenchmark
    stimuli, reference = _parts(TWO_GROUPS)
    swapped = stimuli.copy()
    swapped['base_id'] = swapped.base_id.map({'g0': 'g1', 'g1': 'g0'})
    with pytest.raises(ValueError, match='reference modulation'):
        Sun2025MotionBenchmark(swapped, reference)
    from brainscore_core.supported_data_standards.brainio.assemblies import BehavioralAssembly
    wrong_scenes = BehavioralAssembly(
        reference.values, dims=['presentation', 'participant', 'axis'],
        coords={'stimulus_id': ('presentation', ['g0', 'g1']), 'scene': ('presentation', [99, 98]),
                'modulation': ('presentation', [1, 2]), 'participant': [1, 2, 3], 'axis': ['x', 'y']})
    with pytest.raises(ValueError, match='reference scene'):
        Sun2025MotionBenchmark(stimuli, wrong_scenes)
    with pytest.raises(ValueError, match='participants'):
        Sun2025MotionBenchmark(stimuli, reference.assign_coords(participant=[1, 2, 4]))


def test_a_subset_has_one_identity_whatever_the_order():
    from brainscore.benchmarks.sun2025.benchmark import Sun2025MotionBenchmark
    stimuli, reference = _parts(TWO_GROUPS + [[[1.0, 2.0]] * 3, [[3.0, 4.0]] * 3])
    first = Sun2025MotionBenchmark(stimuli, reference).identifier
    reordered = stimuli.iloc[::-1].reset_index(drop=True)
    reordered.identifier = stimuli.identifier
    transposed = reference.isel(presentation=[3, 1, 0, 2]).transpose('participant', 'axis', 'presentation')
    assert Sun2025MotionBenchmark(reordered, transposed).identifier == first
    changed = reference.copy(data=reference.values + 1)
    assert Sun2025MotionBenchmark(stimuli, changed).identifier != first


def test_the_registered_identifier_needs_the_full_design(build):
    manifest = json.loads((build / 'manifest.json').read_text())
    (build / 'manifest.json').write_text(json.dumps(dict(manifest, complete=True)))
    with pytest.raises(ValueError, match='40 scenes'):
        brainscore.load_benchmark('Sun2025-motion-vector-error', root=build)
