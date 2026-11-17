"""Sun2025 motion benchmark through participate, on a made-up prepared build."""
import hashlib
import json
import math

import numpy as np
import pytest

import brainscore
from brainscore_core.contract import Subject
from brainscore_core.streaming import StreamEvent
from brainscore.benchmarks.sun2025.metric import parse_report, unflip_vector
from brainscore.data import local

pytestmark = pytest.mark.unit

HUMAN = {1: (2.0, -1.0), 2: (0.0, 3.0)}   # made-up (x, y) per modulation, equal for all participants
NO_MOTION = '{"direction_deg": 0, "speed_px_per_frame": 0}'


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def build(tmp_path):
    """A made-up build in the prepared form: scene 1, modulations 1 and 2, four flips each."""
    root = tmp_path / 'sun2025'
    (root / 'clips').mkdir(parents=True)
    stimuli, humans = [], []
    for modulation, (x, y) in HUMAN.items():
        base_id = f'sun2025-s01-m{modulation}-v1'
        for flip in (1, 2, 3, 4):
            name = f'clips/{base_id}-f{flip}.avi'
            (root / name).write_bytes(f'made-up clip {base_id} {flip}'.encode())
            stimuli.append(dict(stimulus_id=f'{base_id}-f{flip}', base_id=base_id, scene=1,
                                modulation=modulation, flip=flip, video_path=name))
        humans += [dict(base_id=base_id, scene=1, modulation=modulation, participant=p, x=x, y=y)
                   for p in (1, 2, 3)]
    (root / 'stimuli.json').write_text(json.dumps(stimuli))
    (root / 'human-responses.json').write_text(json.dumps(humans))
    files = {path.relative_to(root).as_posix(): _sha(path) for path in root.rglob('*') if path.is_file()}
    (root / 'manifest.json').write_text(json.dumps(dict(
        dataset='Sun2025-second-order-motion', schema_version=1, scenes=[1], complete=False,
        files=files, trials=len(stimuli), speed_units='displayed pixels per frame',
        coordinates='x right, y down')))
    return root


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


def test_missing_data_names_the_prepare_command(tmp_path, monkeypatch):
    monkeypatch.setenv('BRAINSCORE_SUN2025_ROOT', str(tmp_path / 'absent'))
    with pytest.raises(local.LocalDataMissing, match='python -m brainscore.data prepare Sun2025'):
        brainscore.load_stimulus_set('Sun2025-second-order-motion')


def test_a_changed_file_is_refused(build):
    (build / 'clips' / 'sun2025-s01-m1-v1-f1.avi').write_bytes(b'changed')
    with pytest.raises(ValueError, match='Checksum'):
        brainscore.load_stimulus_set('Sun2025-second-order-motion', root=build)


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
    """An in-memory design: group g has scene 1, modulation g + 1 and four flips; human is groups x 3 x 2."""
    from brainscore_core.supported_data_standards.brainio.assemblies import BehavioralAssembly
    from brainscore_core.supported_data_standards.brainio.stimuli import StimulusSet
    from brainscore.benchmarks.sun2025.benchmark import Sun2025MotionBenchmark
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
    return Sun2025MotionBenchmark(stimuli, reference)


def test_the_score_equals_the_old_implementation_to_the_last_digit():
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
    assert float(score) == 4.255378301966291
