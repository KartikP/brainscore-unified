"""NodeKit benchmarks and their data reader, on small synthetic stand-ins (no clone, no nkl)."""
import json

import pytest

import brainscore.harnesses.nodekit as nodekit
from brainscore.benchmarks.nodekit import data

pytestmark = pytest.mark.unit


def gabor_record(phase, size, duration, direction, response):
    return {'drifting_gabor_trial': {'phase': phase, 'size_degrees': size, 'contrast': 0.95,
                                     'duration_milliseconds': duration, 'direction': direction},
            'response': response, 'correct': response == direction, 'response_milliseconds': 500.0}


@pytest.fixture
def clone(tmp_path, monkeypatch):
    """A stand-in clone with two completed Gabor runs and one that did not complete."""
    folder = tmp_path / 'prolific' / 'gabor' / 'submissions'
    folder.mkdir(parents=True)
    # Each participant: right at 5 degrees 220 ms, wrong at 8 degrees 45 ms (a profile with a correlation).
    runs = {name: [gabor_record('practice', 5, 667, 'left', 'left'), gabor_record('test', 8, 45, 'right', 'left'),
                   gabor_record('test', 5, 220, 'right', 'right')] for name in ('b.json', 'a.json')}
    for name, result in runs.items():
        (folder / name).write_text(json.dumps([[0, 0, 'PageOpened', {}], [1, 5000, 'ProgramCompleted', result]]))
    (folder / 'c.json').write_text(json.dumps([[0, 0, 'PageOpened', {}], [1, 5, 'ProgramStopped', {}]]))
    monkeypatch.setattr(nodekit, 'source', lambda: tmp_path)
    return tmp_path


def test_the_reader_names_participants_by_file_order_and_counts_runs_that_did_not_complete(clone):
    assembly = data.trials('drifting-gabor')
    table = data.table(assembly)
    assert list(table.participant.unique()) == ['p1', 'p2']           # a.json, then b.json
    assert assembly.attrs['not_completed'] == 1 and assembly.attrs['longest_run_ms'] == 5.0
    assert table[table.participant == 'p1'].correct.tolist() == [True, False, True]
    assert {'phase', 'size_degrees', 'duration_milliseconds', 'direction', 'response'} <= set(table.columns)
    assert not any('.json' in str(value) for value in table.values.ravel())   # file names stay in the clone


def test_rows_puts_a_nested_record_beside_its_trial_and_keeps_lists_whole():
    table = data.rows([{'face_arrangement_trial': {'phase': 'test', 'face_ids': ['1', '2']},
                        'distances': [[0, 1], [1, 0]]}])
    assert table.loc[0, 'face_ids'] == ['1', '2'] and table.loc[0, 'distances'] == [[0, 1], [1, 0]]
    assert table.loc[0, 'trial'] == 0


# --- the benchmarks ------------------------------------------------------------------------------------------
import numpy as np
import pandas as pd

from brainscore.benchmarks.nodekit.benchmark import NodeKitBenchmark, faces, faces_ceiling, gabor, triplets
from brainscore_core.metrics import Score


def _table(records, participant):
    table = data.rows(records)
    table.insert(0, 'participant', participant)
    return table


def _gabor_people():
    """Two participants: right on every 5-degree trial, wrong on every 8-degree trial; practice all wrong."""
    records = [gabor_record('practice', 5, 667, 'left', 'right')] + [
        gabor_record('test', size, duration, 'left', 'left' if size == 5 else 'right')
        for size in (5, 8) for duration in (45, 220)]
    return pd.concat([_table(records, 'p1'), _table(records, 'p2')], ignore_index=True)


def test_gabor_correlates_the_condition_profile_with_the_participants():
    people = _gabor_people()           # right at 5 degrees, wrong at 8
    same = data.rows([gabor_record('test', size, duration, 'left', 'left' if size == 5 else 'right')
                      for size in (5, 8) for duration in (45, 220)])
    measured = gabor(same, people)
    assert measured['raw'] == pytest.approx(1.0) and measured['chance'] == 0.0
    assert measured['proportion_correct'] == 0.5 and measured['proportion_correct_participants'] == 0.5
    assert {row['participants'] for row in measured['by_condition']} == {0.0, 1.0}
    # Better than people everywhere is no profile: stated, not a number.
    better = gabor(data.rows([gabor_record('test', size, duration, 'left', 'left')
                              for size in (5, 8) for duration in (45, 220)]), people)
    assert np.isnan(better['raw']) and 'same in every condition' in better['score_note']


def test_repeated_runs_are_scored_together(clone):
    benchmark = NodeKitBenchmark('NodeKit.DriftingGabor', 'drifting-gabor')
    run = [gabor_record('test', 8, 45, 'right', 'left'), gabor_record('test', 5, 220, 'right', 'right')]
    score = benchmark.score_result([run, run, None])
    assert float(score) == 1.0 and score.attrs['runs'] == 3
    assert score.attrs['n_valid'] == 4 and score.attrs['n_missing'] == 2


def test_a_practice_trial_never_reaches_a_score():
    people = _gabor_people()
    tests = [gabor_record('test', size, duration, 'left', 'left' if duration == 220 else 'right')
             for size in (5, 8) for duration in (45, 220)]
    right = gabor(data.rows([gabor_record('practice', 5, 667, 'left', 'left')] + tests), people)
    wrong = gabor(data.rows([gabor_record('practice', 5, 667, 'left', 'right')] * 5 + tests), people)
    assert right == wrong
    # The participants' practice trials are wrong and their 5-degree tests right: a practice trial in the
    # reference would pull the 5-degree proportion below 1.
    assert max(row['participants'] for row in right['by_condition']) == 1.0


def _triplet(response, phase='test', reference='ref/1.png'):
    return {'triplet_similarity_trial': {'stimulus_id': reference, 'left_id': 'a', 'right_id': 'b', 'phase': phase,
                                         'block': 1, 'prompt': '', 'preferred': 'left', 'votes': 6},
            'response': response, 'response_milliseconds': 1.0}


def test_triplet_agreement_is_the_share_of_participants_answers_that_agree():
    people = pd.concat([_table([_triplet('left')], 'p1'), _table([_triplet('left')], 'p2'),
                        _table([_triplet('right')], 'p3'), _table([_triplet('right', 'practice')] * 9, 'p4')],
                       ignore_index=True)
    assert triplets(data.rows([_triplet('left')]), people)['raw'] == pytest.approx(2 / 3)
    assert triplets(data.rows([_triplet('right'), _triplet('left', 'practice')]), people)['raw'] == pytest.approx(1 / 3)


def _faces(distance, participant):
    """One trial of three faces; the pair (1, 2) at ``distance``, the others at 0.5 and 0.9."""
    record = {'face_arrangement_trial': {'phase': 'test', 'face_ids': ['1', '2', '3']},
              'distances': [[0, distance, 0.5], [distance, 0, 0.9], [0.5, 0.9, 0]], 'response_milliseconds': 1.0}
    return _table([record], participant)


def test_face_arrangement_correlates_pair_distances_with_the_participants():
    people = pd.concat([_faces(0.1, 'p1'), _faces(0.2, 'p2'), _faces(0.3, 'p3')], ignore_index=True)
    own = _faces(0.1, 'subject')
    measured = faces(own, people)
    assert measured['raw'] == pytest.approx(1.0)
    assert -1 <= measured['floor'] <= 1 and len(measured['floor_interval']) == 2
    assert faces_ceiling(people) == pytest.approx(1.0)


@pytest.mark.parametrize('identifier, task, form', [('NodeKit.DriftingGabor', 'drifting-gabor', 'motor:key'),
                                                    ('NodeKit.TripletSimilarity', 'triplet-similarity', 'motor:pointer'),
                                                    ('NodeKit.FaceArrangement', 'face-arrangement', 'motor:pointer')])
def test_a_subject_without_the_screen_or_the_act_form_is_na_before_any_compute(monkeypatch, identifier, task, form):
    import brainscore
    from brainscore import BrainScoreModel
    from brainscore_core.compatibility import CompatibilityError, check_compatibility
    from tests.benchmark_declarations import Declared
    benchmark = NodeKitBenchmark(identifier, task)
    assert benchmark.requested_output_channels == {form}
    with pytest.raises(CompatibilityError, match='motor'):
        check_compatibility(Declared({'vision:image', 'instruction'}, {'behavior:choice'}), benchmark)
    with pytest.raises(CompatibilityError):
        check_compatibility(Declared({'instruction'}, {form}), benchmark)
    monkeypatch.setattr(nodekit, 'source', lambda: pytest.fail('N/A must clone and run nothing'))
    called = []
    bare = BrainScoreModel('bare-motor', policy=lambda row: called.append(row) or {'motor': 0},
                           policy_in_channels=('vision:image', 'instruction'))
    for subject in (brainscore.load_model('chance-baseline'), bare):
        assert np.isnan(float(brainscore.score(subject, benchmark)))
    assert called == []


def test_a_run_that_does_not_complete_is_invalid_with_every_test_trial_missing(clone):
    benchmark = NodeKitBenchmark('NodeKit.DriftingGabor', 'drifting-gabor', trials='all')
    score = benchmark.score_result(None)
    assert score.attrs['n_missing'] == 2 and score.attrs['result_validity'] == 'invalid'
    assert benchmark.max_ms == 10.0                      # the whole task: twice the participants' longest run


def test_the_subject_response_times_are_recorded_not_scored(clone):
    benchmark = NodeKitBenchmark('NodeKit.DriftingGabor', 'drifting-gabor')
    fast = [gabor_record('test', 8, 45, 'right', 'left'), gabor_record('test', 5, 220, 'right', 'right')]
    slow = [fast[0], dict(fast[1], response_milliseconds=9000.0)]
    assert float(benchmark.score_result(fast)) == float(benchmark.score_result(slow)) == 1.0
    assert benchmark.score_result(slow).attrs['subject_response_ms'] == [500.0, 9000.0]


def test_the_three_benchmarks_are_registered():
    from brainscore import benchmark_registry
    assert {'NodeKit.DriftingGabor', 'NodeKit.TripletSimilarity', 'NodeKit.FaceArrangement'} <= set(benchmark_registry)


def test_the_pacing_is_realtime_by_default_stamped_and_checked(clone):
    benchmark = NodeKitBenchmark('NodeKit.DriftingGabor', 'drifting-gabor')
    assert benchmark.pacing == 'realtime' and benchmark.score_result(None).attrs['pacing'] == 'realtime'
    assert NodeKitBenchmark('NodeKit.DriftingGabor', 'drifting-gabor', pacing='dense').pacing == 'dense'
    with pytest.raises(ValueError, match='pacing'):
        NodeKitBenchmark('NodeKit.DriftingGabor', 'drifting-gabor', pacing='fast')


def test_the_gabor_briefing_names_the_key_that_turns_a_page():
    from brainscore.benchmarks.nodekit.benchmark import INSTRUCTIONS
    assert 'Enter' in INSTRUCTIONS['drifting-gabor']


def test_participants_are_numbered_among_the_runs_that_completed(tmp_path, monkeypatch):
    folder = tmp_path / 'prolific' / 'gabor' / 'submissions'
    folder.mkdir(parents=True)
    (folder / 'a.json').write_text(json.dumps([[0, 0, 'PageOpened', {}], [1, 5, 'ProgramStopped', {}]]))
    (folder / 'b.json').write_text(json.dumps([[0, 0, 'PageOpened', {}], [1, 5000, 'ProgramCompleted',
                                                [gabor_record('test', 8, 45, 'right', 'right')]]]))
    monkeypatch.setattr(nodekit, 'source', lambda: tmp_path)
    assert list(data.table(data.trials('drifting-gabor')).participant.unique()) == ['p1']
    assert len(NodeKitBenchmark('NodeKit.DriftingGabor', 'drifting-gabor').participants) == 1


def test_face_arrangement_is_off_the_leaderboard():
    assert NodeKitBenchmark('NodeKit.FaceArrangement', 'face-arrangement').leaderboard_eligible is False
    assert NodeKitBenchmark('NodeKit.DriftingGabor', 'drifting-gabor').leaderboard_eligible is True


def test_the_score_states_why_the_run_stopped(clone, monkeypatch):
    from types import SimpleNamespace
    environment = SimpleNamespace(steps=4, invalid_steps=4, acts_per_step=100, t_ms=10.0, trace=None,
                                  stop_cause='invalid answers in a row')
    session = SimpleNamespace(source=SimpleNamespace(end_reason='truncated'))
    monkeypatch.setattr(nodekit, 'play', lambda *args, **kwargs: (None, environment, session))
    monkeypatch.setattr(nodekit, 'bundle', lambda task: task)
    score = NodeKitBenchmark('NodeKit.DriftingGabor', 'drifting-gabor', record='unused')(object())
    assert score.attrs['stop_cause'] == 'invalid answers in a row'


def test_a_pair_shown_twice_has_the_mean_distance():
    from brainscore.benchmarks.nodekit.benchmark import pair_distances
    twice = pd.concat([_faces(0.1, 's'), _faces(0.3, 's')], ignore_index=True)
    assert pair_distances(twice)[('1', '2')] == pytest.approx(0.2)


def test_a_constant_arrangement_is_an_invalid_result(monkeypatch):
    people = pd.concat([_faces(0.1, 'p1'), _faces(0.2, 'p2'), _faces(0.3, 'p3')], ignore_index=True)
    benchmark = NodeKitBenchmark('NodeKit.FaceArrangement', 'face-arrangement')
    monkeypatch.setattr(NodeKitBenchmark, 'participants', people)
    monkeypatch.setattr(NodeKitBenchmark, 'ceiling', Score(1.0))
    flat = [{'face_arrangement_trial': {'phase': 'test', 'face_ids': ['1', '2', '3']},
             'distances': [[0, 0.5, 0.5], [0.5, 0, 0.5], [0.5, 0.5, 0]]}]
    score = benchmark.score_result(flat, steps=10, invalid_steps=2)
    assert score.attrs['result_validity'] == 'invalid' and score.attrs['invalid_step_share'] == 0.2


def test_the_dense_step_is_a_setting_of_dense_pacing_only(clone):
    benchmark = NodeKitBenchmark('NodeKit.DriftingGabor', 'drifting-gabor', pacing='dense', dense_step_ms=250)
    assert benchmark.score_result(None).attrs['dense_step_ms'] == 250
    with pytest.raises(ValueError, match='dense_step_ms'):
        NodeKitBenchmark('NodeKit.DriftingGabor', 'drifting-gabor', dense_step_ms=250)


def test_the_faces_score_states_how_far_each_arrangement_spreads():
    from brainscore.benchmarks.nodekit.benchmark import spread
    record = {'face_arrangement_trial': {'phase': 'test', 'face_ids': ['1', '2']}, 'arena_radius': 100.0,
              'positions': [{'x': 0, 'y': 0}, {'x': 150, 'y': 0}], 'distances': [[0, 1], [1, 0]]}
    pile = dict(record, positions=[{'x': 0, 'y': 0}, {'x': 5, 'y': 0}])
    assert spread(data.rows([record])) == [1.5] and spread(data.rows([pile])) == [0.05]


def test_the_defaults_are_the_nodekit_author_s_settings_and_the_score_states_them(clone):
    attrs = NodeKitBenchmark('NodeKit.DriftingGabor', 'drifting-gabor').score_result(None).attrs
    assert 'NodeKit author' in attrs['settings'] and attrs['view'] == 'agent' and attrs['picture'] == (640, 480)
    assert attrs['trials'] == 'slice' and attrs['acts_per_step_limit'] == 100 and attrs['motor']['top_speed'] == 5.0
    changed = NodeKitBenchmark('NodeKit.DriftingGabor', 'drifting-gabor', pacing='frozen').score_result(None).attrs
    assert 'settings' not in changed and changed['pacing'] == 'frozen'
    assert NodeKitBenchmark('NodeKit.DriftingGabor', 'drifting-gabor', view='participant').acts_per_step is None


def test_the_slice_compares_the_participants_first_test_trials_and_trials_all_the_whole_task(monkeypatch):
    from brainscore.benchmarks.nodekit import benchmark as module
    monkeypatch.setitem(module.AUTHOR_TASKS, 'drifting-gabor', {'shown': 1, 'max_steps': 160, 'max_ms': 900_000})
    people = data.rows([gabor_record('practice', 5, 667, 'left', 'left'), gabor_record('test', 8, 45, 'right', 'left'),
                        gabor_record('test', 5, 220, 'right', 'right')]).assign(participant='p1')
    monkeypatch.setattr(NodeKitBenchmark, 'participants', people)
    sliced = NodeKitBenchmark('NodeKit.DriftingGabor', 'drifting-gabor')
    assert sliced._tests == 1 and list(module.test_rows(sliced.compared)['size_degrees']) == [8]
    assert len(sliced.compared) == 2                      # the practice trial stays (it never reaches a score)
    whole = NodeKitBenchmark('NodeKit.DriftingGabor', 'drifting-gabor', trials='all')
    assert whole._tests == 2 and whole.max_steps == 20000
    with pytest.raises(ValueError, match='trials'):
        NodeKitBenchmark('NodeKit.DriftingGabor', 'drifting-gabor', trials=5)


def _arranged(radius_share):
    """One trial of three faces whose largest distance is ``radius_share`` arena radii."""
    d = radius_share
    return {'face_arrangement_trial': {'phase': 'test', 'face_ids': ['1', '2', '3']}, 'arena_radius': 100.0,
            'positions': [{'x': 0.0, 'y': 0.0}, {'x': 100.0 * d, 'y': 0.0}, {'x': 50.0 * d, 'y': 20.0 * d}],
            'distances': [[0, 0.3, 0.5], [0.3, 0, 0.9], [0.5, 0.9, 0]]}


def test_a_pile_of_faces_is_an_invalid_answer(monkeypatch):
    people = pd.concat([_faces(0.1, 'p1'), _faces(0.2, 'p2'), _faces(0.3, 'p3')], ignore_index=True)
    monkeypatch.setattr(NodeKitBenchmark, 'participants', people)
    monkeypatch.setattr(NodeKitBenchmark, 'ceiling', Score(1.0))
    benchmark = NodeKitBenchmark('NodeKit.FaceArrangement', 'face-arrangement')
    pile = benchmark.score_result([_arranged(0.2)] + [_arranged(1.0)] * 2)
    assert pile.attrs['n_invalid'] == 1 and pile.attrs['result_validity'] == 'invalid'
    assert 'arena radius' in pile.attrs['unarranged_rule']
    assert benchmark.score_result([_arranged(1.0)] * 3).attrs['n_invalid'] == 0
