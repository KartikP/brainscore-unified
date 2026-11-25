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
    runs = {'b.json': [gabor_record('practice', 5, 667, 'left', 'left'), gabor_record('test', 8, 45, 'right', 'left')],
            'a.json': [gabor_record('practice', 5, 667, 'left', 'left'), gabor_record('test', 8, 45, 'right', 'right')]}
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
    assert table[table.participant == 'p1'].correct.tolist() == [True, True]
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


def test_gabor_compares_proportion_correct_by_condition():
    people = _gabor_people()
    own = data.rows([gabor_record('test', size, duration, 'left', 'left')
                     for size in (5, 8) for duration in (45, 220)])
    measured = gabor(own, people)
    assert measured['raw'] == 1.0 and measured['chance'] == 0.5
    assert {row['participants'] for row in measured['by_condition']} == {0.0, 1.0}


def test_a_practice_trial_never_reaches_a_score():
    people = _gabor_people()
    tests = [gabor_record('test', size, duration, 'left', 'left') for size in (5, 8) for duration in (45, 220)]
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
    benchmark = NodeKitBenchmark('NodeKit.DriftingGabor', 'drifting-gabor')
    score = benchmark.score_result(None)
    assert score.attrs['n_missing'] == 1 and score.attrs['result_validity'] == 'invalid'
    assert benchmark.max_ms == 10.0                      # twice the participants' longest run


def test_the_subject_response_times_are_recorded_not_scored(clone):
    benchmark = NodeKitBenchmark('NodeKit.DriftingGabor', 'drifting-gabor')
    fast = [gabor_record('test', 8, 45, 'right', 'right')]
    slow = [dict(fast[0], response_milliseconds=9000.0)]
    assert float(benchmark.score_result(fast)) == float(benchmark.score_result(slow))
    assert benchmark.score_result(slow).attrs['subject_response_ms'] == [9000.0]


def test_the_three_benchmarks_are_registered():
    from brainscore import benchmark_registry
    assert {'NodeKit.DriftingGabor', 'NodeKit.TripletSimilarity', 'NodeKit.FaceArrangement'} <= set(benchmark_registry)


def test_the_pacing_is_frozen_by_default_stamped_and_checked(clone):
    benchmark = NodeKitBenchmark('NodeKit.DriftingGabor', 'drifting-gabor')
    assert benchmark.pacing == 'frozen' and benchmark.score_result(None).attrs['pacing'] == 'frozen'
    assert NodeKitBenchmark('NodeKit.DriftingGabor', 'drifting-gabor', pacing='dense').pacing == 'dense'
    with pytest.raises(ValueError, match='pacing'):
        NodeKitBenchmark('NodeKit.DriftingGabor', 'drifting-gabor', pacing='fast')


def test_the_gabor_briefing_names_the_key_that_turns_a_page():
    from brainscore.benchmarks.nodekit.benchmark import INSTRUCTIONS
    assert 'Enter' in INSTRUCTIONS['drifting-gabor']
