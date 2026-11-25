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
        (folder / name).write_text(json.dumps([[0, 0, 'PageOpened', {}], [1, 5, 'ProgramCompleted', result]]))
    (folder / 'c.json').write_text(json.dumps([[0, 0, 'PageOpened', {}], [1, 5, 'ProgramStopped', {}]]))
    monkeypatch.setattr(nodekit, 'source', lambda: tmp_path)
    return tmp_path


def test_the_reader_names_participants_by_file_order_and_counts_runs_that_did_not_complete(clone):
    assembly = data.trials('drifting-gabor')
    table = data.table(assembly)
    assert list(table.participant.unique()) == ['p1', 'p2']           # a.json, then b.json
    assert assembly.attrs['not_completed'] == 1
    assert table[table.participant == 'p1'].correct.tolist() == [True, True]
    assert {'phase', 'size_degrees', 'duration_milliseconds', 'direction', 'response'} <= set(table.columns)
    assert not any('.json' in str(value) for value in table.values.ravel())   # file names stay in the clone


def test_rows_puts_a_nested_record_beside_its_trial_and_keeps_lists_whole():
    table = data.rows([{'face_arrangement_trial': {'phase': 'test', 'face_ids': ['1', '2']},
                        'distances': [[0, 1], [1, 0]]}])
    assert table.loc[0, 'face_ids'] == ['1', '2'] and table.loc[0, 'distances'] == [[0, 1], [1, 0]]
    assert table.loc[0, 'trial'] == 0
