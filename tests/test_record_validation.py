"""Reject damaged or incompatible recordings before downstream analysis."""
import hashlib
import json

import numpy as np
import pytest

from brainscore.run_record import RunRecord, RunRecorder


@pytest.fixture
def recording(tmp_path):
    directory = tmp_path / 'record'
    with RunRecorder(directory, metadata={'model': 'fixture'}) as recorder:
        recorder.record('input', 'question')
        recorder.record('output', np.array([1., 2.]))
    return directory


def rewrite_events(directory, change):
    """Simulate an internally inconsistent export with a valid outer checksum."""
    path = directory / 'events.jsonl'
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    change(rows)
    path.write_text(''.join(json.dumps(row) + '\n' for row in rows))
    manifest_path = directory / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    manifest['events_sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(manifest))


@pytest.mark.parametrize('version', [None, 0, 2, '1'])
def test_unknown_record_schema_rejected(recording, version):
    path = recording / 'manifest.json'
    manifest = json.loads(path.read_text())
    manifest['schema_version'] = version
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='schema version'):
        RunRecord(recording)


@pytest.mark.parametrize('version', [None, 0, 2, '1'])
def test_unknown_experiment_schema_rejected(recording, version):
    (recording / 'experiment.json').write_text(json.dumps({'schema_version': version}))
    with pytest.raises(ValueError, match='experiment schema'):
        RunRecord(recording)


@pytest.mark.parametrize('damage,message', [
    (lambda rows: rows.pop(), 'count mismatch'),
    (lambda rows: rows.reverse(), 'order mismatch'),
    (lambda rows: rows[1].update(sequence=0), 'order mismatch'),
])
def test_consistent_checksum_does_not_hide_missing_or_reordered_events(recording, damage, message):
    rewrite_events(recording, damage)
    with pytest.raises(ValueError, match=message):
        list(RunRecord(recording).outputs())


@pytest.mark.parametrize('payload,message', [
    ({'$array': '../outside'}, 'array artifact identity'),
    ({'$array': 'z' * 64}, 'array artifact identity'),
    ({'$file': None}, 'file artifact identity'),
    ({'$file': '../outside'}, 'file artifact identity'),
    ({'$event': 'UnknownEvent', 'fields': {}}, 'Unknown event schema'),
    ({'$assembly': 'UnknownAssembly'}, 'Unsupported assembly'),
    ({'$float': '1.0'}, 'nonfinite float encoding'),
    ({'$unknown': 'value'}, 'Unknown payload encoding'),
])
def test_invalid_payloads_fail_before_metric_runs(recording, payload, message):
    rewrite_events(recording, lambda rows: rows[1].update(payload=payload))
    called = []
    with pytest.raises(ValueError, match=message):
        RunRecord(recording).evaluate(lambda output, target: called.append(output), None)
    assert called == []


def test_file_artifact_tampering_is_detected(tmp_path):
    source = tmp_path / 'observation.txt'
    source.write_text('original')
    directory = tmp_path / 'record'
    with RunRecorder(directory, metadata={}) as recorder:
        recorder.record('input', source)
    next((directory / 'files').iterdir()).write_text('modified')
    with pytest.raises(ValueError, match='File artifact checksum'):
        list(RunRecord(directory).events())


@pytest.mark.parametrize('kind', ['array', 'file'])
def test_reusing_corrupt_artifact_fails_recording(tmp_path, kind):
    source = tmp_path / 'observation.txt'
    source.write_text('original')
    payload = np.arange(3) if kind == 'array' else source
    directory = tmp_path / 'record'
    with pytest.raises(ValueError, match='[Cc]orrupt|checksum'):
        with RunRecorder(directory, metadata={}) as recorder:
            recorder.record('input', payload)
            next((directory / ('arrays' if kind == 'array' else 'files')).iterdir()).write_bytes(b'bad')
            recorder.record('input', payload)
    assert RunRecord(directory).manifest['status'] == 'failed'


def test_closed_recorder_cannot_append_and_close_is_idempotent(recording):
    directory = recording.parent / 'closed'
    recorder = RunRecorder(directory, metadata={})
    recorder.record('output', 1)
    recorder.close()
    before = {p.name: p.read_bytes() for p in directory.iterdir()}
    recorder.close(failed=True)
    with pytest.raises(RuntimeError, match='closed'):
        recorder.record('output', 2)
    assert before == {p.name: p.read_bytes() for p in directory.iterdir()}
    assert list(RunRecord(directory).outputs()) == [1]


def test_invalid_direction_does_not_append(tmp_path):
    with RunRecorder(tmp_path / 'record', metadata={}) as recorder:
        with pytest.raises(ValueError, match='direction'):
            recorder.record('prediction', 1)
        recorder.record('output', 2)
    assert list(RunRecord(recorder.directory).outputs()) == [2]


@pytest.mark.parametrize('index', [-1, 1])
def test_missing_output_cannot_be_scored(recording, index):
    calls = []
    with pytest.raises(IndexError):
        RunRecord(recording).evaluate(lambda actual, target: calls.append(actual), None, output_index=index)
    assert calls == []


def test_nonfinite_and_numpy_scalars_roundtrip(tmp_path):
    values = [np.int64(3), float('nan'), float('inf'), -float('inf')]
    with RunRecorder(tmp_path / 'record', metadata={}) as recorder:
        recorder.record('output', values)
    actual = list(RunRecord(recorder.directory).outputs())[0]
    assert actual[0] == 3
    assert np.isnan(actual[1]) and actual[2] == np.inf and actual[3] == -np.inf


@pytest.mark.parametrize('payload', ['not an envelope', {'value': 1}])
def test_invalid_experiment_envelope_rejected(tmp_path, payload):
    root = tmp_path / 'experiment'
    with RunRecorder(root / 'inputs_outputs', metadata={}) as recorder:
        recorder.record('output', payload)
    (root / 'experiment.json').write_text(json.dumps({'schema_version': 1, 'status': 'complete'}))
    with pytest.raises(ValueError, match='event envelope'):
        list(RunRecord(root).events())
