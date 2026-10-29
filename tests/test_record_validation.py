"""Reject damaged or incompatible recordings before downstream analysis."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from brainscore.run_record import RecordingBudgetExceeded, RunRecord, RunRecorder


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


def long_recording(directory, count, width=0):
    with RunRecorder(directory, metadata={}) as recorder:
        for index in range(count):
            recorder.record('output', [index, 'x' * width])
    return directory


def damage_last_byte(directory):
    path = directory / 'events.jsonl'
    blob = bytearray(path.read_bytes())
    blob[-3] = ord('9') if blob[-3] != ord('9') else ord('8')
    path.write_bytes(bytes(blob))


def truncate_last_json(directory):
    """Make the last line invalid JSON while keeping checksum and count consistent."""
    path = directory / 'events.jsonl'
    lines = path.read_bytes().splitlines(keepends=True)
    lines[-1] = b'{"sequence": 49, \n'
    path.write_bytes(b''.join(lines))
    manifest_path = directory / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    manifest['events_sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(manifest))


@pytest.mark.parametrize('damage,message', [
    (damage_last_byte, 'checksum mismatch'),
    (lambda d: rewrite_events(d, lambda rows: rows.pop()), 'count mismatch'),
    (lambda d: rewrite_events(d, lambda rows: rows.__setitem__(slice(-2, None), rows[:-3:-1])),
     'order mismatch'),
    (truncate_last_json, 'Expecting'),
])
def test_damage_late_in_log_fails_before_first_event(tmp_path, damage, message):
    directory = long_recording(tmp_path / 'record', 50)
    damage(directory)
    with pytest.raises(ValueError, match=message):
        next(RunRecord(directory).events())


def test_closed_record_hash_and_values_are_exact(tmp_path):
    directory = long_recording(tmp_path / 'record', 5, width=3)
    manifest = RunRecord(directory).manifest
    assert manifest['events_sha256'] == hashlib.sha256((directory / 'events.jsonl').read_bytes()).hexdigest()
    assert manifest['event_count'] == 5
    assert list(RunRecord(directory).outputs()) == [[index, 'xxx'] for index in range(5)]


def test_empty_log_reads_as_no_events(tmp_path):
    with RunRecorder(tmp_path / 'record', metadata={}):
        pass
    record = RunRecord(tmp_path / 'record')
    assert record.manifest['events_sha256'] == hashlib.sha256(b'').hexdigest()
    assert record.manifest['event_count'] == 0
    assert list(record.events()) == []


def test_open_record_reads_validated_prefix_and_stays_unscorable(tmp_path):
    recorder = RunRecorder(tmp_path / 'record', metadata={})
    recorder.record('output', 1)
    recorder.record('output', 2)
    record = RunRecord(recorder.directory)
    events = record.outputs()
    assert next(events) == 1
    recorder.record('output', 3)  # a live recorder may append; the reader keeps its prefix
    assert list(events) == [2]
    with pytest.raises(ValueError, match='completed run'):
        record.evaluate(lambda actual, target: actual, None)


def test_closed_log_changed_during_iteration_is_rejected(tmp_path):
    directory = long_recording(tmp_path / 'record', 3)
    events = RunRecord(directory).outputs()
    assert next(events) == [0, '']
    with (directory / 'events.jsonl').open('a') as handle:
        handle.write('{}\n')
    with pytest.raises(ValueError, match='changed while reading'):
        next(events)


def test_restored_mtime_edit_is_rejected_before_affected_output(tmp_path):
    """Restoring mtime with utime still changes ctime, which the reader checks before each yield."""
    import os
    directory = long_recording(tmp_path / 'record', 3)
    path = directory / 'events.jsonl'
    events = RunRecord(directory).outputs()
    assert next(events) == [0, '']
    status = path.stat()
    with path.open('r+b') as handle:
        handle.seek(path.read_bytes().index(b'[1, '))
        handle.write(b'[7, ')
    os.utime(path, ns=(status.st_atime_ns, status.st_mtime_ns))
    # Allow coarse filesystem clocks to advance without skipping a required check.
    import time
    deadline = time.monotonic() + 2
    while path.stat().st_ctime_ns == status.st_ctime_ns and time.monotonic() < deadline:
        time.sleep(0.02)
        os.utime(path, ns=(status.st_atime_ns, status.st_mtime_ns))
    assert path.stat().st_ctime_ns != status.st_ctime_ns
    with pytest.raises(ValueError, match='changed while reading'):
        next(events)


@pytest.mark.parametrize('call,message', [
    (1, 'checksum mismatch'),  # before validation: the change is validated and rejected
    (2, 'changed while reading'),  # between validation and iteration
    (3, 'changed while reading'),  # after decoding, immediately before the first yield
])
def test_ordinary_write_at_each_stamp_is_rejected(tmp_path, monkeypatch, call, message):
    from brainscore import run_record
    directory = long_recording(tmp_path / 'record', 1)
    path = directory / 'events.jsonl'
    stamp, calls = run_record._stamp, []

    def write_at_call(handle):
        calls.append(None)
        if len(calls) == call:
            path.write_bytes(path.read_bytes().replace(b'[0, ', b'[9, '))
        return stamp(handle)
    monkeypatch.setattr(run_record, '_stamp', write_at_call)
    yielded = []
    with pytest.raises(ValueError, match=message):
        for output in RunRecord(directory).outputs():
            yielded.append(output)
    assert yielded == []


def test_replaced_closed_log_fails_closed(tmp_path):
    """Rename-over drops a link from the open inode, changing its ctime."""
    import os
    directory = long_recording(tmp_path / 'record', 3)
    events = RunRecord(directory).outputs()
    assert next(events) == [0, '']
    replacement = directory / 'replacement'
    replacement.write_text('')
    os.replace(replacement, directory / 'events.jsonl')
    with pytest.raises(ValueError, match='changed while reading'):
        next(events)


def test_finalize_and_read_memory_do_not_scale_with_log_length(tmp_path):
    import tracemalloc
    peaks = {}
    for count in (200, 20000):
        directory = tmp_path / str(count)
        recorder = RunRecorder(directory, metadata={})
        for index in range(count):
            recorder.record('output', [index, 'x' * 500])
        tracemalloc.start()
        recorder.close()
        close_peak = tracemalloc.get_traced_memory()[1]
        tracemalloc.reset_peak()
        assert sum(1 for _ in RunRecord(directory).events()) == count
        peaks[count] = (close_peak, tracemalloc.get_traced_memory()[1])
        tracemalloc.stop()
    size = (tmp_path / '20000' / 'events.jsonl').stat().st_size
    assert size > 10_000_000
    for small, large in zip(peaks[200], peaks[20000]):
        assert large < 2 * small + 100_000, peaks
        assert large < size / 20, peaks
    print(f'log bytes={size} peaks(close, read)={peaks}')


def recorded_size(directory):
    """Bytes the budget covers: the event log plus stored array and file assets."""
    assets = [p for name in ('arrays', 'files') for p in (directory / name).glob('*')]
    return (directory / 'events.jsonl').stat().st_size + sum(p.stat().st_size for p in assets)


def budget_record(directory, max_bytes, events, metadata=None):
    recorder = RunRecorder(directory, metadata=metadata or {'model': 'fixture'},
                           max_bytes=max_bytes)
    with recorder:
        for event in events:
            recorder.record('output', event)
    return json.loads((directory / 'manifest.json').read_text())


@pytest.mark.parametrize('value', [0, -1, True, 1.5, '10'])
def test_invalid_budget_fails_before_output(tmp_path, value):
    from brainscore.experiments import RecordInputsOutputs
    with pytest.raises(ValueError, match='max_bytes'):
        RunRecorder(tmp_path / 'record', metadata={}, max_bytes=value)
    with pytest.raises(ValueError, match='max_bytes'):
        RecordInputsOutputs(max_bytes=value)
    assert not (tmp_path / 'record').exists()


def test_budget_exact_boundary_counts_utf8_log_and_assets(tmp_path):
    asset = tmp_path / 'stimulus.txt'
    asset.write_bytes(b'file asset')
    events = ['café ☃', {'nested': [np.arange(3.), asset]}]
    reference = budget_record(tmp_path / 'reference', 10 ** 9, events)
    needed = recorded_size(tmp_path / 'reference')
    assert reference['budget']['reserved_bytes'] == needed
    exact = budget_record(tmp_path / 'exact', needed, events)
    assert exact['status'] == 'complete' and exact['budget'] == {
        'max_bytes': needed, 'reserved_bytes': needed, 'exceeded': None}
    assert exact['events_sha256'] == reference['events_sha256']
    with pytest.raises(RecordingBudgetExceeded, match='exceed max_bytes'):
        budget_record(tmp_path / 'short', needed - 1, events)
    short = json.loads((tmp_path / 'short' / 'manifest.json').read_text())
    assert short['status'] == 'failed' and short['event_count'] == 1
    assert recorded_size(tmp_path / 'short') == short['budget']['reserved_bytes'] <= needed - 1


def test_default_and_unlimited_records_are_unchanged(tmp_path):
    events = ['text', np.arange(4), (1.5, float('nan'))]
    default = budget_record(tmp_path / 'default', None, events)
    capped = budget_record(tmp_path / 'capped', 10 ** 9, events)
    assert 'budget' not in default and default['status'] == 'complete'
    assert default['events_sha256'] == capped['events_sha256']
    assert (tmp_path / 'default' / 'events.jsonl').read_bytes() == (
        tmp_path / 'capped' / 'events.jsonl').read_bytes()
    restored = list(RunRecord(tmp_path / 'default').outputs())
    np.testing.assert_equal(restored[1], np.arange(4))


def test_duplicate_assets_count_once(tmp_path):
    array = np.arange(100.)
    manifest = budget_record(tmp_path / 'record', 10 ** 6, [array, {'again': array}],
                             metadata={'reference': array})
    assert len(list((tmp_path / 'record' / 'arrays').glob('*'))) == 1
    assert manifest['budget']['reserved_bytes'] == recorded_size(tmp_path / 'record')


def test_metadata_asset_over_budget_leaves_no_record(tmp_path):
    with pytest.raises(RecordingBudgetExceeded):
        RunRecorder(tmp_path / 'record', metadata={'weights': np.zeros(100)}, max_bytes=64)
    assert not (tmp_path / 'record' / 'manifest.json').exists()
    assert not list((tmp_path / 'record').rglob('*.npy'))
    with pytest.raises(FileNotFoundError):
        RunRecord(tmp_path / 'record')


def test_oversized_asset_is_not_written(tmp_path):
    with pytest.raises(RecordingBudgetExceeded):
        budget_record(tmp_path / 'record', 1000, ['small', np.zeros(1000)])
    manifest = json.loads((tmp_path / 'record' / 'manifest.json').read_text())
    assert manifest['status'] == 'failed' and manifest['event_count'] == 1
    assert not list((tmp_path / 'record' / 'arrays').glob('*'))
    assert [row['payload'] for row in RunRecord(tmp_path / 'record').events()] == ['small']


def test_partial_event_assets_stay_counted_and_event_is_absent(tmp_path):
    first, second = np.zeros(10), np.ones(1000)
    with pytest.raises(RecordingBudgetExceeded):
        budget_record(tmp_path / 'record', 1000, [[first, second]])
    manifest = json.loads((tmp_path / 'record' / 'manifest.json').read_text())
    assert manifest['status'] == 'failed' and manifest['event_count'] == 0
    assert len(list((tmp_path / 'record' / 'arrays').glob('*'))) == 1
    assert manifest['budget']['reserved_bytes'] == recorded_size(tmp_path / 'record') <= 1000
    with pytest.raises(ValueError, match='completed run'):
        RunRecord(tmp_path / 'record').evaluate(lambda o, t: 1, None)


def test_caught_exhaustion_stops_recording_and_fails_on_normal_close(tmp_path):
    recorder = RunRecorder(tmp_path / 'record', metadata={}, max_bytes=200)
    recorder.record('input', 'fits')
    with pytest.raises(RecordingBudgetExceeded):
        recorder.record('output', 'x' * 500)
    with pytest.raises(RecordingBudgetExceeded, match='Recording stopped'):
        recorder.record('output', 'y')  # would fit, but a gap must not follow
    recorder.close()
    manifest = json.loads((tmp_path / 'record' / 'manifest.json').read_text())
    assert manifest['status'] == 'failed' and manifest['event_count'] == 1
    assert 'max_bytes=200' in manifest['budget']['exceeded']


def test_corrupt_existing_asset_still_fails_under_budget(tmp_path):
    array = np.arange(5)
    recorder = RunRecorder(tmp_path / 'record', metadata={}, max_bytes=10 ** 6)
    recorder.record('output', array)
    (asset,) = (tmp_path / 'record' / 'arrays').glob('*')
    asset.write_bytes(b'damaged')
    with pytest.raises(ValueError, match='Corrupt array artifact'):
        recorder.record('output', array)
    with pytest.raises(RuntimeError, match='earlier failure: ValueError'):
        recorder.record('output', 'later')
    recorder.close()
    manifest = json.loads((tmp_path / 'record' / 'manifest.json').read_text())
    assert manifest['status'] == 'failed' and manifest['event_count'] == 1
    assert manifest['budget']['exceeded'] is None


class FailingWrites:
    """Patch Path.open so the first matching write stores a prefix, then fails."""

    def __init__(self, monkeypatch, name, mode, *, partial):
        original = Path.open

        def opener(path, mode_="r", *args, **kwargs):
            handle = original(path, mode_, *args, **kwargs)
            if path.name.endswith(name) and mode_ == mode and not self.fired:
                self.fired = True
                if not partial:
                    handle.close()
                    raise OSError('injected open failure')
                write = handle.write

                def torn(data):
                    write(data[:len(data) // 2])
                    handle.flush()
                    raise OSError('injected write failure')
                handle.write = torn
            return handle
        self.fired = False
        monkeypatch.setattr(Path, 'open', opener)


@pytest.mark.parametrize('max_bytes', [None, 1000])
@pytest.mark.parametrize('partial', [False, True])
def test_caught_event_log_failure_fails_record_and_stops(tmp_path, monkeypatch, max_bytes,
                                                         partial):
    recorder = RunRecorder(tmp_path / 'record', metadata={}, max_bytes=max_bytes)
    FailingWrites(monkeypatch, 'events.jsonl', 'ab', partial=partial)
    with pytest.raises(OSError, match='injected'):
        recorder.record('output', 'lost')
    monkeypatch.undo()
    with pytest.raises(RuntimeError, match='earlier failure: OSError'):
        recorder.record('output', 'after')
    recorder.close()
    manifest = json.loads((tmp_path / 'record' / 'manifest.json').read_text())
    stored = (tmp_path / 'record' / 'events.jsonl').stat().st_size
    assert manifest['status'] == 'failed' and manifest['event_count'] == 0
    assert (stored > 0) == partial
    if max_bytes:
        assert stored <= manifest['budget']['reserved_bytes'] <= max_bytes
    with pytest.raises(ValueError, match='completed run'):
        RunRecord(tmp_path / 'record').evaluate(lambda o, t: 1, None)


def test_caught_partial_asset_write_fails_record(tmp_path, monkeypatch):
    recorder = RunRecorder(tmp_path / 'record', metadata={}, max_bytes=10 ** 6)
    FailingWrites(monkeypatch, '.npy', 'xb', partial=True)
    with pytest.raises(OSError, match='injected'):
        recorder.record('output', np.arange(100.))
    monkeypatch.undo()
    with pytest.raises(RuntimeError, match='earlier failure'):
        recorder.record('output', np.arange(100.))  # must not reuse the torn asset
    recorder.close()
    manifest = json.loads((tmp_path / 'record' / 'manifest.json').read_text())
    assert manifest['status'] == 'failed' and manifest['event_count'] == 0
    assert recorded_size(tmp_path / 'record') <= manifest['budget']['reserved_bytes']


def test_caught_codec_failure_fails_record(tmp_path):
    recorder = RunRecorder(tmp_path / 'record', metadata={})
    recorder.record('input', 'kept')
    with pytest.raises(TypeError, match='Unsupported record payload'):
        recorder.record('output', object())
    with pytest.raises(RuntimeError, match='earlier failure: TypeError'):
        recorder.record('output', 'after')
    recorder.close()
    manifest = json.loads((tmp_path / 'record' / 'manifest.json').read_text())
    assert manifest['status'] == 'failed' and manifest['event_count'] == 1


def test_metadata_codec_failure_leaves_no_manifest(tmp_path):
    with pytest.raises(TypeError):
        RunRecorder(tmp_path / 'record', metadata={'bad': object()})
    assert not (tmp_path / 'record' / 'manifest.json').exists()
