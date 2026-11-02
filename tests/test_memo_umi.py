"""Contract and recording fidelity tests; model inference stays in memo's worker."""

import json
import os
import subprocess
import sys
import textwrap

import numpy as np
import pytest

from brainscore.experiments import (
    Experiment, RecordInputsOutputs, SessionProtocol, Trial, compare_outputs,
    read_events, replay_sessions,
)
from brainscore_core.streaming import BufferedSession, StreamEvent

from brainscore.memo_umi import MemoConfig, MemoSubject, MemoWorker, MemoWorkerError, MemoWorkerTimeout
from brainscore.memo_umi.subject import CHOICES, SCENE, UTTERANCES


LITERAL = np.array([
    [0.5, 0.5, 0.0],
    [0.0, 0.0, 1.0],
    [1.0, 0.0, 0.0],
    [0.0, 0.5, 0.5],
])
RECURSIVE = np.array([
    [0.43, 0.57, 0.0],
    [0.0, 0.0, 1.0],
    [1.0, 0.0, 0.0],
    [0.0, 0.57, 0.43],
])


class FakeWorker:
    def __init__(self, python_executable, *, timeout_s):
        self.runtime = None
        self.closed = False

    def predict(self, *, depth, beta, utterances):
        self.runtime = {'python': '3.12.0', 'memo_version': 'fixture',
                        'jax_version': 'fixture', 'backend': 'cpu'}
        table = LITERAL if depth == 0 else RECURSIVE
        positions = {name: i for i, name in enumerate(UTTERANCES)}
        return table[[positions[name] for name in utterances]].tolist()

    def close(self):
        self.closed = True
        self.runtime = None


def inputs():
    return [StreamEvent('instruction', SCENE, 0.0)] + [
        StreamEvent('text', utterance, float(index + 1), {'stimulus_id': utterance})
        for index, utterance in enumerate(UTTERANCES)
    ]


def make_subject(monkeypatch, depth=1, beta=1.0):
    import brainscore.memo_umi.subject as subject_module
    monkeypatch.setattr(subject_module, 'MemoWorker', FakeWorker)
    return MemoSubject(config=MemoConfig(depth=depth, beta=beta),
                       python_executable='unused-python')


def test_memo_subject_runs_via_interact_and_preserves_paper_tables(monkeypatch):
    for depth, expected in ((0, LITERAL), (1, RECURSIVE)):
        subject = make_subject(monkeypatch, depth=depth)
        session = BufferedSession(inputs(), requested_output_channels=('behavior',))
        subject.interact(session)
        assembly = session.collect('behavior')
        assert assembly.dims == ('presentation', 'choice')
        assert list(assembly.choice.values) == list(CHOICES)
        assert list(assembly.stimulus_id.values) == list(UTTERANCES)
        np.testing.assert_array_equal(assembly.values, expected)
        assert session.emitted[0].meta['provenance']['reasoning_depth'] == depth
        subject.reset()


def test_subject_declares_only_the_channels_it_consumes_and_emits(monkeypatch):
    subject = make_subject(monkeypatch)
    assert subject.in_channels == {'instruction', 'text'}
    assert subject.required_channels == {'instruction', 'text'}
    assert subject.out_channels == {'behavior'}


@pytest.mark.parametrize('bad_inputs', [
    [StreamEvent('text', 'green', 0, {})],
    [StreamEvent('instruction', 'wrong task', 0, {}),
     StreamEvent('text', 'green', 1, {})],
    [StreamEvent('instruction', SCENE, 0, {}),
     StreamEvent('text', 'blue', 1, {})],
])
def test_malformed_inputs_fail_before_calling_worker(monkeypatch, bad_inputs):
    subject = make_subject(monkeypatch)
    session = BufferedSession(bad_inputs, requested_output_channels=('behavior',))
    with pytest.raises(ValueError):
        subject.interact(session)
    assert subject._worker.runtime is None


def test_unrequested_output_fails_before_memo_compute(monkeypatch):
    subject = make_subject(monkeypatch)
    session = BufferedSession(inputs(), requested_output_channels=('neural:IT',))
    with pytest.raises(ValueError, match='Unsupported requested outputs'):
        subject.interact(session)
    assert subject._worker.runtime is None


def test_configuration_is_explicit_and_depth_changes_predictions(monkeypatch):
    literal = make_subject(monkeypatch, depth=0).config
    recursive = make_subject(monkeypatch, depth=1).config
    assert literal != recursive
    assert MemoConfig(depth=1, beta=1.0) == recursive
    with pytest.raises(ValueError):
        MemoConfig(depth=-1)
    with pytest.raises(ValueError):
        MemoConfig(depth=True)
    subject = make_subject(monkeypatch, depth=0)
    with pytest.raises(AttributeError):
        subject.config = recursive


def test_repeated_runs_and_reset_do_not_change_a_subject_configuration(monkeypatch):
    subject = make_subject(monkeypatch, depth=1)
    for _ in range(2):
        session = BufferedSession(inputs(), requested_output_channels=('behavior',))
        subject.interact(session)
        np.testing.assert_array_equal(session.collect('behavior').values, RECURSIVE)
    subject.reset()
    assert subject._worker.runtime is None
    assert subject.config == MemoConfig(depth=1, beta=1.0)


def test_invalid_worker_output_closes_worker_without_emitting_partial_rows(monkeypatch):
    import brainscore.memo_umi.subject as subject_module

    class InvalidWorker(FakeWorker):
        def predict(self, *, depth, beta, utterances):
            self.runtime = {'python': '3.12.0', 'memo_version': 'fixture',
                            'jax_version': 'fixture', 'backend': 'cpu'}
            return [[1.0]] * len(utterances)

    monkeypatch.setattr(subject_module, 'MemoWorker', InvalidWorker)
    subject = MemoSubject(config=MemoConfig(depth=1), python_executable='unused')
    session = BufferedSession(inputs(), requested_output_channels=('behavior',))
    with pytest.raises(RuntimeError, match='probability shape'):
        subject.interact(session)
    assert session.emitted == []
    assert subject._worker.closed


def test_experiment_recording_and_input_replay_preserve_probabilities(monkeypatch, tmp_path):
    import brainscore.memo_umi.subject as subject_module
    monkeypatch.setattr(subject_module, 'MemoWorker', FakeWorker)
    output = tmp_path / 'original'

    def factory(_trial: Trial):
        return _Session(inputs())

    protocol = SessionProtocol(
        'memo-fixture', factory, trials=('scene-1',),
        input_channels=('instruction', 'text'), output_channels=('behavior',),
    )
    original = Experiment(
        subject=MemoSubject(config=MemoConfig(depth=1), python_executable='fake'),
        protocol=protocol, tools=(RecordInputsOutputs(),), output_dir=output,
        metadata={'configuration': {'depth': 1, 'beta': 1.0}},
    ).run()
    events = list(original.record.events())
    assert [event['kind'] for event in events].count('input') == 5
    outputs = [event['payload'] for event in events if event['kind'] == 'output']
    assert len(outputs) == 4
    assert outputs[0].meta['provenance']['beta'] == 1.0
    assert outputs[0].meta['label_set'] == CHOICES
    assert outputs[0].meta['stimulus_id'] == 'green'
    np.testing.assert_array_equal(original.value[0].values, RECURSIVE)
    np.testing.assert_array_equal(outputs[0].payload, RECURSIVE[0])
    assert outputs[0].meta['provenance']['reasoning_depth'] == 1
    assert outputs[0].meta['provenance']['memo_source_reviewed_revision'] == (
        'd8aac635ee53483e8e0311b85f6898ef37710a26')
    assert original.manifest['plan']['metadata']['configuration'] == {
        'depth': 1, 'beta': 1.0,
    }

    replay_protocol = replay_sessions(output)
    replay = Experiment(
        subject=MemoSubject(config=MemoConfig(depth=1), python_executable='fake'),
        protocol=replay_protocol, tools=(RecordInputsOutputs(),),
        output_dir=tmp_path / 'replay',
    ).run()
    assert compare_outputs(output, replay.directory)['equal']
    replay_manifest = replay.manifest['plan']['protocol']['config']['metadata']
    assert replay_manifest['mode'] == 'open_loop_saved_inputs'
    assert read_events(replay.directory)


class _Session:
    def __init__(self, events):
        self._session = BufferedSession(events, requested_output_channels=('behavior',))

    def __enter__(self):
        return self._session

    def __exit__(self, exc_type, exc, tb):
        return False


def fake_worker_script(tmp_path, body):
    path = tmp_path / 'fake_worker.py'
    path.write_text(textwrap.dedent(body))
    return path


def test_worker_batches_requests_and_closes_cleanly(tmp_path):
    script = fake_worker_script(tmp_path, '''
        import json, sys
        for line in sys.stdin:
            request = json.loads(line)
            if request['op'] == 'shutdown': break
            if request['op'] == 'hello':
                print(json.dumps({'ok': True, 'python_major_minor': [3, 12],
                    'memo_version': 'fixture', 'jax_version': 'fixture'}), flush=True)
            else:
                print(json.dumps({'ok': True, 'runtime': {'backend': 'cpu'},
                    'probabilities': [[0.5, 0.5, 0.0] for _ in request['utterances']]}), flush=True)
    ''')
    worker = MemoWorker(sys.executable, timeout_s=2, worker_script=script)
    assert worker.predict(depth=0, beta=1, utterances=['green', 'round']) == [
        [0.5, 0.5, 0.0], [0.5, 0.5, 0.0],
    ]
    process = worker._process
    worker.close()
    assert worker._process is None
    assert process.poll() == 0


def test_python_311_is_rejected_with_a_clear_worker_error():
    worker = MemoWorker(sys.executable, timeout_s=2)
    with pytest.raises(MemoWorkerError, match='requires Python >=3.12'):
        worker.predict(depth=0, beta=1, utterances=['green'])
    assert worker._process is None


def test_worker_errors_are_explicit_and_failure_closes_process(tmp_path):
    script = fake_worker_script(tmp_path, '''
        import json, sys
        for line in sys.stdin:
            request = json.loads(line)
            if request['op'] == 'shutdown': break
            if request['op'] == 'hello':
                print(json.dumps({'ok': True, 'python_major_minor': [3, 12],
                    'memo_version': 'fixture', 'jax_version': 'fixture'}), flush=True)
            else:
                print(json.dumps({'ok': False, 'error': 'ValueError: malformed batch'}), flush=True)
    ''')
    worker = MemoWorker(sys.executable, timeout_s=2, worker_script=script)
    with pytest.raises(MemoWorkerError, match='malformed batch'):
        worker.predict(depth=0, beta=1, utterances=['green'])
    assert worker._process is None


def test_worker_timeout_kills_and_cleans_up_child(tmp_path):
    script = fake_worker_script(tmp_path, '''
        import time
        time.sleep(5)
    ''')
    worker = MemoWorker(sys.executable, timeout_s=0.05, worker_script=script)
    with pytest.raises(MemoWorkerTimeout):
        worker.predict(depth=0, beta=1, utterances=['green'])
    assert worker._process is None


@pytest.mark.skipif(not os.environ.get('MEMO_PYTHON'),
                    reason='Set MEMO_PYTHON to a Python >=3.12 memo/JAX environment')
def test_direct_memo_and_umi_tables_are_exact_and_match_published_rounding():
    for depth, published in ((0, LITERAL), (1, RECURSIVE)):
        direct = MemoWorker(os.environ['MEMO_PYTHON'])
        direct_rows = np.asarray(direct.predict(
            depth=depth, beta=1.0, utterances=list(UTTERANCES)))
        direct.close()
        subject = MemoSubject(config=MemoConfig(depth=depth, beta=1.0),
                              python_executable=os.environ['MEMO_PYTHON'])
        session = BufferedSession(inputs(), requested_output_channels=('behavior',))
        subject.interact(session)
        umi_rows = session.collect('behavior').values
        np.testing.assert_array_equal(direct_rows, umi_rows)
        np.testing.assert_array_equal(np.round(direct_rows, 2), published)
        subject.reset()
