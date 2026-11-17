"""Contract and transport checks without loading the released GPU model."""

from contextlib import contextmanager
import json
from pathlib import Path
import sys
import time

import numpy as np
import pytest

from brainscore.experiments import (
    Experiment, RecordInputsOutputs, SessionProtocol, compare_outputs, read_events,
    replay_sessions,
)
from brainscore.models.mirage import MirageRuntime, MirageSubject
from brainscore.models.mirage.model import OUTPUT_CHANNEL, Prediction
from brainscore.models.mirage.runtime import MirageRuntimeError, MirageTimeout
from brainscore_core.contract import Subject
from brainscore_core.streaming import Session, Source, StreamEvent

pytestmark = pytest.mark.unit


class NativeFixture:
    def __init__(self):
        self.calls = []

    def __call__(self, clip):
        self.calls.append(clip)
        value = sum(clip.video.read_bytes())
        return Prediction(
            values=np.full((2, 1000), value, dtype=np.float32),
            provenance={'participant': 'sub-01', 'implementation': 'test fixture'},
        )


def inputs(video, *, identifier='clip'):
    meta = {'stimulus_id': identifier}
    return [
        StreamEvent('vision:video', video, None, meta.copy()),
        StreamEvent('audio', video, None, meta.copy()),
        StreamEvent('text', 'hello', None, {
            **meta, 'transcript': [{'word': 'hello', 'onset': 0.0, 'duration': 0.5}],
        }),
    ]


@pytest.fixture
def video(tmp_path):
    path = tmp_path / 'clip.mp4'
    path.write_bytes(b'fixture container; never decoded as actual media')
    return path


def test_native_subject_satisfies_subject_through_participate(video):
    subject = MirageSubject(predictor=NativeFixture())
    assert isinstance(subject, Subject)
    assert 'participate' in vars(MirageSubject)
    assert not any('interact' in vars(cls) for cls in MirageSubject.__mro__)
    session = Session(inputs(video), (OUTPUT_CHANNEL,))
    subject.participate(session)
    assert [event.channel for event in session.emitted] == [OUTPUT_CHANNEL]


def test_subject_preserves_native_values_dtype_labels_and_time_grid(video):
    predictor = NativeFixture()
    subject = MirageSubject(predictor=predictor)
    assert subject.in_channels == subject.required_channels == {'vision:video', 'audio', 'text'}
    assert subject.out_channels == {OUTPUT_CHANNEL}
    session = Session(inputs(video), (OUTPUT_CHANNEL,))
    subject.participate(session)
    assembly = session.collect(OUTPUT_CHANNEL)
    assert assembly.dims == ('presentation', 'neuroid')
    assert assembly.dtype == np.float32
    assert assembly.attrs['signal_type'] == 'predicted_BOLD'
    assert list(assembly.tr_index.values) == [0, 1]
    assert list(assembly.relative_time_ms.values) == [0, 1490]
    assert list(assembly.parcel_index.values) == list(range(1000))
    assert list(assembly.stimulus_id.values) == ['clip:tr-0', 'clip:tr-1']
    assert session.emitted[0].t_ms is None
    np.testing.assert_array_equal(assembly.values, predictor(predictor.calls[0]).values)


@pytest.mark.parametrize('defect', [
    'duplicate', 'incomplete', 'missing_id', 'wrong_id', 'wrong_audio', 'wrong_text',
    'negative_onset', 'bad_word', 'mismatched_onset', 'unknown_output',
])
def test_malformed_clip_fails_before_native_compute(video, tmp_path, defect):
    events = inputs(video)
    requested = OUTPUT_CHANNEL
    if defect == 'duplicate':
        events.insert(1, events[0])
    elif defect == 'incomplete':
        events.pop()
    elif defect == 'missing_id':
        events[0].meta.clear()
    elif defect == 'wrong_id':
        events[1].meta['stimulus_id'] = 'other'
    elif defect == 'wrong_audio':
        other = tmp_path / 'other.mp4'
        other.write_bytes(b'different')
        events[1].payload = other
    elif defect == 'wrong_text':
        events[2].payload = 'different text'
    elif defect == 'negative_onset':
        events[2].meta['transcript'][0]['onset'] = -1
    elif defect == 'bad_word':
        events[2].meta['transcript'][0]['word'] = ''
    elif defect == 'mismatched_onset':
        events[0].t_ms = 100.0
    elif defect == 'unknown_output':
        requested = 'neural:IT'
    predictor = NativeFixture()
    subject = MirageSubject(predictor=predictor)
    session = Session(events, (requested,))
    with pytest.raises((ValueError, TypeError)):
        subject.participate(session)
    assert predictor.calls == []
    assert session.emitted == []


def test_subject_answers_complete_clip_before_next_reactive_input(video):
    class Reactive(Source):
        def __init__(self):
            self.position = 0
            self.session = None

        def next_input(self):
            if self.position == 3:
                assert len(self.session.emitted) == 1
                return None
            event = inputs(video)[self.position]
            self.position += 1
            return event

    predictor = NativeFixture()
    subject = MirageSubject(predictor=predictor)
    for _ in range(2):
        source = Reactive()
        session = source.session = Session(source, (OUTPUT_CHANNEL,))
        subject.participate(session)
        assert len(session.emitted) == 1
        subject.reset()
    assert len(predictor.calls) == 2


def test_recording_copies_inputs_and_replay_reruns_without_original_file(video, tmp_path):
    predictor = NativeFixture()

    @contextmanager
    def factory(_trial):
        yield Session(inputs(video), (OUTPUT_CHANNEL,))

    original = Experiment(
        subject=MirageSubject(predictor=predictor),
        protocol=SessionProtocol('mirage-fixture', factory,
                                 required_input_channels=('vision:video', 'audio', 'text'),
                                 requested_output_channels=(OUTPUT_CHANNEL,)),
        tools=(RecordInputsOutputs(),), output_dir=tmp_path / 'record',
    ).run()
    video.unlink()
    replay = Experiment(
        subject=MirageSubject(predictor=predictor),
        protocol=replay_sessions(original.directory),
        tools=(RecordInputsOutputs(),), output_dir=tmp_path / 'replay',
    ).run()
    assert len(predictor.calls) == 2
    assert predictor.calls[1].video.exists()
    assert predictor.calls[1].video.suffix == ''
    assert compare_outputs(original.directory, replay.directory)['equal']
    outputs = [event['payload'] for event in read_events(original.directory)
               if event['kind'] == 'output']
    assert outputs[0].meta['provenance']['participant'] == 'sub-01'


def runtime(tmp_path, executable):
    return MirageRuntime(python_executable=str(executable), source_dir=tmp_path,
                         run_dir=tmp_path, work_dir=tmp_path / 'scratch', timeout_s=0.2)


@pytest.mark.parametrize('failure', ['timeout', 'error', 'missing_artifacts'])
def test_runtime_failure_timeout_and_temporary_file_cleanup(video, tmp_path, failure):
    executable = tmp_path / 'fake-python'
    body = {
        'timeout': 'import time; print("partial", flush=True); time.sleep(5)',
        'error': 'import sys; sys.stderr.write("fixture failure"); sys.exit(4)',
        'missing_artifacts': 'pass',
    }[failure]
    executable.write_text(f'#!{sys.executable}\n{body}\n')
    executable.chmod(0o755)
    from brainscore.models.mirage.model import _clip
    predictor = runtime(tmp_path, executable)
    clip = _clip({event.channel: event for event in inputs(video)})
    started = time.monotonic()
    error_type = MirageTimeout if failure == 'timeout' else MirageRuntimeError
    with pytest.raises(error_type):
        predictor(clip)
    assert time.monotonic() - started < 2
    assert list(predictor.work_dir.iterdir()) == []


def test_python311_worker_fails_clearly_without_importing_upstream(video, tmp_path):
    from brainscore.models.mirage.model import _clip
    predictor = runtime(tmp_path, sys.executable)
    predictor.timeout_s = 2
    with pytest.raises(MirageRuntimeError, match='requires Python 3.12'):
        predictor(_clip({event.channel: event for event in inputs(video)}))
    assert list(predictor.work_dir.iterdir()) == []


@pytest.mark.parametrize('defect', ['shape', 'dtype', 'nonfinite', 'wrong_participant'])
def test_invalid_native_output_is_not_emitted(video, defect):
    def predictor(_clip):
        values = np.ones((2, 1000), dtype=np.float32)
        participant = 'sub-01'
        if defect == 'shape':
            values = values[:, :999]
        elif defect == 'dtype':
            values = values.astype(np.float64)
        elif defect == 'nonfinite':
            values[0, 0] = np.nan
        elif defect == 'wrong_participant':
            participant = 'sub-02'
        return Prediction(values, {'participant': participant})

    subject = MirageSubject(predictor=predictor)
    session = Session(inputs(video), (OUTPUT_CHANNEL,))
    with pytest.raises(ValueError):
        subject.participate(session)
    assert session.emitted == []


def test_runtime_stages_media_suffix_and_returns_typed_result(video, tmp_path):
    executable = tmp_path / 'fake-python'
    executable.write_text(f'''#!{sys.executable}
import json, sys
from pathlib import Path
import numpy as np
request = json.loads(Path(sys.argv[2]).read_text())
assert Path(request['video']).suffix == '.mp4'
assert Path(request['video']).read_bytes().startswith(b'fixture')
assert json.loads(Path(request['transcript']).read_text())[0]['word'] == 'hello'
np.save(request['output'], np.ones((2, 1000), dtype=np.float32))
Path(request['provenance']).write_text(json.dumps({{'participant': request['participant']}}))
''')
    executable.chmod(0o755)
    from brainscore.models.mirage.model import _clip
    predictor = runtime(tmp_path, executable)
    predictor.timeout_s = 3
    result = predictor(_clip({event.channel: event for event in inputs(video)}))
    assert isinstance(result, Prediction)
    assert result.values.dtype == np.float32
    assert result.values.shape == (2, 1000)
    assert list(predictor.work_dir.iterdir()) == []


def test_timeout_stops_the_worker_and_its_decoder_child(video, tmp_path):
    import os
    import signal
    import psutil
    executable = tmp_path / 'fake-python'
    pid_file = tmp_path / 'decoder.pid'
    executable.write_text(f'''#!{sys.executable}
import subprocess, sys, time
from pathlib import Path
child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)'])
Path({str(pid_file)!r}).write_text(str(child.pid))
time.sleep(30)
''')
    executable.chmod(0o755)
    from brainscore.models.mirage.model import _clip
    predictor = runtime(tmp_path, executable)
    predictor.timeout_s = 0.5
    try:
        with pytest.raises(MirageTimeout):
            predictor(_clip({event.channel: event for event in inputs(video)}))
        child_pid = int(pid_file.read_text())
        deadline = time.monotonic() + 5.0
        while True:
            try:
                status = psutil.Process(child_pid).status()
            except psutil.NoSuchProcess:
                status = 'gone'
            if status in ('gone', psutil.STATUS_ZOMBIE) or time.monotonic() > deadline:
                break
            time.sleep(0.05)
        assert status in ('gone', psutil.STATUS_ZOMBIE)
        assert list(predictor.work_dir.iterdir()) == []
    finally:
        if pid_file.exists():
            try:
                os.kill(int(pid_file.read_text()), signal.SIGKILL)
            except ProcessLookupError:
                pass
