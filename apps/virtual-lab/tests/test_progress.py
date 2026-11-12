"""Progress observes public events without inspecting or splitting model payloads."""
from types import SimpleNamespace

from umi_lab.experiment import progress_tool


def test_live_phases_count_only_successful_repetitions_and_throttle_inputs(monkeypatch):
    from brainscore_core.streaming import StreamEvent
    updates = []
    now = [10.]
    monkeypatch.setattr('umi_lab.experiment.time.monotonic', lambda: now[0])
    observer = progress_tool(updates.append, lambda: False, activity='Viewing stimuli and generating responses')
    observer.validate(SimpleNamespace(protocol=SimpleNamespace(
        conditions=['intact', 'lesioned'], repetitions=[0, 1], output_channels={'neural:IT', 'behavior:text'})))

    def event(kind, payload=None):
        observer.on_event(dict(kind=kind, payload=payload, condition='intact', repetition_id=0))

    event('repetition_start')
    event('session_context', {})
    for _ in range(1000):
        event('input', StreamEvent('vision:image', object(), None))
    assert [update['activity'] for update in updates] == [
        'Preparing repetition', 'Viewing stimuli and generating responses']
    assert updates[-1]['progress']['completed'] == 0
    # These opaque payloads must never be read or serialized by telemetry.
    event('output', StreamEvent('neural:IT', object(), None))
    assert updates[-1]['activity'] == 'Recording neural activity'
    now[0] += .6
    event('output', StreamEvent('behavior:text', object(), None))
    assert updates[-1]['activity'] == 'Recording responses'
    now[0] += .6
    event('repetition_end', {'failed': True})
    assert updates[-1]['progress']['completed'] == 0
    now[0] += .6
    event('repetition_end', {'failed': False})
    assert updates[-1]['progress'] == {'completed': 1, 'total': 4, 'unit': 'repetition'}
    now[0] += .6
    event('repetition_end', {'failed': False})
    assert updates[-1]['progress']['completed'] == 1
    assert all(update['recording_neural'] for update in updates)
