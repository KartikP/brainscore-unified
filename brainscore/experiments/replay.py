"""Explicit same-input replay and comparison of completed experiment records."""
from contextlib import contextmanager
from pathlib import Path
import json
import numpy as np
from brainscore.run_record import RunRecord
from brainscore_core.streaming import InMemorySession, StreamEvent
from .runner import SessionProtocol, CallableProtocol


def read_events(directory):
    """Load verified events only from a completed experiment and recorder."""
    directory = Path(directory)
    manifest = json.loads((directory / 'experiment.json').read_text())
    if manifest['status'] != 'complete':
        raise ValueError('Replay requires a completed experiment')
    record = RunRecord(directory / 'inputs_outputs')
    if record.manifest['status'] != 'complete':
        raise ValueError('Replay requires a completed record')
    return [row['payload'] for row in record.events()]


def _source(directory):
    path = Path(directory)
    record = RunRecord(path / 'inputs_outputs')
    return {'source': str(path.resolve()), 'source_record_id': record.manifest['run_id'],
            'source_events_sha256': record.manifest['events_sha256']}


def replay_sessions(directory):
    """Replay saved session inputs, preserving trials and event timestamps.

    This is open-loop: outputs cannot change the saved future observations.
    It does not restart a simulator or reproduce a conversational environment.
    The normal SessionProtocol reset policy applies to each trial.
    """
    events = read_events(directory)
    trials = [e['trial_id'] for e in events if e['kind'] == 'trial_start']
    inputs = {trial: [] for trial in trials}
    outputs = set()
    for event in events:
        if event['kind'] == 'input':
            if not isinstance(event['payload'], StreamEvent):
                raise TypeError('Use replay_calls for recorded method calls')
            inputs[event['trial_id']].append(event['payload'])
        elif event['kind'] == 'output':
            if isinstance(event['payload'], StreamEvent):
                outputs.add(event['payload'].channel)
    @contextmanager
    def factory(trial):
        # Independent copies prevent a mutating subject altering another replay.
        from copy import deepcopy
        yield InMemorySession(deepcopy(inputs[trial]))
    return SessionProtocol('saved-session-inputs', factory, trials=trials,
        input_channels={e.channel for group in inputs.values() for e in group},
        output_channels=outputs,
        metadata={**_source(directory), 'mode': 'open_loop_saved_inputs'})


def replay_calls(directory, *, methods, reset):
    """Replay explicit allowlisted methods with caller-supplied reset behavior.

    reset(subject) runs before and after the saved call sequence. Use a no-op
    only for a known stateless provider. Restart remote policy RNG separately
    when its API offers no reset. Never infer reset behavior from a method name.
    """
    events = read_events(directory)
    requests = [e['payload'] for e in events if e['kind'] == 'input']
    if not requests:
        raise ValueError('No saved calls')
    for request in requests:
        if not isinstance(request, dict) or request.get('method') not in methods:
            raise ValueError('Saved method is not explicitly allowed')
    def evaluate(subject, context):
        from copy import deepcopy
        results = []
        try:
            reset(subject)
            for request in requests:
                request = deepcopy(request)
                results.append(getattr(subject, request['method'])(*request['args'], **request['kwargs']))
        finally:
            reset(subject)
        return results
    return CallableProtocol('saved-method-inputs', evaluate, methods=methods,
        metadata={**_source(directory), 'mode': 'open_loop_saved_calls',
                  'reset_policy': 'caller supplied; before and after sequence'})


def _equal(left, right):
    if isinstance(left, StreamEvent) and isinstance(right, StreamEvent):
        return (left.channel == right.channel and left.t_ms == right.t_ms
                and _equal(left.meta, right.meta) and _equal(left.payload, right.payload))
    if isinstance(left, np.ndarray) or isinstance(right, np.ndarray):
        return (isinstance(left, np.ndarray) and isinstance(right, np.ndarray)
                and left.dtype == right.dtype and np.array_equal(
                    left, right, equal_nan=left.dtype.kind in 'fc'))
    if type(left) is not type(right):
        return False
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(_equal(left[k], right[k]) for k in left)
    if isinstance(left, (tuple, list)):
        return len(left) == len(right) and all(_equal(a, b) for a, b in zip(left, right))
    result = left == right
    if not isinstance(result, (bool, np.bool_)):
        raise TypeError('Supply a domain metric for unsupported comparison payloads')
    return bool(result)


def compare_outputs(reference, candidate):
    """Exact sequential output comparison, not a benchmark score or causal claim."""
    left = [e['payload'] for e in read_events(reference) if e['kind'] == 'output']
    right = [e['payload'] for e in read_events(candidate) if e['kind'] == 'output']
    different = [i for i, (a, b) in enumerate(zip(left, right)) if not _equal(a, b)]
    different.extend(range(min(len(left), len(right)), max(len(left), len(right))))
    return {'reference_outputs': len(left), 'candidate_outputs': len(right),
            'different_indices': different, 'equal': bool(left) and not different,
            'scope': 'exact sequential outputs; not task success or simulator equivalence'}
