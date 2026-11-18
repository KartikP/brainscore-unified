"""observed() keeps each action proposal, payload, receipt and status in the record."""
import numpy as np
import pytest

from brainscore.instrumentation import observed
from brainscore_core.streaming import StreamEvent
from brainscore_core._sessions.environment import environment_session


class Env:
    def __init__(self, fail=False):
        self.fail = fail

    def reset(self):
        return {'observation': 0}

    def step(self, outputs):
        if self.fail:
            raise RuntimeError('execution failed')
        return {'observation': 1}, 'terminal'


class Sink:
    def __init__(self):
        self.outputs, self.errors = [], []

    def record(self, direction, event):
        if direction == 'output':
            self.outputs.append(event)

    def on_error(self, call, error):
        self.errors.append(error)


def run(env, action, validator=None):
    session = environment_session(env, action_validator=validator)
    session._request_control('test')
    sink = Sink()
    with session, observed(session, sink):
        while (event := session.next_input()) is not None:
            if event.payload == 'respond':
                try:
                    session.respond(StreamEvent('motor', action, event.t_ms, {}))
                except Exception:
                    pass
    return sink


def view(sink):
    return [(e.meta['action_status'], e.payload) for e in sink.outputs]


def test_successful_action_has_proposal_and_receipt_once():
    sink = run(Env(), np.array([2]))
    assert [s for s, _ in view(sink)] == ['proposed', 'applied']
    assert all(p[0] == 2 for _, p in view(sink)) and not sink.errors


def test_rejected_action_keeps_payload_and_status():
    def refuse(action):
        raise ValueError('refused')
    sink = run(Env(), np.array([2]), refuse)
    assert [s for s, _ in view(sink)] == ['proposed', 'rejected']
    assert sink.outputs[1].meta['error'] == 'refused' and sink.outputs[0].payload[0] == 2
    assert len(sink.errors) == 1


def test_failure_while_executing_keeps_unknown_status():
    sink = run(Env(fail=True), np.array([2]))
    assert [s for s, _ in view(sink)] == ['proposed', 'execution_unknown']
    assert sink.outputs[1].payload[0] == 2 and sink.outputs[1].meta['error_type'] == 'RuntimeError'
    assert [str(e) for e in sink.errors] == ['execution failed']
