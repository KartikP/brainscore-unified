"""A subject keeps no run state between sessions (Contract Rules 4.4, 4.5): after any session A,
a session B gives what it gives on a fresh subject, and each session keeps the source it was given.

Every subject below runs with no weights and no download. Each probe session is one a caller can
give to ``participate``; a refusal is an outcome and must be the same refusal as on a fresh subject."""
import numpy as np
import pytest

from brainscore import load_model
from brainscore.harnesses.human import HumanSubject
from brainscore_core.contract import TaskContext
from brainscore_core.responses import Choice
from brainscore_core.streaming_helpers import stimulus_session
from brainscore_core.supported_data_standards.brainio.assemblies import (
    BehavioralAssembly, NeuroidAssembly)
from brainscore_core.supported_data_standards.brainio.stimuli import StimulusSet

pytestmark = pytest.mark.unit


def _stimuli():
    stimuli = StimulusSet({'stimulus_id': ['s0', 's1'], 'image_path': ['s0.png', 's1.png']})
    stimuli.identifier = 'no-run-state'
    return stimuli


# The probe sessions: (outputs, task). Each pair (A, B) of them runs on one subject.
PROBES = (
    (['behavior'], TaskContext(task_type='probabilities', label_set=['cat', 'dog'])),
    (['behavior:choice'], TaskContext(response=Choice(['cat', 'dog']))),
    (['behavior:probabilities'], TaskContext(response=Choice(['cat', 'dog']))),
    (['neural:IT'], TaskContext(task_type='passive')),
)


def _rig():
    def record(region, table):
        ids = list(table['stimulus_id'])
        return NeuroidAssembly(np.ones((len(ids), 1)), dims=['presentation', 'neuroid'],
                               coords={'stimulus_id': ('presentation', ids), 'neuroid_id': ('neuroid', ['n'])})

    def respond(task, table):
        ids = list(table['stimulus_id'])
        return BehavioralAssembly(np.array(['cat'] * len(ids)), dims=['presentation'],
                                  coords={'stimulus_id': ('presentation', ids)})
    return HumanSubject('rig', region_layer_map={'IT': 'array'}, present_fn=lambda table: None,
                        record_fn=record, respond_fn=respond)


SUBJECTS = {
    'chance-baseline': lambda: load_model('chance-baseline'),
    'human-harness': _rig,
}


def _run(subject, probe):
    """What one session gives: its answers, or the kind of its refusal. The session must keep the
    source it was built with."""
    outputs, task = probe
    session = stimulus_session(_stimuli(), outputs, task)
    source = session.source
    try:
        subject.participate(session)
        outcome = [(event.channel, str(getattr(event.payload, 'values', event.payload)))
                   for event in session.emitted]
    except Exception as refusal:
        outcome = type(refusal).__name__
    assert session.source is source, f'{subject.identifier} replaced the source of the session'
    return outcome


def _second_session_gives_what_a_fresh_subject_gives(make, first, second):
    """(what ``second`` gives after ``first`` on one subject, what it gives on a fresh subject)."""
    used = make()
    first(used)
    return second(used), second(make())


@pytest.mark.parametrize('name', sorted(SUBJECTS))
def test_a_second_session_gives_what_a_fresh_subject_gives(name):
    for a in PROBES:
        for b in PROBES:
            after_a, fresh = _second_session_gives_what_a_fresh_subject_gives(
                SUBJECTS[name], lambda subject: _run(subject, a), lambda subject: _run(subject, b))
            assert after_a == fresh, f'{name}: {b[0]} after {a[0]} gives {after_a}; a fresh subject gives {fresh}'
