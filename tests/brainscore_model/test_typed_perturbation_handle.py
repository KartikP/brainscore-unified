"""score() sees a perturbation left active by a subject that answers with typed records."""
import pytest

from brainscore import score
from brainscore_core.contract import Subject
from brainscore_core.events import Perturbation, PerturbationApplied, Selection, StateChange
from brainscore_core.metrics import Score
from brainscore_core.streaming import StreamEvent
from brainscore_core.streaming_helpers import apply_state_change


class _Benchmark:
    identifier = 'applies-a-lesion'
    version = 1
    required_input_channels = {'text'}
    requested_output_channels = {'perturbation'}

    def __init__(self, undo):
        self.undo = undo

    def __call__(self, subject):
        applied = apply_state_change(subject, StateChange(
            kind='ablation', target=Selection(layer='x'), perturbation=Perturbation(kind='zero')))
        if self.undo:
            apply_state_change(subject, StateChange(kind='reset', handle_id=applied.handle_id))
        return Score(1.0)


class _TypedLesionSubject(Subject):
    """Answers a perturbation session with a typed payload and default metadata."""
    identifier = 'typed-lesion'
    in_channels = {'text'}
    out_channels = {'perturbation'}

    def participate(self, session):
        change = session.next_input().payload
        session.respond(StreamEvent(channel='perturbation', t_ms=0.0, payload=PerturbationApplied(
            change.handle_id or 'h1', change.target, change.perturbation)))


def test_a_typed_response_without_metadata_still_counts_as_left_active():
    with pytest.raises(RuntimeError, match='left intervention handle'):
        score(_TypedLesionSubject(), _Benchmark(undo=False), check_mem=False)
    score(_TypedLesionSubject(), _Benchmark(undo=True), check_mem=False)
