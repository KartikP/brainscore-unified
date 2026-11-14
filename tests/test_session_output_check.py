"""A benchmark session that requests neural:IT refuses a subject that answers on neural:V4."""
import numpy as np
import pytest

from brainscore_core import Benchmark, Score, Subject
from brainscore_core.streaming import Session, StreamEvent
from brainscore_core.supported_data_standards.brainio.assemblies import NeuroidAssembly

pytestmark = pytest.mark.unit


def Answers(channel):
    """A subject that answers every text on ``channel``."""
    def participate(self, session):
        while (event := session.next_input()) is not None:
            if event.channel == 'text':
                session.respond(StreamEvent(channel, NeuroidAssembly(
                    np.ones((1, 2)), dims=('presentation', 'neuroid'),
                    coords={'stimulus_id': ('presentation', ['a']), 'neuroid_id': ('neuroid', [0, 1])}),
                    event.t_ms))
    return type('Answers', (Subject,), dict(
        identifier=f'answers-{channel}', in_channels={'text'}, out_channels={channel},
        participate=participate))()


class AskForIT(Benchmark):
    identifier = 'ask-for-it'
    parent = 'neural'
    uses_session = True
    required_input_channels = {'text'}
    requested_output_channels = {'neural:IT'}

    def __call__(self, subject):
        session = Session([StreamEvent('text', 'hello', 0.0)], ['neural:IT'])
        subject.participate(session)
        return Score(float(session.collect('neural:IT').sum()))


def test_a_subject_that_answers_the_requested_region_is_scored():
    assert float(AskForIT()(Answers('neural:IT'))) == 2.0


def test_a_subject_that_answers_another_region_is_refused():
    with pytest.raises(ValueError, match="'neural:V4' was not requested"):
        AskForIT()(Answers('neural:V4'))
