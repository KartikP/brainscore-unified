"""A raw event stream with no table gives its picture or clip to the subject (core rebuild of a trial).

Core rebuilds each trial of a source with no table from its events. An event
that names no column gets the catalog's column for its channel
(``image_file_name`` for ``vision:image``, ``video_path`` for ``vision:video``),
so the prompt layout, an answer part and an extractor see the part, as they do in
a table. Before, the column was the channel's name, which no reader knew: a task
that named the part in ``present`` was refused, and without ``present`` the
picture was lost silently. Sound and text keep their names; an event that names
its column keeps it.
"""
import pytest

from brainscore import BrainScoreModel
from brainscore_core.contract import TaskContext
from brainscore_core.responses import Choice
from brainscore_core.streaming import Session, StreamEvent
from tests.brainscore_model.test_session_plan_recordings import _Recordable

pytestmark = pytest.mark.unit

PAYLOAD = {'vision:image': 'pic.png', 'vision:video': 'clip.mp4', 'audio': 'sound.wav', 'text': 'real a'}
KIND = {'vision:image': 'image', 'vision:video': 'video', 'audio': 'audio', 'text': 'text'}


def _events(channel, **meta):
    return [StreamEvent(channel, PAYLOAD[channel], float(i), {'stimulus_id': sid, **meta})
            for i, sid in enumerate(('s0', 's1'))]


def _replies(channel):
    seen = []
    model = BrainScoreModel('replies', in_channels={channel}, reply=lambda prompt: seen.append(prompt.parts) or 'cat')
    return model, seen


@pytest.mark.parametrize('channel', ['vision:image', 'vision:video', 'audio'])
@pytest.mark.parametrize('present', [True, False], ids=['present', 'no present'])
def test_a_reply_part_receives_the_part_of_a_raw_event(channel, present):
    model, seen = _replies(channel)
    task = TaskContext(instruction='Which?', response=Choice(['cat', 'dog']),
                       present=(KIND[channel],) if present else None)
    model.participate(Session(_events(channel), ['behavior:choice'], task))
    assert [parts[1] for parts in seen] == [(KIND[channel], PAYLOAD[channel])] * 2


@pytest.mark.parametrize('channel, column', [('vision:image', 'image_path'), ('vision:video', 'video_file_name')])
def test_an_event_that_names_its_column_keeps_it(channel, column):
    model, seen = _replies(channel)
    task = TaskContext(instruction='Which?', response=Choice(['cat', 'dog']))
    model.participate(Session(_events(channel, column=column), ['behavior:choice'], task))
    assert [parts[1] for parts in seen] == [(KIND[channel], PAYLOAD[channel])] * 2


def test_an_answer_part_reads_the_picture_of_a_raw_event_by_the_catalog():
    from brainscore.model_helpers.api_behavioral import _find_column
    rows = []
    model = BrainScoreModel('answers', in_channels={'vision:image'},
                            answer=lambda stimulus_row, instruction, label_set: rows.append(stimulus_row) or label_set[0])
    task = TaskContext(instruction='Which?', response=Choice(['cat', 'dog']))
    model.participate(Session(_events('vision:image'), ['behavior:choice'], task))
    assert [_find_column(row, 'image') for row in rows] == ['image_file_name'] * 2
    assert [row['image_file_name'] for row in rows] == ['pic.png'] * 2


@pytest.mark.parametrize('channel, key', [('vision:image', 'vision'), ('vision:video', 'video')])
def test_an_extractor_records_a_region_from_a_raw_event(channel, key):
    log = []
    model = BrainScoreModel('records', extractor={key: _Recordable(key, log)}, region_layer_map={'IT': 'a'})
    session = Session(_events(channel), ['neural:IT'])
    model.participate(session)
    assert log == [(key, 'call', ('a',))]
    assert session.collect('neural:IT').sizes['presentation'] == 2


def test_a_region_is_recorded_while_a_reply_answers_a_raw_event():
    log = []
    model = BrainScoreModel('records', extractor={'vision': _Recordable('image', log)}, region_layer_map={'IT': 'a'},
                            reply=lambda prompt: 'cat')
    task = TaskContext(instruction='Which?', response=Choice(['cat', 'dog']))
    session = Session(_events('vision:image'), ['behavior:choice', 'neural:IT'], task)
    model.participate(session)
    assert ('image', 'record', ('a',)) in log
    assert session.collect('neural:IT').sizes['presentation'] == 2


def test_a_picture_and_a_text_without_present_are_refused_as_in_a_table():
    """Before, the picture was lost silently and the text alone was shown."""
    model, seen = _replies('vision:image')
    model = BrainScoreModel('replies', in_channels={'vision:image', 'text'}, reply=lambda prompt: seen.append(1) or 'cat')
    events = [StreamEvent('vision:image', 'pic.png', 0.0, {'stimulus_id': 's0'}),
              StreamEvent('text', 'real a', 0.0, {'stimulus_id': 's0'})]
    task = TaskContext(instruction='Which?', response=Choice(['cat', 'dog']))
    with pytest.raises(ValueError, match=r"Each stimulus has several parts \['image', 'text'\]"):
        model.participate(Session(events, ['behavior:choice'], task))
    assert seen == []
