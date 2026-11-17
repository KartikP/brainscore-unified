"""Reasoning records preserve exposed text, timing, context, and failure status."""
from contextlib import contextmanager
import json

import pytest
from brainscore.experiments import (
    CallableProtocol, Experiment, RecordInputsOutputs, RecordReasoning, SessionProtocol,
)
from brainscore.run_record import RunRecord
from brainscore_core.contract import Subject
from brainscore_core.streaming import Session, StreamEvent

pytestmark = pytest.mark.unit


@contextmanager
def session(trial):
    yield Session([StreamEvent('text', '6 * 7?', 10)], ['behavior:text'])


def protocol():
    return SessionProtocol(
        'reasoning', session, conditions=['normal', 'repeat'], repetitions=[0, 1],
        required_input_channels=['text'], requested_output_channels=['behavior:text'],
    )


def run(tmp_path, subject, *, procedure=None, tool=None):
    return Experiment(
        subject=subject, protocol=procedure or protocol(),
        tools=[RecordInputsOutputs(), tool or RecordReasoning()],
        output_dir=tmp_path / 'run',
    ).run()


class Replier(Subject):
    """Answers on behavior:text; ``meta`` is the extra data of each answer event."""
    identifier = 'fixture'
    in_channels = frozenset({'text'})
    out_channels = frozenset({'behavior:text'})

    def __init__(self, text, meta):
        self.text, self.meta = text, meta

    def participate(self, session):
        while (question := session.next_input()) is not None:
            session.respond(StreamEvent('behavior:text', self.text, question.t_ms, self.meta))

    def reset(self):
        pass


def answering(text='42', **meta):
    """A subject whose answer event carries ``meta`` (for example ``reasoning``)."""
    return Replier(text, meta)


def test_completed_reasoning_preserves_response_and_input_links(tmp_path):
    meta = {'reasoning': {'text': '6 times 7 is 42.', 'kind': 'chain_of_thought'}}
    result = run(tmp_path, Replier('42', meta))
    records = list(RunRecord(result.directory / 'reasoning').outputs())
    assert len(records) == 4
    assert [(r['condition'], r['repetition_id']) for r in records] == [
        ('normal', 0), ('normal', 1), ('repeat', 0), ('repeat', 1)]
    inputs = {e['event_id'] for e in result.record.events() if e['kind'] == 'input'}
    for record in records:
        assert record['event_id'] in inputs
        assert record['payload'].t_ms == 10
        assert record['payload'].payload == '42'
        assert record['payload'].meta == meta
        assert record['reasoning'] == [{'text': '6 times 7 is 42.', 'kind': 'chain_of_thought', 'format': 'complete'}]
    meta['reasoning']['text'] = 'changed after the run'
    assert list(RunRecord(result.directory / 'reasoning').outputs())[0]['reasoning'][0]['text'] == '6 times 7 is 42.'
    assert {a['path'] for a in result.manifest['artifacts']} >= {
        'reasoning/manifest.json', 'reasoning/events.jsonl'}


@pytest.mark.parametrize('reasoning,available,blocks', [
    (None, False, []), ('', True, [{'text': '', 'kind': 'unspecified', 'format': 'complete'}]),
    ([], True, []),
])
def test_absent_is_different_from_empty(tmp_path, reasoning, available, blocks):
    result = run(tmp_path, answering(reasoning=reasoning))
    record = next(RunRecord(result.directory / 'reasoning').outputs())
    assert record['reasoning_available'] is available
    assert record['reasoning'] == blocks


def test_answer_is_never_automatically_labeled_reasoning(tmp_path):
    result = run(tmp_path, answering('First compute 6 * 7, giving 42.'))
    record = next(RunRecord(result.directory / 'reasoning').outputs())
    assert record['reasoning_available'] is False
    assert record['payload'].payload == 'First compute 6 * 7, giving 42.'


class StreamingSubject(Subject):
    """Answers in fragments: three events with reasoning in their meta, then the answer."""
    identifier = 'stream-fixture'
    in_channels = {'text'}
    out_channels = {'behavior:text'}

    def __init__(self, fail=False):
        self.fail = fail
        self.resets = 0

    def reset(self):
        self.resets += 1

    def participate(self, session):
        while (request := session.next_input()) is not None:
            for offset, block in enumerate([
                {'text': '6 times ', 'format': 'delta', 'kind': 'chain_of_thought'},
                {'text': '7 is 42.', 'format': 'delta', 'kind': 'chain_of_thought'},
                {'text': '6 times 7 is 42.', 'format': 'snapshot', 'kind': 'summary'},
            ]):
                session.respond(StreamEvent('behavior:text', '', request.t_ms + offset,
                                            {'reasoning': block}))
                if self.fail:
                    raise RuntimeError('provider disconnected')
            session.respond(StreamEvent('behavior:text', '42', 20, {}))


def test_stream_fragments_keep_order_and_are_not_combined(tmp_path):
    result = run(tmp_path, StreamingSubject())
    records = list(RunRecord(result.directory / 'reasoning').outputs())
    assert len(records) == 16
    assert len({r['event_id'] for r in records[:4]}) == 1
    assert [r['payload'].t_ms for r in records[:4]] == [10, 11, 12, 20]
    assert [r['reasoning'][0]['format'] for r in records[:3]] == ['delta', 'delta', 'snapshot']
    assert records[2]['reasoning'][0]['kind'] == 'summary'
    assert records[3]['payload'].payload == '42'
    assert records[3]['reasoning_available'] is False
    assert [r['sequence'] for r in records] == sorted(r['sequence'] for r in records)


def test_partial_stream_remains_readable_and_failed(tmp_path):
    subject = StreamingSubject(fail=True)
    with pytest.raises(RuntimeError, match='provider disconnected'):
        run(tmp_path, subject)
    record = RunRecord(tmp_path / 'run/reasoning')
    assert record.manifest['status'] == 'failed'
    assert next(record.outputs())['reasoning'][0]['text'] == '6 times '
    assert subject.resets == 2
    assert json.loads((tmp_path / 'run/experiment.json').read_text())['status'] == 'failed'
    with pytest.raises(ValueError, match='completed'):
        record.evaluate(lambda *args: 0, None)


def test_custom_selector_works_with_callable_protocol(tmp_path):
    class Model:
        identifier = 'custom-format'
        def infer(self, request):
            return {'answer': 42, 'thinking': 'Multiplication gives 42.'}
    subject = Model()
    procedure = CallableProtocol('external', lambda model, context: model.infer('6 * 7?'),
                                 methods=['infer'])
    result = run(tmp_path, subject, procedure=procedure,
                 tool=RecordReasoning(extract=lambda output: output['thinking']))
    record = next(RunRecord(result.directory / 'reasoning').outputs())
    assert record['payload']['answer'] == 42
    assert record['reasoning'][0]['text'] == 'Multiplication gives 42.'
    assert 'infer' not in vars(subject)  # The original method is restored.


@pytest.mark.parametrize('reasoning', [42, {'text': 42}, {'text': 'x', 'format': 'guess'}, {'text': 'x', 'kind': ''}])
def test_malformed_reasoning_fails_explicitly(tmp_path, reasoning):
    with pytest.raises((TypeError, ValueError), match='Reasoning'):
        run(tmp_path, answering(reasoning=reasoning))
    assert RunRecord(tmp_path / 'run/reasoning').manifest['status'] == 'failed'
    assert RunRecord(tmp_path / 'run/inputs_outputs').manifest['status'] == 'failed'


def test_requires_input_recorder_before_execution(tmp_path):
    experiment = Experiment(subject=answering(), protocol=protocol(),
                            tools=[RecordReasoning()], output_dir=tmp_path / 'run')
    with pytest.raises(ValueError, match='RecordInputsOutputs'):
        experiment.run()
    assert not experiment.output_dir.exists()


def test_reasoning_from_a_reply_part_reaches_the_reasoning_record(tmp_path):
    """A normal reply= part returns its reasoning as extra data; the record keeps it."""
    from brainscore import BrainScoreModel
    from brainscore_core.contract import TaskContext
    from brainscore_core.responses import Text
    from brainscore_core.streaming_helpers import stimulus_session
    from brainscore_core.supported_data_standards.brainio.stimuli import StimulusSet

    def reply(prompt):
        return StreamEvent('behavior:text', '42', None,
                           {'reasoning': '6 times 7 is 42.', 'token_usage': {'output_tokens': 3}})

    @contextmanager
    def task_session(trial):
        stimuli = StimulusSet({'stimulus_id': ['q'], 'text': ['6 * 7?']})
        stimuli.identifier = 'reply-reasoning'
        yield stimulus_session(stimuli, ['behavior:text'], TaskContext(response=Text()))

    procedure = SessionProtocol('reasoning', task_session, conditions=['normal'], repetitions=[0],
                                required_input_channels=['text'], requested_output_channels=['behavior:text'])
    subject = BrainScoreModel(identifier='replier', in_channels={'text'}, reply=reply)
    result = run(tmp_path, subject, procedure=procedure)
    record, = RunRecord(result.directory / 'reasoning').outputs()
    assert record['reasoning_available'] is True
    assert record['reasoning'] == [{'text': '6 times 7 is 42.', 'kind': 'unspecified',
                                    'format': 'complete'}]
    assert record['payload'].meta['token_usage'] == {'output_tokens': 3}
    assert list(record['payload'].payload.values) == ['42']


def test_multi_trial_reasoning_skips_trials_without_reasoning(tmp_path):
    """One answer event for three trials: the trials with reasoning are recorded, None is skipped."""
    result = run(tmp_path, answering(reasoning=['x is short', None, 'zzz is short']))
    record = next(RunRecord(result.directory / 'reasoning').outputs())
    assert record['reasoning_available'] is True
    assert [block['text'] for block in record['reasoning']] == ['x is short', 'zzz is short']
    result = run(tmp_path / 'none', answering(reasoning=[None, None]))
    record = next(RunRecord(result.directory / 'reasoning').outputs())
    assert record['reasoning_available'] is False and record['reasoning'] == []


def test_multi_trial_reasoning_accepts_a_list_of_blocks_for_a_trial(tmp_path):
    """Two trials: the first has two blocks, the second has none. All blocks stay, in order."""
    first = [{'text': 'step one', 'kind': 'chain_of_thought'}, 'step two']
    result = run(tmp_path, answering(reasoning=[first, None]))
    record = next(RunRecord(result.directory / 'reasoning').outputs())
    assert record['reasoning_available'] is True
    assert [block['text'] for block in record['reasoning']] == ['step one', 'step two']
    assert record['reasoning'][0]['kind'] == 'chain_of_thought'
