"""Reasoning records preserve exposed text, timing, context, and failure status."""
from contextlib import contextmanager
import json

import pytest
from brainscore.experiments import (
    CallableProtocol, Experiment, RecordInputsOutputs, RecordReasoning, SessionProtocol,
)
from brainscore.model_helpers.response_trace import build_trace_subject
from brainscore.run_record import RunRecord
from brainscore_core.contract import Subject
from brainscore_core.streaming import InMemorySession, StreamEvent

pytestmark = pytest.mark.unit


@contextmanager
def session(trial):
    active = InMemorySession([StreamEvent('generation_request', {'prompt': '6 * 7?'}, 10)])
    active.requested_output_channels = ('response_trace',)
    yield active


def protocol():
    return SessionProtocol(
        'reasoning', session, conditions=['normal', 'repeat'], trials=[0, 1],
        input_channels=['generation_request'], output_channels=['response_trace'],
    )


def run(tmp_path, subject, *, procedure=None, tool=None):
    return Experiment(
        subject=subject, protocol=procedure or protocol(),
        tools=[RecordInputsOutputs(), tool or RecordReasoning()],
        output_dir=tmp_path / 'run',
    ).run()


def trace_subject(raw):
    return build_trace_subject('fixture', provider=lambda request: raw, parse=int,
                               provenance={'model': 'fixture', 'trained': False})


def test_completed_reasoning_preserves_response_and_input_links(tmp_path):
    raw = {'text': '42', 'reasoning': {'text': '6 times 7 is 42.', 'kind': 'chain_of_thought'}}
    result = run(tmp_path, trace_subject(raw))
    records = list(RunRecord(result.directory / 'reasoning').outputs())
    assert len(records) == 4
    assert [(r['condition'], r['trial_id']) for r in records] == [
        ('normal', 0), ('normal', 1), ('repeat', 0), ('repeat', 1)]
    inputs = {e['event_id'] for e in result.record.events() if e['kind'] == 'input'}
    for record in records:
        assert record['event_id'] in inputs
        assert record['payload'].t_ms == 10
        assert record['payload'].payload['answer'] == 42
        assert record['payload'].payload['raw'] == raw
        assert record['reasoning'] == [{'text': '6 times 7 is 42.', 'kind': 'chain_of_thought', 'format': 'complete'}]
    raw['reasoning']['text'] = 'changed after the run'
    assert list(RunRecord(result.directory / 'reasoning').outputs())[0]['reasoning'][0]['text'] == '6 times 7 is 42.'
    assert {a['path'] for a in result.manifest['artifacts']} >= {
        'reasoning/manifest.json', 'reasoning/events.jsonl'}


@pytest.mark.parametrize('reasoning,available,blocks', [
    (None, False, []), ('', True, [{'text': '', 'kind': 'unspecified', 'format': 'complete'}]),
    ([], True, []),
])
def test_absent_is_different_from_empty(tmp_path, reasoning, available, blocks):
    result = run(tmp_path, trace_subject({'text': '42', 'reasoning': reasoning}))
    record = next(RunRecord(result.directory / 'reasoning').outputs())
    assert record['reasoning_available'] is available
    assert record['reasoning'] == blocks


def test_answer_is_never_automatically_labeled_reasoning(tmp_path):
    result = run(tmp_path, trace_subject({'text': 'First compute 6 * 7, giving 42.'}))
    record = next(RunRecord(result.directory / 'reasoning').outputs())
    assert record['reasoning_available'] is False
    assert record['payload'].payload['valid'] is False
    assert record['payload'].payload['parse_error']


class StreamingSubject(Subject):
    identifier = 'stream-fixture'
    in_channels = {'generation_request'}
    out_channels = {'response_trace'}

    def __init__(self, fail=False):
        self.fail = fail
        self.resets = 0

    def reset(self):
        self.resets += 1

    def interact(self, session):
        while (request := session.next_input()) is not None:
            for offset, block in enumerate([
                {'text': '6 times ', 'format': 'delta', 'kind': 'chain_of_thought'},
                {'text': '7 is 42.', 'format': 'delta', 'kind': 'chain_of_thought'},
                {'text': '6 times 7 is 42.', 'format': 'snapshot', 'kind': 'summary'},
            ]):
                session.emit(StreamEvent('response_trace', {'raw': {'reasoning': block}}, request.t_ms + offset))
                if self.fail:
                    raise RuntimeError('provider disconnected')
            session.emit(StreamEvent('response_trace', {'raw': {'text': '42'}, 'answer': 42}, 20))


def test_stream_fragments_keep_order_and_are_not_combined(tmp_path):
    result = run(tmp_path, StreamingSubject())
    records = list(RunRecord(result.directory / 'reasoning').outputs())
    assert len(records) == 16
    assert len({r['event_id'] for r in records[:4]}) == 1
    assert [r['payload'].t_ms for r in records[:4]] == [10, 11, 12, 20]
    assert [r['reasoning'][0]['format'] for r in records[:3]] == ['delta', 'delta', 'snapshot']
    assert records[2]['reasoning'][0]['kind'] == 'summary'
    assert records[3]['payload'].payload['answer'] == 42
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
        def process(self, request):
            return {'answer': 42, 'thinking': 'Multiplication gives 42.'}
    subject = Model()
    procedure = CallableProtocol('external', lambda model, context: model.process('6 * 7?'))
    result = run(tmp_path, subject, procedure=procedure,
                 tool=RecordReasoning(extract=lambda output: output['thinking']))
    record = next(RunRecord(result.directory / 'reasoning').outputs())
    assert record['payload']['answer'] == 42
    assert record['reasoning'][0]['text'] == 'Multiplication gives 42.'
    assert 'process' not in vars(subject)  # The original method is restored.


@pytest.mark.parametrize('reasoning', [42, {'text': 42}, {'text': 'x', 'format': 'guess'}, {'text': 'x', 'kind': ''}])
def test_malformed_reasoning_fails_explicitly(tmp_path, reasoning):
    with pytest.raises((TypeError, ValueError), match='Reasoning'):
        run(tmp_path, trace_subject({'text': '42', 'reasoning': reasoning}))
    assert RunRecord(tmp_path / 'run/reasoning').manifest['status'] == 'failed'
    assert RunRecord(tmp_path / 'run/inputs_outputs').manifest['status'] == 'failed'


def test_requires_input_recorder_before_execution(tmp_path):
    experiment = Experiment(subject=trace_subject({'text': '42'}), protocol=protocol(),
                            tools=[RecordReasoning()], output_dir=tmp_path / 'run')
    with pytest.raises(ValueError, match='RecordInputsOutputs'):
        experiment.run()
    assert not experiment.output_dir.exists()


def test_trace_subject_streams_before_final_answer_and_parses_once_per_trial(tmp_path):
    calls = []
    def provider(request):
        yield {'text': '', 'reasoning': {'text': '6 * 7', 'format': 'delta'}, 'final': False}
        yield {'text': '42', 'final': True}
    subject = build_trace_subject(
        'streaming-provider', provider=provider,
        parse=lambda text: calls.append(text) or int(text), provenance={}, streaming=True,
    )
    result = run(tmp_path, subject)
    records = list(RunRecord(result.directory / 'reasoning').outputs())
    assert calls == ['42'] * 4
    assert len(records) == 8
    assert records[0]['payload'].payload['valid'] is None
    assert records[0]['payload'].payload['answer'] is None
    assert records[0]['payload'].meta['fragment_index'] == 0
    assert records[1]['payload'].payload['answer'] == 42
    assert records[1]['payload'].payload['final'] is True


def test_streaming_direct_call_keeps_every_fragment_and_buffer_snapshot(tmp_path):
    closed = []
    def provider(request):
        raw = {'text': '', 'reasoning': {'text': 'first', 'format': 'delta'}, 'final': False}
        try:
            yield raw
            raw['reasoning']['text'] = 'second'
            yield raw
            yield {'text': '42', 'final': True}
        finally:
            closed.append(True)
    subject = build_trace_subject('direct-stream', provider=provider, parse=int,
                                  provenance={}, streaming=True)
    procedure = CallableProtocol('direct', lambda model, context: model.process(
        StreamEvent('generation_request', {'prompt': '6 * 7?'}, 0)))
    result = run(tmp_path, subject, procedure=procedure)
    record = next(RunRecord(result.directory / 'reasoning').outputs())
    assert record['payload'].payload['answer'] == 42
    assert [b['text'] for b in record['reasoning']] == ['first', 'second']
    assert len(record['payload'].payload['stream']) == 3
    assert closed == [True]


@pytest.mark.parametrize('fragments,match', [
    ([], 'without a final'),
    ([{'text': '', 'reasoning': 'partial', 'final': False}], 'without a final'),
    ([{'text': '42'}], 'boolean final'),
    ([{'text': '42', 'final': True}, {'text': 'extra', 'final': False}], 'after the final'),
])
def test_invalid_provider_streams_fail_without_claiming_completion(tmp_path, fragments, match):
    closed = []
    def provider(request):
        try:
            yield from fragments
        finally:
            closed.append(True)
    subject = build_trace_subject('invalid-stream', provider=provider, parse=int,
                                  provenance={}, streaming=True)
    with pytest.raises((ValueError, TypeError), match=match):
        run(tmp_path, subject)
    assert closed == [True]
    assert RunRecord(tmp_path / 'run/reasoning').manifest['status'] == 'failed'


def test_stream_provider_closes_when_recorder_rejects_fragment(tmp_path):
    closed = []
    def provider(request):
        try:
            yield {'text': '', 'reasoning': 42, 'final': False}
            pytest.fail('A failing recorder must close the stream immediately')
        finally:
            closed.append(True)
    subject = build_trace_subject('bad-fragment', provider=provider, parse=int,
                                  provenance={}, streaming=True)
    with pytest.raises(TypeError, match='Reasoning'):
        run(tmp_path, subject)
    assert closed == [True]
