"""Loaded reasoning tools preserve exposed text and support separate named records."""
from contextlib import contextmanager

import pytest

from brainscore.experiments import Experiment, RecordReasoning, compare_outputs
from brainscore.run_record import RunRecord
from brainscore.tools import attach_tools, load_tool
from tests.test_reasoning_recorder import StreamingSubject, protocol, trace_subject

pytestmark = pytest.mark.unit


def test_loaded_reasoning_matches_existing_class_and_record(tmp_path):
    subject = trace_subject({'text': '42', 'reasoning': {'text': '6 times 7.', 'kind': 'summary'}})
    loaded = load_tool('record-reasoning')
    assert type(loaded) is RecordReasoning and loaded.name == 'reasoning'
    assert loaded.describe() == RecordReasoning().describe()
    for name, tool in [('native', RecordReasoning()), ('loaded', loaded)]:
        Experiment(subject=subject, protocol=protocol(), tools=[load_tool('record-io'), tool],
                   output_dir=tmp_path / name).run()
    assert compare_outputs(tmp_path / 'native', tmp_path / 'loaded')['equal']
    native = list(RunRecord(tmp_path / 'native' / 'reasoning').outputs())
    observed = list(RunRecord(tmp_path / 'loaded' / 'reasoning').outputs())
    assert [r['reasoning'] for r in observed] == [r['reasoning'] for r in native]


def test_two_named_extractors_keep_distinct_records_and_input_links(tmp_path):
    subject = trace_subject({'text': '42', 'reasoning': 'Exposed reasoning.', 'summary': 'Short summary.'})
    result = Experiment(subject=subject, protocol=protocol(), tools=[load_tool('record-io'),
        load_tool('record-reasoning', name='exposed'),
        load_tool('record-reasoning', name='summaries', extract=lambda output: output.payload['raw']['summary'])],
        output_dir=tmp_path / 'run').run()
    inputs = {e['event_id'] for e in result.record.events() if e['kind'] == 'input'}
    for name, text in [('exposed', 'Exposed reasoning.'), ('summaries', 'Short summary.')]:
        records = list(RunRecord(result.directory / name).outputs())
        assert len(records) == 4
        assert all(r['reasoning'][0]['text'] == text and r['event_id'] in inputs for r in records)
    assert {a['path'] for a in result.manifest['artifacts']} >= {
        'exposed/events.jsonl', 'summaries/events.jsonl'}


def test_streamed_formats_and_unavailable_reasoning_are_not_reinterpreted(tmp_path):
    result = Experiment(subject=StreamingSubject(), protocol=protocol(),
        tools=[load_tool('record-io'), load_tool('record-reasoning')], output_dir=tmp_path / 'run').run()
    records = list(RunRecord(result.directory / 'reasoning').outputs())
    assert [r['reasoning'][0]['format'] for r in records[:3]] == ['delta', 'delta', 'snapshot']
    assert records[3]['reasoning_available'] is False
    assert records[3]['reasoning'] == [] and records[3]['payload'].payload['answer'] == 42


def test_extractor_failure_fails_both_records_and_unsubscribes(tmp_path):
    from brainscore_core.streaming import StreamEvent

    def broken(output):
        raise RuntimeError('extractor failed')

    subject = trace_subject({'text': '42'})
    with pytest.raises(RuntimeError, match='extractor failed'):
        with attach_tools(subject, tools=[load_tool('record-io'),
                          load_tool('record-reasoning', extract=broken)],
                          output_dir=tmp_path / 'run') as context:
            context.publish('output', StreamEvent('response_trace', {'raw': {'text': '42'}}, None))
    assert context.listeners == []
    assert RunRecord(tmp_path / 'run').manifest['status'] == 'failed'
    assert RunRecord(tmp_path / 'run' / 'reasoning').manifest['status'] == 'failed'


def test_missing_shared_writer_fails_before_creating_output(tmp_path):
    with pytest.raises(ValueError, match='RecordInputsOutputs'):
        with attach_tools(trace_subject({'text': '42'}), tools=[load_tool('record-reasoning')],
                          output_dir=tmp_path / 'run'):
            pytest.fail('missing writer entered scope')
    assert not (tmp_path / 'run').exists()


def test_partial_observer_attachment_marks_earlier_records_failed(tmp_path):
    from brainscore.experiments import Tool

    class Broken(Tool):
        name = 'broken'

        @contextmanager
        def attach(self, context):
            raise RuntimeError('attachment failed')
            yield

    with pytest.raises(RuntimeError, match='attachment failed'):
        with attach_tools(trace_subject({'text': '42'}),
                          tools=[load_tool('record-io'), load_tool('record-reasoning'), Broken()],
                          output_dir=tmp_path / 'run'):
            pytest.fail('broken observer entered scope')
    assert RunRecord(tmp_path / 'run').manifest['status'] == 'failed'
    assert RunRecord(tmp_path / 'run' / 'reasoning').manifest['status'] == 'failed'


@pytest.mark.parametrize('name', ['', ' ', 7, '..', '../outside', '/outside', 'a\\b',
                                   'inputs_outputs', 'arrays', 'experiment.json', 'C:outside', 'a\0b'])
def test_named_reasoning_records_cannot_escape_or_collide(name):
    with pytest.raises(ValueError, match='name'):
        load_tool('record-reasoning', name=name)
