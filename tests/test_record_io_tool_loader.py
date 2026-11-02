"""The loaded event recorder preserves the existing format and failure semantics."""
from contextlib import contextmanager

import numpy as np
import pytest
import torch

from brainscore.experiments import (
    Experiment, RecordInputsOutputs, SessionProtocol, TorchInstrumentation,
    compare_outputs, replay_sessions,
)
from brainscore.run_record import RecordingBudgetExceeded, RunRecord
from brainscore.tools import attach_tools, list_tools, load_tool
from brainscore_core.contract import Subject
from brainscore_core.streaming import BufferedSession, Session, StreamEvent

pytestmark = pytest.mark.unit


class FixtureSubject(Subject):
    identifier = 'loaded-record-io-fixture'
    in_channels, out_channels = {'stimulus'}, {'behavior'}

    def __init__(self) -> None:
        self.layer = torch.nn.Identity()

    def participate(self, session: Session) -> None:
        while (event := session.next_input()) is not None:
            values = self.layer(torch.tensor(event.payload, dtype=torch.float32)).numpy()
            session.respond(StreamEvent('behavior', values, event.t_ms, {'fixture': True}))


def protocol() -> SessionProtocol:
    @contextmanager
    def factory(trial):
        yield BufferedSession([StreamEvent('stimulus', [[1., 2.]], None)])
    return SessionProtocol('record-io-check', factory, input_channels=['stimulus'],
                           output_channels=['behavior'], trials=[0, 1])


def test_loader_returns_existing_recorder_and_preserves_default_configuration():
    loaded = load_tool('record-io')
    assert type(loaded) is RecordInputsOutputs
    assert loaded.name == 'inputs_outputs' and loaded.phase == 'observe'
    assert loaded.describe() == RecordInputsOutputs().describe()
    assert loaded is not load_tool('record-io')
    assert load_tool('record-io', max_bytes=1000).describe() == {'max_bytes': 1000}
    assert 'record-io' in list_tools()


def test_loaded_recorder_preserves_outputs_and_supports_existing_replay(tmp_path):
    subject = FixtureSubject()
    for name, recorder in [('constructed', RecordInputsOutputs()), ('loaded', load_tool('record-io'))]:
        Experiment(subject=subject, protocol=protocol(), tools=[recorder],
                   output_dir=tmp_path / name).run()
    assert compare_outputs(tmp_path / 'constructed', tmp_path / 'loaded')['equal']
    record = RunRecord(tmp_path / 'loaded')
    events = list(record.events())
    assert len([event for event in events if event['kind'] == 'input']) == 2
    assert len(list(record.outputs())) == 2
    replay = Experiment(subject=subject, protocol=replay_sessions(tmp_path / 'loaded'),
                        tools=[load_tool('record-io')], output_dir=tmp_path / 'replayed').run()
    assert compare_outputs(tmp_path / 'loaded', replay.directory)['equal']


def test_loaded_recorder_and_ablation_share_lifecycle_and_provenance(tmp_path):
    subject = FixtureSubject()
    with attach_tools(subject,
                      tools=[load_tool('record-io'), load_tool('ablate', targets=['population'])],
                      instrumentation=TorchInstrumentation({'population': subject.layer}),
                      metadata={'checkpoint': 'fixture-v1'}, output_dir=tmp_path / 'scope') as context:
        values = subject.layer(torch.ones(1, 2)).numpy()
        context.publish('measurement', values)
    record = RunRecord(tmp_path / 'scope')
    assert record.metadata['plan']['metadata']['checkpoint'] == 'fixture-v1'
    assert any(event['kind'] == 'intervention_start' for event in record.events())
    measurements = [event['payload'] for event in record.events() if event['kind'] == 'measurement']
    np.testing.assert_array_equal(measurements, [[[0., 0.]]])
    assert not subject.layer._forward_hooks


def test_loaded_recording_budget_fails_closed(tmp_path):
    subject = FixtureSubject()
    with pytest.raises(RecordingBudgetExceeded):
        with attach_tools(subject, tools=[load_tool('record-io', max_bytes=1)],
                          output_dir=tmp_path / 'scope'):
            pytest.fail('an exhausted recorder was attached')
    record = RunRecord(tmp_path / 'scope')
    assert record.manifest['status'] == 'failed'
    assert record.experiment_manifest['status'] == 'failed'


@pytest.mark.parametrize('limit', [0, -1, True, '100'])
def test_loader_retains_recorder_budget_validation(limit):
    with pytest.raises(ValueError, match='max_bytes'):
        load_tool('record-io', max_bytes=limit)
