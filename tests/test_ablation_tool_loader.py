"""Loaded ablations reuse existing interventions, filtering and cleanup."""
from contextlib import contextmanager

import numpy as np
import pytest
import torch

from brainscore.experiments import (
    Ablate, Experiment, PatchActivity, RecordInputsOutputs, ScaleActivity,
    SessionProtocol, TorchInstrumentation,
)
from brainscore.run_record import RunRecord
from brainscore.tools import attach_tools, list_tools, load_tool
from brainscore_core.contract import Subject
from brainscore_core.events import Selection
from brainscore_core.streaming import Session, StreamEvent

pytestmark = pytest.mark.unit


class FixtureSubject(Subject):
    identifier = 'loaded-ablation-fixture'
    in_channels, out_channels = {'stimulus'}, {'behavior'}

    def __init__(self) -> None:
        self.layer = torch.nn.Identity()

    def participate(self, session: Session) -> None:
        while (event := session.next_input()) is not None:
            values = self.layer(torch.tensor(event.payload, dtype=torch.float32)).numpy()
            session.respond(StreamEvent('behavior', values, event.t_ms))


def protocol() -> SessionProtocol:
    @contextmanager
    def factory(trial):
        yield Session([StreamEvent('stimulus', [[1., 2., 3.]], None)])
    return SessionProtocol('loaded-ablation', factory, conditions=['baseline', 'lesioned'],
                           repetitions=[0, 1], input_channels=['stimulus'], output_channels=['behavior'])


def test_loader_returns_existing_implementation_and_fresh_instances():
    loaded = load_tool('ablate', targets=['population'])
    assert type(loaded) is Ablate
    assert loaded.name == 'ablation' and loaded.phase == 'intervene'
    assert loaded.describe() == Ablate(['population']).describe()
    assert loaded is not load_tool('ablate', targets=['population'])
    assert 'ablate' in list_tools()


def test_loaded_and_constructed_ablation_have_identical_filtered_results(tmp_path):
    subject = FixtureSubject()
    provider = TorchInstrumentation({'population': subject.layer})
    outputs = []
    for name, tool in [
        ('constructed', Ablate([Selection('population', [1])], conditions=['lesioned'], repetitions=[1])),
        ('loaded', load_tool('ablate', targets=[Selection('population', [1])],
                            conditions=['lesioned'], repetitions=[1])),
    ]:
        result = Experiment(subject=subject, protocol=protocol(),
                            tools=[RecordInputsOutputs(), tool], instrumentation=provider,
                            output_dir=tmp_path / name).run()
        outputs.append([(event['condition'], event['repetition_id'], event['payload'].payload)
                        for event in result.record.events() if event['kind'] == 'output'])
        assert not subject.layer._forward_hooks
    assert [(condition, trial) for condition, trial, _ in outputs[0]] == [
        (condition, trial) for condition, trial, _ in outputs[1]]
    for (_, _, expected), (_, _, actual) in zip(*outputs):
        np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal([row[2] for row in outputs[1]],
                                  [[[1., 2., 3.]], [[1., 2., 3.]],
                                   [[1., 2., 3.]], [[1., 0., 3.]]])


@pytest.mark.parametrize('error', [None, RuntimeError, KeyboardInterrupt])
def test_multiple_named_ablations_preserve_other_hooks_and_restore(tmp_path, error):
    subject = FixtureSubject()
    provider = TorchInstrumentation({'population': subject.layer})
    original = subject.layer.register_forward_hook(lambda *args: None)
    tools = [load_tool('ablate', targets=[Selection('population', [index])], name=name)
             for index, name in [(0, 'first'), (1, 'second')]]

    def run():
        with attach_tools(subject, tools=[RecordInputsOutputs(), *tools], instrumentation=provider,
                          output_dir=tmp_path / 'run'):
            np.testing.assert_array_equal(subject.layer(torch.ones(1, 3)).numpy(), [[0., 0., 1.]])
            if error is not None:
                raise error('scope failed')

    if error is None:
        run()
    else:
        with pytest.raises(error, match='scope failed'):
            run()
    assert list(subject.layer._forward_hooks) == [original.id]
    np.testing.assert_array_equal(subject.layer(torch.ones(1, 3)).numpy(), [[1., 1., 1.]])
    record = RunRecord(tmp_path / 'run')
    assert record.manifest['status'] == ('complete' if error is None else 'failed')
    starts = [event['source'] for event in record.events() if event['kind'] == 'intervention_start']
    ends = [event['source'] for event in record.events() if event['kind'] == 'intervention_end']
    assert starts == ['first', 'second'] and ends == ['second', 'first']
    original.remove()


def test_naming_ablation_does_not_change_other_intervention_defaults():
    assert ScaleActivity(['population'], factor=0.5).name == 'scale_activity'
    assert PatchActivity(['population'], reference={}).name == 'patch_activity'


@pytest.mark.parametrize('name', ['', ' ', 7])
def test_invalid_instance_name_fails_during_loading(name):
    with pytest.raises(ValueError, match='name'):
        load_tool('ablate', targets=['population'], name=name)


def test_invalid_collection_fails_before_any_attachment(tmp_path):
    subject = FixtureSubject()
    tools = [load_tool('ablate', targets=['population'], name='known'),
             load_tool('ablate', targets=['missing'], name='missing')]
    with pytest.raises(ValueError, match='missing'):
        with attach_tools(subject, tools=tools,
                          instrumentation=TorchInstrumentation({'population': subject.layer}),
                          output_dir=tmp_path / 'run'):
            pytest.fail('invalid target entered scope')
    assert not subject.layer._forward_hooks and not (tmp_path / 'run').exists()
