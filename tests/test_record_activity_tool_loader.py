"""Loaded activity recorders compose with loaded writers and interventions."""
from contextlib import contextmanager
import subprocess
import sys

import numpy as np
import pytest
import torch

from brainscore.experiments import (
    Experiment, RecordActivity, SessionProtocol, TorchInstrumentation, compare_outputs,
)
from brainscore.run_record import RunRecord
from brainscore.tools import attach_tools, list_tools, load_tool
from brainscore_core.contract import Subject
from brainscore_core.events import Selection
from brainscore_core.streaming import BufferedSession, Session, StreamEvent

pytestmark = pytest.mark.unit


class FixtureSubject(Subject):
    identifier = 'loaded-record-activity-fixture'
    in_channels, out_channels = {'stimulus'}, {'behavior'}

    def __init__(self) -> None:
        self.layer = torch.nn.Identity()

    def interact(self, session: Session) -> None:
        while (event := session.next_input()) is not None:
            values = self.layer(torch.tensor(event.payload, dtype=torch.float32)).numpy()
            session.respond(StreamEvent('behavior', values, event.t_ms))


def protocol() -> SessionProtocol:
    @contextmanager
    def factory(trial):
        yield BufferedSession([StreamEvent('stimulus', [[1., 2., 3.]], None)])
    return SessionProtocol('record-activity-check', factory, conditions=['baseline', 'lesioned'],
                           input_channels=['stimulus'], output_channels=['behavior'])


def test_loader_returns_existing_recorder_with_names_and_positions():
    loaded = load_tool('record-activity', targets=['population'], name='before', when='before')
    assert type(loaded) is RecordActivity
    assert loaded.name == 'before' and loaded.phase == 'before_intervention'
    assert loaded.describe() == RecordActivity(['population'], name='before', when='before').describe()
    assert loaded is not load_tool('record-activity', targets=['population'])
    assert 'record-activity' in list_tools()


def test_two_loaded_recorders_match_existing_tools_around_loaded_ablation(tmp_path):
    subject = FixtureSubject()
    provider = TorchInstrumentation({'population': subject.layer})
    target = Selection('population', [1])
    for name, before, after in [
        ('constructed', RecordActivity([target], name='before', when='before'),
         RecordActivity([target], name='after')),
        ('loaded', load_tool('record-activity', targets=[target], name='before', when='before'),
         load_tool('record-activity', targets=[target], name='after')),
    ]:
        Experiment(subject=subject, protocol=protocol(),
                   tools=[load_tool('record-io'), before,
                          load_tool('ablate', targets=[target], conditions=['lesioned']), after],
                   instrumentation=provider, output_dir=tmp_path / name).run()
        assert not subject.layer._forward_hooks
    assert compare_outputs(tmp_path / 'constructed', tmp_path / 'loaded')['equal']
    captured = []
    for name in ['constructed', 'loaded']:
        events = [event for event in RunRecord(tmp_path / name).events() if event['kind'] == 'activity']
        assert [(e['source'], e['condition']) for e in events] == [
            ('before', 'baseline'), ('after', 'baseline'), ('before', 'lesioned'), ('after', 'lesioned')]
        captured.append([event['payload']['value']['array'] for event in events])
    np.testing.assert_array_equal(captured[0], captured[1])
    np.testing.assert_array_equal(captured[1], [[[2.]], [[2.]], [[2.]], [[0.]]])
    np.testing.assert_array_equal(subject.layer(torch.ones(1, 3)).numpy(), [[1., 1., 1.]])


@pytest.mark.parametrize('axis,shape', [(-1, (1, 2, 3)), (1, (1, 3, 2)), (None, (1, 2, 3))])
def test_loaded_recorder_retains_selection_axes_and_original_values(tmp_path, axis, shape):
    subject = FixtureSubject()
    values = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
    target = Selection('population', [1], axis=axis,
                       expected_shape=shape[1:] if axis is None else None)
    with attach_tools(subject,
                      tools=[load_tool('record-io'), load_tool('record-activity', targets=[target])],
                      instrumentation=TorchInstrumentation({'population': subject.layer}),
                      output_dir=tmp_path / 'scope'):
        output = subject.layer(torch.from_numpy(values))
        np.testing.assert_array_equal(output.numpy(), values)
    event = next(event for event in RunRecord(tmp_path / 'scope').events() if event['kind'] == 'activity')
    observed = event['payload']['value']
    source = values.reshape(1, -1) if axis is None else values
    np.testing.assert_array_equal(observed['array'], np.take(source, [1], axis=-1 if axis is None else axis))
    assert observed['axis'] == axis and observed['indices'] == [1]
    assert observed['expected_shape'] == target.expected_shape


def test_loaded_recording_cleans_up_on_interrupt_and_preserves_other_hooks(tmp_path):
    subject = FixtureSubject()
    original = subject.layer.register_forward_hook(lambda *args: None)
    with pytest.raises(KeyboardInterrupt):
        with attach_tools(subject,
                          tools=[load_tool('record-io'), load_tool('record-activity', targets=['population'])],
                          instrumentation=TorchInstrumentation({'population': subject.layer}),
                          output_dir=tmp_path / 'scope'):
            subject.layer(torch.ones(1, 3))
            raise KeyboardInterrupt()
    assert list(subject.layer._forward_hooks) == [original.id]
    record = RunRecord(tmp_path / 'scope')
    assert record.manifest['status'] == 'failed'
    assert len([e for e in record.events() if e['kind'] == 'activity']) == 1
    original.remove()


@pytest.mark.parametrize('invalid', ['sink', 'provider', 'target', 'duplicates'])
def test_loaded_recording_validates_dependencies_before_attachment(tmp_path, invalid):
    subject = FixtureSubject()
    tools = [load_tool('record-io'), load_tool('record-activity', targets=['population'])]
    provider = TorchInstrumentation({'population': subject.layer})
    if invalid == 'sink':
        tools.pop(0)
    elif invalid == 'provider':
        provider = None
    elif invalid == 'target':
        provider = TorchInstrumentation({'other': subject.layer})
    else:
        tools.append(load_tool('record-activity', targets=['population']))
    with pytest.raises(ValueError):
        with attach_tools(subject, tools=tools, instrumentation=provider, output_dir=tmp_path / 'scope'):
            pytest.fail('invalid recording scope entered')
    assert not subject.layer._forward_hooks and not (tmp_path / 'scope').exists()


@pytest.mark.parametrize('name', ['', ' ', 7])
def test_loaded_recording_requires_a_valid_instance_name(name):
    with pytest.raises(ValueError, match='name'):
        load_tool('record-activity', targets=['population'], name=name)


def test_loading_all_tools_does_not_import_model_or_fitting_dependencies():
    result = subprocess.run([sys.executable, '-c',
        'import sys; from brainscore.tools import load_tool; '
        'load_tool("ablate", targets=["population"]); '
        'load_tool("record-io"); '
        'load_tool("record-activity", targets=["population"]); '
        'load_tool("linear-probe", target="population"); '
        'assert not ({"torch", "transformers", "sklearn"} & set(sys.modules))'],
        capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
