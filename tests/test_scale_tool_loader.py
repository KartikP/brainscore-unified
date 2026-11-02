"""Loaded scaling uses the existing backend and supports named composition."""
import numpy as np
import pytest
import torch

from brainscore.experiments import Experiment, ScaleActivity, TorchInstrumentation, compare_outputs
from brainscore.run_record import RunRecord
from brainscore.tools import attach_tools, list_tools, load_tool
from brainscore_core.events import Selection
from tests.test_ablation_tool_loader import FixtureSubject, protocol

pytestmark = pytest.mark.unit


def test_loaded_scaling_preserves_existing_class_defaults_and_filters(tmp_path):
    subject = FixtureSubject()
    provider = TorchInstrumentation({'population': subject.layer})
    target = Selection('population', [1])
    loaded = load_tool('scale-activity', targets=[target], factor=0.5,
                       conditions=['lesioned'], trials=[1])
    native = ScaleActivity([target], factor=0.5, conditions=['lesioned'], trials=[1])
    assert type(loaded) is ScaleActivity and loaded.name == 'scale_activity'
    assert loaded.describe() == native.describe()
    assert loaded is not load_tool('scale-activity', targets=[target], factor=0.5)
    assert 'scale-activity' in list_tools()
    for name, tool in [('native', native), ('loaded', loaded)]:
        Experiment(subject=subject, protocol=protocol(), tools=[load_tool('record-io'), tool],
                   instrumentation=provider, output_dir=tmp_path / name).run()
    assert compare_outputs(tmp_path / 'native', tmp_path / 'loaded')['equal']
    outputs = list(RunRecord(tmp_path / 'loaded').outputs())
    np.testing.assert_array_equal(outputs[-1].payload, [[1., 1., 3.]])
    assert not subject.layer._forward_hooks


@pytest.mark.parametrize('error', [None, RuntimeError, KeyboardInterrupt])
def test_named_scales_compose_with_recording_and_restore(tmp_path, error):
    subject = FixtureSubject()
    original = subject.layer.register_forward_hook(lambda *args: None)
    tools = [load_tool('record-io'),
             load_tool('record-activity', targets=['population'], name='before', when='before'),
             load_tool('scale-activity', targets=['population'], factor=0.5, name='halve'),
             load_tool('scale-activity', targets=['population'], factor=-2, name='invert_double'),
             load_tool('record-activity', targets=['population'], name='after')]

    def run():
        with attach_tools(subject, tools=tools,
                          instrumentation=TorchInstrumentation({'population': subject.layer}),
                          output_dir=tmp_path / 'run'):
            np.testing.assert_array_equal(subject.layer(torch.tensor([[2., 4.]])).numpy(), [[-2., -4.]])
            if error is not None:
                raise error('stopped')

    if error is None:
        run()
    else:
        with pytest.raises(error, match='stopped'):
            run()
    assert list(subject.layer._forward_hooks) == [original.id]
    np.testing.assert_array_equal(subject.layer(torch.ones(1, 2)).numpy(), [[1., 1.]])
    events = list(RunRecord(tmp_path / 'run').events())
    activities = [e['payload']['value']['array'] for e in events if e['kind'] == 'activity']
    np.testing.assert_array_equal(activities, [[[2., 4.]], [[-2., -4.]]])
    assert [e['source'] for e in events if e['kind'] == 'intervention_end'] == ['invert_double', 'halve']
    original.remove()


@pytest.mark.parametrize('factor', [True, '1', float('nan'), float('inf')])
def test_loaded_scale_rejects_invalid_factors(factor):
    with pytest.raises(ValueError, match='finite number'):
        load_tool('scale-activity', targets=['population'], factor=factor)


@pytest.mark.parametrize('name', ['', ' ', 7])
def test_loaded_scale_rejects_invalid_instance_names(name):
    with pytest.raises(ValueError, match='name'):
        load_tool('scale-activity', targets=['population'], factor=0.5, name=name)
