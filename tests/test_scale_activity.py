"""Scaling uses real hooks, preserves tuple outputs, and detaches on failure."""
from contextlib import contextmanager
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from brainscore.experiments import (
    Ablate, CallableProtocol, Experiment, RecordActivity, RecordInputsOutputs, ScaleActivity,
    SessionProtocol, TorchInstrumentation,
)
from brainscore_core.contract import Subject
from brainscore_core.events import Selection
from brainscore_core.streaming import InMemorySession, StreamEvent

pytestmark = pytest.mark.unit


@pytest.mark.parametrize('tool_class', [Ablate, ScaleActivity])
@pytest.mark.parametrize('invalid', ['instrumentation', 'conditions', 'trials'])
def test_intervention_validation_names_the_selected_tool(tool_class, invalid):
    layer = torch.nn.Identity()
    options = {'factor': 0.5} if tool_class is ScaleActivity else {}
    if invalid != 'instrumentation':
        options[invalid] = ['coggym-question-1']
    tool = tool_class(['layer'], **options)
    experiment = SimpleNamespace(
        instrumentation=(
            None if invalid == 'instrumentation'
            else TorchInstrumentation({'layer': layer})
        ),
        protocol=CallableProtocol('external-evaluation', lambda subject, context: None),
    )
    message = (
        f'{tool_class.__name__} requires an instrumentation provider'
        if invalid == 'instrumentation'
        else f'{tool_class.__name__} {invalid} must belong to the protocol'
    )
    with pytest.raises(ValueError, match=f'^{message}$'):
        tool.validate(experiment)
    assert not layer._forward_hooks


def test_selected_units_conditions_and_recording_order(tmp_path):
    class Model(Subject):
        identifier = 'scaled'
        in_channels, out_channels = {'stimulus'}, {'behavior'}

        def __init__(self):
            self.layer = torch.nn.Identity()

        def reset(self):
            pass

        def interact(self, session):
            event = session.next_input()
            session.emit(StreamEvent('behavior', self.layer(torch.tensor(event.payload)).numpy(), None))

    @contextmanager
    def factory(trial):
        yield InMemorySession([StreamEvent('stimulus', [2., 4.], None)])

    subject = Model()
    result = Experiment(
        subject=subject,
        protocol=SessionProtocol('scale', factory, conditions=['baseline', 'damped', 'restored']),
        tools=[
            RecordInputsOutputs(),
            RecordActivity(['layer'], when='before', name='before'),
            ScaleActivity([Selection('layer', [1])], factor=0.5, conditions=['damped']),
            RecordActivity(['layer']),
        ],
        instrumentation=TorchInstrumentation({'layer': subject.layer}),
        output_dir=tmp_path / 'run',
    ).run()
    events = list(result.record.events())
    outputs = [e['payload'].payload for e in events if e['kind'] == 'output']
    np.testing.assert_array_equal(outputs, [[2., 4.], [2., 2.], [2., 4.]])
    activities = [e['payload']['value']['array'] for e in events
                  if e['kind'] == 'activity' and e['condition'] == 'damped']
    np.testing.assert_array_equal(activities, [[2., 4.], [2., 2.]])
    assert not subject.layer._forward_hooks
    assert any(e['kind'] == 'intervention_start' and e['payload']['factor'] == 0.5 for e in events)


def test_every_fifth_block_and_failure_cleanup():
    model = torch.nn.Sequential(*[torch.nn.Identity() for _ in range(12)])
    targets = [str(i) for i in range(4, len(model), 5)]
    original = torch.tensor([4., 8.])
    provider = TorchInstrumentation(model)
    with pytest.raises(KeyboardInterrupt):
        with provider.scale(targets, factor=0.5):
            torch.testing.assert_close(model(original), original * 0.25)
            raise KeyboardInterrupt()
    assert all(not layer._forward_hooks for layer in model)
    torch.testing.assert_close(model(original), original)


def test_tuple_output_preserves_auxiliary_values_and_original_tensor():
    class Block(torch.nn.Module):
        def forward(self, value):
            return value, 'cache'
    block = Block()
    original = torch.tensor([4., 8.])
    with TorchInstrumentation({'block': block}).scale(['block'], factor=0.5):
        changed, cache = block(original)
        torch.testing.assert_close(changed, original * 0.5)
        assert cache == 'cache'
        torch.testing.assert_close(original, torch.tensor([4., 8.]))
    torch.testing.assert_close(block(original)[0], original)


@pytest.mark.parametrize('factor', [float('nan'), float('inf'), True, '0.5'])
def test_invalid_factor_rejected_before_hooks(factor):
    layer = torch.nn.Identity()
    with pytest.raises(ValueError, match='finite number'):
        ScaleActivity(['layer'], factor=factor)
    with pytest.raises(ValueError, match='finite number'):
        with TorchInstrumentation({'layer': layer}).scale(['layer'], factor=factor):
            pass
    assert not layer._forward_hooks


def test_all_targets_validated_before_any_attachment():
    layer = torch.nn.Identity()
    with pytest.raises(ValueError):
        with TorchInstrumentation({'layer': layer}).scale(['layer', 'missing'], factor=0.5):
            pass
    assert not layer._forward_hooks
