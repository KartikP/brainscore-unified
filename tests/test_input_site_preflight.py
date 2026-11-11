"""TorchInstrumentation hooks module outputs only, so input-site names are refused up front.

An explicit alias such as {'projection:input': module} would otherwise record or
change the module's OUTPUT under an input label. Every operation, loaded tool and
direct scope rejects it before any hook is installed or any file is written.
"""
from contextlib import contextmanager

import numpy as np
import pytest
import torch

from brainscore.experiments import (
    Experiment, RecordInputsOutputs, SessionProtocol, TorchInstrumentation,
)
from brainscore.tools import attach_tools, load_tool
from brainscore_core.contract import Subject
from brainscore_core.events import Selection
from brainscore_core.streaming import Session, StreamEvent

pytestmark = pytest.mark.unit

ADDRESSES = ['projection:input', 'projection:other', 'projection:input:input']


class FixtureSubject(Subject):
    identifier = 'input-site-preflight-fixture'
    in_channels, out_channels = {'observation'}, {'behavior'}

    def __init__(self) -> None:
        self.layer = torch.nn.Linear(3, 2)

    def participate(self, session: Session) -> None:
        while (event := session.next_input()) is not None:
            with torch.no_grad():
                values = self.layer(torch.tensor(event.payload, dtype=torch.float32)).numpy()
            session.respond(StreamEvent('behavior', values, event.t_ms))


def protocol() -> SessionProtocol:
    @contextmanager
    def factory(trial):
        yield Session([StreamEvent('observation', [[1., 2., 3.]], None)])
    return SessionProtocol('input-site-preflight', factory, conditions=['baseline'],
                           input_channels=['observation'], output_channels=['behavior'])


def _no_hooks(module):
    return not module._forward_hooks and not module._forward_pre_hooks


def _loaded(identifier, target):
    if identifier == 'record-activity':
        return load_tool(identifier, targets=[target])
    if identifier == 'ablate':
        return load_tool(identifier, targets=[target])
    return load_tool(identifier, targets=[target], factor=0.5)


@pytest.mark.parametrize('address', ADDRESSES)
@pytest.mark.parametrize('operation', ['record', 'ablate', 'scale', 'transform', 'patch', 'attribute'])
def test_every_operation_rejects_site_aliases_before_hooking(address, operation):
    module = torch.nn.Linear(3, 2)
    provider = TorchInstrumentation({address: module})
    with pytest.raises(ValueError, match='outputs only'):
        if operation == 'record':
            provider.record([address], lambda *args: None).__enter__()
        elif operation == 'ablate':
            provider.ablate([address]).__enter__()
        elif operation == 'scale':
            provider.scale([address], factor=0.5).__enter__()
        elif operation == 'transform':
            provider.transform([address], apply=lambda value: value).__enter__()
        elif operation == 'patch':
            provider.patch([address], reference={}).__enter__()
        else:
            provider.attribute([address], reference_call=lambda: module(torch.ones(1, 3)),
                               comparison_call=lambda: module(torch.ones(1, 3)),
                               objective=lambda output: output.sum())
    assert _no_hooks(module)


@pytest.mark.parametrize('identifier', ['record-activity', 'ablate', 'scale-activity'])
def test_loaded_tools_reject_an_input_alias_with_no_files_or_hooks(tmp_path, identifier):
    subject = FixtureSubject()
    provider = TorchInstrumentation({'projection:input': subject.layer})
    tool = _loaded(identifier, Selection('projection:input', [0]))
    with pytest.raises(ValueError, match='outputs only'):
        Experiment(subject=subject, protocol=protocol(), tools=[RecordInputsOutputs(), tool], instrumentation=provider,
                   output_dir=tmp_path / 'run').run()
    assert not (tmp_path / 'run').exists()
    assert _no_hooks(subject.layer)


def test_direct_scope_rejects_an_input_alias_with_no_files_or_hooks(tmp_path):
    subject = FixtureSubject()
    provider = TorchInstrumentation({'projection:input': subject.layer})
    with pytest.raises(ValueError, match='outputs only'):
        with attach_tools(subject, tools=[RecordInputsOutputs(),
                                         load_tool('record-activity', targets=['projection:input'])],
                          output_dir=tmp_path / 'scope', instrumentation=provider):
            pass
    assert not (tmp_path / 'scope').exists()
    assert _no_hooks(subject.layer)


def test_plain_output_aliases_still_record_the_output(tmp_path):
    subject = FixtureSubject()
    provider = TorchInstrumentation({'projection': subject.layer})
    captures = []
    value = torch.ones(1, 3)
    with provider.record(['projection'], lambda name, snapshot: captures.append(snapshot['array'])):
        with torch.no_grad():
            output = subject.layer(value)
    np.testing.assert_array_equal(captures[0], output.numpy())
    assert _no_hooks(subject.layer)
    # A model's own module paths never contain a colon and keep working.
    named = TorchInstrumentation(torch.nn.Sequential(torch.nn.Identity()))
    with named.ablate(['0']):
        assert torch.equal(named.targets['0'](value), torch.zeros(1, 3))
