"""Loaded exact patches retain references, scope limits and explicit ordering."""
import numpy as np
import pytest
import torch

from brainscore.experiments import Experiment, PatchActivity, TorchInstrumentation, compare_outputs
from brainscore.run_record import RunRecord
from brainscore.tools import attach_tools, load_tool
from brainscore_core.events import Selection
from tests.test_ablation_tool_loader import FixtureSubject, protocol

pytestmark = pytest.mark.unit


def reference(subject):
    subject.layer.eval()
    provider = TorchInstrumentation({'population': subject.layer})
    analysis = provider.attribute(['population'],
        reference_call=lambda: subject.layer(torch.tensor([[5., 6., 7.]])),
        comparison_call=lambda: subject.layer(torch.tensor([[1., 2., 3.]])),
        objective=lambda output: output.sum())
    return provider, analysis['reference']


def test_loaded_patch_is_existing_class_and_preserves_filters(tmp_path):
    subject = FixtureSubject()
    provider, donor = reference(subject)
    target = Selection('population', [0])
    loaded = load_tool('patch-activity', targets=[target], reference=donor,
                       conditions=['lesioned'], repetitions=[1])
    native = PatchActivity([target], reference=donor, conditions=['lesioned'], repetitions=[1])
    assert type(loaded) is PatchActivity and loaded.name == 'patch_activity'
    assert loaded.describe() == native.describe()
    for name, tool in [('native', native), ('loaded', loaded)]:
        Experiment(subject=subject, protocol=protocol(), tools=[load_tool('record-io'), tool],
                   instrumentation=provider, output_dir=tmp_path / name).run()
    assert compare_outputs(tmp_path / 'native', tmp_path / 'loaded')['equal']
    outputs = list(RunRecord(tmp_path / 'loaded').outputs())
    np.testing.assert_array_equal(outputs[-1].payload, [[5., 2., 3.]])
    assert not subject.layer._forward_hooks


@pytest.mark.parametrize('patch_first,expected', [(True, 2.5), (False, 5.)])
def test_patch_and_scale_follow_declared_order_and_save_donor(tmp_path, patch_first, expected):
    subject = FixtureSubject()
    provider, donor = reference(subject)
    target = Selection('population', [0])
    patch = load_tool('patch-activity', targets=[target], reference=donor, name='donor')
    donor['values']['population'][0, 0] = 99  # Loaded patch owns its snapshot.
    scale = load_tool('scale-activity', targets=[target], factor=0.5, name='half')
    tools = [patch, scale] if patch_first else [scale, patch]
    with attach_tools(subject, tools=[load_tool('record-io'), *tools],
                      instrumentation=provider, output_dir=tmp_path / 'run'):
        np.testing.assert_array_equal(subject.layer(torch.tensor([[1., 2., 3.]])).numpy(),
                                      [[expected, 2., 3.]])
    np.testing.assert_array_equal(subject.layer(torch.ones(1, 3)).numpy(), [[1., 1., 1.]])
    events = list(RunRecord(tmp_path / 'run').events())
    saved = next(e for e in events if e['kind'] == 'patch_reference')
    assert saved['source'] == 'donor' and saved['payload']['values']['population'][0, 0] == 5
    assert [e['source'] for e in events if e['kind'] == 'intervention_end'] == [t.name for t in tools[::-1]]


def test_two_named_patches_can_target_disjoint_units(tmp_path):
    subject = FixtureSubject()
    provider, donor = reference(subject)
    tools = [load_tool('patch-activity', targets=[Selection('population', [i])],
                       reference=donor, name=f'unit_{i}') for i in [0, 2]]
    with attach_tools(subject, tools=[load_tool('record-io'), *tools],
                      instrumentation=provider, output_dir=tmp_path / 'run'):
        np.testing.assert_array_equal(subject.layer(torch.ones(1, 3)).numpy(), [[5., 1., 7.]])
    assert not subject.layer._forward_hooks


@pytest.mark.parametrize('failure', ['missing', 'repeated', 'interrupt'])
def test_loaded_patch_limits_and_cleanup_remain_explicit(tmp_path, failure):
    subject = FixtureSubject()
    provider, donor = reference(subject)
    original = subject.layer.register_forward_hook(lambda *args: None)
    error = KeyboardInterrupt if failure == 'interrupt' else ValueError
    with pytest.raises(error):
        with attach_tools(subject,
                          tools=[load_tool('record-io'), load_tool('patch-activity',
                              targets=['population'], reference=donor)],
                          instrumentation=provider, output_dir=tmp_path / 'run'):
            if failure != 'missing':
                subject.layer(torch.ones(1, 3))
            if failure == 'repeated':
                subject.layer(torch.ones(1, 3))
            if failure == 'interrupt':
                raise KeyboardInterrupt()
    assert list(subject.layer._forward_hooks) == [original.id]
    assert RunRecord(tmp_path / 'run').manifest['status'] == 'failed'
    original.remove()


def test_invalid_reference_rejected_before_hooks_or_files(tmp_path):
    subject = FixtureSubject()
    with pytest.raises(ValueError, match='reference'):
        with attach_tools(subject, tools=[load_tool('record-io'),
                          load_tool('patch-activity', targets=['population'], reference={})],
                          instrumentation=TorchInstrumentation({'population': subject.layer}),
                          output_dir=tmp_path / 'run'):
            pytest.fail('invalid reference entered scope')
    assert not subject.layer._forward_hooks and not (tmp_path / 'run').exists()
