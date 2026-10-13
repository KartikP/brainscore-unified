"""Exact linear effects, nonlinear approximation limits, and real hook cleanup."""
from contextlib import contextmanager
import copy
import json

import numpy as np
import pytest
import torch

from brainscore.experiments import (
    Experiment, PatchActivity, RecordActivity, RecordInputsOutputs,
    SessionProtocol, TorchInstrumentation,
)
from brainscore_core.contract import Subject
from brainscore_core.events import Selection
from brainscore_core.streaming import InMemorySession, StreamEvent

pytestmark = pytest.mark.unit


class LinearModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.units = torch.nn.Identity()
        self.readout = torch.nn.Linear(3, 1, bias=False)
        with torch.no_grad():
            self.readout.weight.copy_(torch.tensor([[2., -1., 0.]]))
        self.eval()

    def forward(self, values):
        return self.readout(self.units(values)).sum()


def estimate(model=None, *, targets=('units',), reference=None, comparison=None, **kwargs):
    model = model if model is not None else LinearModel()
    reference = torch.tensor([3., 1., 9.]) if reference is None else reference
    comparison = torch.tensor([1., 2., 4.]) if comparison is None else comparison
    return TorchInstrumentation(model).attribute(
        targets,
        reference_call=lambda: model(reference),
        comparison_call=lambda: model(comparison),
        objective=lambda output: output,
        **kwargs,
    )


def assert_clean(model):
    assert all(not module._forward_hooks for module in model.modules())


def test_estimates_match_exact_single_unit_patches_and_restore():
    model = LinearModel()
    provider = TorchInstrumentation(model)
    result = estimate(model)
    assert result['reference_value'] == 5 and result['comparison_value'] == 0
    np.testing.assert_allclose(result['attributions']['units'], [.8, .2, 0])
    for unit, expected in enumerate([4., 1., 0.]):
        with provider.patch([Selection('units', [unit])], reference=result['reference']):
            assert model(torch.tensor([1., 2., 4.])).item() == expected
        assert model(torch.tensor([1., 2., 4.])).item() == 0
    assert_clean(model)


def test_nonlinear_estimate_is_not_reported_as_an_exact_effect():
    class Squared(LinearModel):
        def forward(self, values):
            return self.units(values).square().sum()
    model = Squared()
    result = estimate(model, reference=torch.tensor([2., 0., 0.]),
                      comparison=torch.tensor([1., 0., 0.]), normalize=False)
    assert result['attributions']['units'][0] == 2
    with TorchInstrumentation(model).patch(['units'], reference=result['reference']):
        actual = model(torch.tensor([1., 0., 0.])).item() - result['comparison_value']
    assert actual == 3  # The linear approximation differs from the finite patch.


def test_frozen_parameters_and_existing_gradients_are_unchanged():
    model = LinearModel()
    model.readout.weight.grad = torch.ones_like(model.readout.weight) * 7
    before = model.readout.weight.detach().clone()
    model.requires_grad_(False)
    estimate(model)
    torch.testing.assert_close(model.readout.weight, before)
    torch.testing.assert_close(model.readout.weight.grad, torch.ones_like(before) * 7)
    assert not model.readout.weight.requires_grad
    assert_clean(model)


def test_location_uses_only_requested_batch_and_token():
    model = LinearModel()
    reference = torch.ones(2, 3, 3) * 3
    comparison = torch.ones(2, 3, 3)
    result = estimate(model, reference=reference, comparison=comparison,
                      location=(0, -1), normalize=False)
    np.testing.assert_equal(result['attributions']['units'], [4., -2., 0.])
    with TorchInstrumentation(model).patch([Selection('units', [0])], reference=result['reference']):
        assert model(comparison).item() == 10  # Original total 6, one coordinate adds 4.
    torch.testing.assert_close(comparison, torch.ones(2, 3, 3))


def test_subset_reference_preserves_original_unit_numbers_and_tuple_tail():
    class TupleUnits(torch.nn.Module):
        def forward(self, values):
            return values, 'untouched'
    class TupleModel(LinearModel):
        def __init__(self):
            super().__init__()
            self.units = TupleUnits()
            self.eval()
        def forward(self, values):
            activity, extra = self.units(values)
            assert extra == 'untouched'
            return self.readout(activity).sum()
    model = TupleModel()
    result = estimate(model, targets=[Selection('units', [2, 0])])
    with TorchInstrumentation(model).patch([Selection('units', [0])], reference=result['reference']):
        assert model(torch.tensor([1., 2., 4.])).item() == 4
    with pytest.raises(ValueError, match='not captured'):
        TorchInstrumentation(model).validate_reference([Selection('units', [1])], result['reference'])
    assert_clean(model)


@pytest.mark.parametrize('failure', ['reference', 'comparison', 'objective'])
def test_failure_cleans_hooks_and_preserves_preexisting_hooks(failure):
    model = LinearModel()
    existing = model.units.register_forward_hook(lambda *args: None)
    def call(stage):
        result = model(torch.ones(3))
        if stage == failure:
            raise KeyboardInterrupt(stage)
        return result
    def objective(output):
        if failure == 'objective':
            raise KeyboardInterrupt('objective')
        return output
    with pytest.raises(KeyboardInterrupt):
        TorchInstrumentation(model).attribute(
            ['units'], reference_call=lambda: call('reference'),
            comparison_call=lambda: call('comparison'), objective=objective,
        )
    assert len(model.units._forward_hooks) == 1
    existing.remove()
    assert_clean(model)


@pytest.mark.parametrize('failure', ['interrupt', 'missing', 'repeat', 'shape', 'dtype'])
def test_patch_failure_cleans_hooks(failure):
    model = LinearModel()
    ref = estimate(model)['reference']
    error = KeyboardInterrupt if failure == 'interrupt' else ValueError
    with pytest.raises(error):
        with TorchInstrumentation(model).patch(['units'], reference=ref):
            if failure == 'interrupt':
                raise KeyboardInterrupt()
            if failure == 'missing':
                pass
            elif failure == 'repeat':
                model(torch.ones(3)); model(torch.ones(3))
            elif failure == 'shape':
                model(torch.ones(1, 3))
            elif failure == 'dtype':
                model(torch.ones(3, dtype=torch.float64))
    assert_clean(model)


@pytest.mark.parametrize('case', ['zero', 'detached', 'nonscalar', 'nonfinite', 'shape', 'repeat', 'missing'])
def test_invalid_paired_analysis_fails_without_leaked_hooks(case):
    model = LinearModel()
    provider = TorchInstrumentation(model)
    def call(reference):
        if case == 'missing':
            return model.readout(torch.ones(3)).sum()
        value = torch.tensor([3., 1., 9.]) if reference else torch.ones(3)
        if case == 'zero':
            value = torch.ones(3)
        if case == 'shape' and reference:
            value = value[None, :]
        if case == 'repeat':
            model(value)
        return model(value)
    def objective(output):
        if case == 'detached': return output.detach()
        if case == 'nonscalar': return output[None]
        if case == 'nonfinite': return output * float('nan')
        return output
    with pytest.raises(ValueError):
        provider.attribute(['units'], reference_call=lambda: call(True),
                           comparison_call=lambda: call(False), objective=objective)
    assert_clean(model)


def test_inference_mode_and_training_mode_rejected_before_forward():
    model = LinearModel()
    with torch.inference_mode(), pytest.raises(ValueError, match='inference_mode'):
        estimate(model)
    model.train()
    with pytest.raises(ValueError, match='eval'):
        estimate(model)
    assert_clean(model)


def test_disconnected_units_are_explicit_zero_attributions():
    class Disconnected(LinearModel):
        def forward(self, values):
            self.units(values)
            return self.readout(values).sum()
    result = estimate(Disconnected())
    assert result['disconnected_targets'] == ['units']
    np.testing.assert_equal(result['attributions']['units'], [0, 0, 0])


@pytest.mark.parametrize('change', ['values', 'indices', 'location', 'version'])
def test_malformed_reference_rejected_before_attachment(change):
    model = LinearModel()
    ref = estimate(model)['reference']
    if change == 'values': ref['values']['units'][0] = float('nan')
    elif change == 'indices': ref['indices']['units'] = [0, 0, 0]
    elif change == 'location': ref['location'] = [8]
    else: ref['schema_version'] = 2
    with pytest.raises(ValueError):
        with TorchInstrumentation(model).patch(['units'], reference=ref):
            pass
    assert_clean(model)


def test_patch_tool_records_reference_conditions_and_restoration(tmp_path):
    class ModelSubject(Subject):
        identifier = 'patch-fixture'
        in_channels, out_channels = {'stimulus'}, {'behavior'}
        def __init__(self): self.model = LinearModel()
        def reset(self): pass
        def interact(self, session):
            event = session.next_input()
            value = self.model(torch.tensor(event.payload)).item()
            session.emit(StreamEvent('behavior', value, None))
    @contextmanager
    def session(trial):
        yield InMemorySession([StreamEvent('stimulus', [1., 2., 4.], None)])
    subject = ModelSubject()
    reference = estimate(subject.model)['reference']
    tool = PatchActivity([Selection('units', [0])], reference=reference, conditions=['patched'])
    reference['values']['units'][0] = 99  # The tool owns an independent snapshot.
    result = Experiment(
        subject=subject,
        protocol=SessionProtocol('patch', session, conditions=['baseline', 'patched', 'restored']),
        tools=[RecordInputsOutputs(), tool, RecordActivity(['units'])],
        instrumentation=TorchInstrumentation(subject.model), output_dir=tmp_path / 'run',
    ).run()
    events = list(result.record.events())
    assert [e['payload'].payload for e in events if e['kind'] == 'output'] == [0, 4, 0]
    saved = next(e['payload'] for e in events if e['kind'] == 'patch_reference')
    assert saved['values']['units'][0] == 3
    assert result.manifest['status'] == 'complete'
    assert_clean(subject.model)


def test_multiple_layers_and_outer_no_grad_keep_the_gradient_path():
    model = LinearModel()
    with torch.no_grad():
        result = estimate(model, targets=['units', 'readout'])
    np.testing.assert_allclose(result['attributions']['units'], [.8, .2, 0])
    np.testing.assert_allclose(result['attributions']['readout'], [1.])
    # Layer totals overlap causally; they are not additive contributions.
    assert_clean(model)


def test_in_place_downstream_operation_does_not_rewrite_captured_activity():
    model = torch.nn.Sequential(
        torch.nn.Identity(), torch.nn.ReLU(inplace=True), torch.nn.Linear(2, 1, bias=False),
    ).eval()
    with torch.no_grad():
        model[2].weight.fill_(1)
    result = TorchInstrumentation(model).attribute(
        ['0'], reference_call=lambda: model(torch.tensor([-2., 4.])),
        comparison_call=lambda: model(torch.tensor([-1., 2.])),
        objective=lambda result: result.sum(), normalize=False,
    )
    np.testing.assert_equal(result['reference']['values']['0'], [-2., 4.])
    np.testing.assert_equal(result['attributions']['0'], [0., 2.])
    assert_clean(model)


def test_no_grad_callback_fails_with_clear_message():
    model = LinearModel()
    @torch.no_grad()
    def comparison():
        return model(torch.ones(3))
    with pytest.raises(ValueError, match='gradients enabled'):
        TorchInstrumentation(model).attribute(
            ['units'], reference_call=lambda: model(torch.ones(3) * 2),
            comparison_call=comparison, objective=lambda value: value,
        )
    assert_clean(model)


def test_reference_survives_record_serialization(tmp_path):
    from brainscore.run_record import PayloadCodec
    from brainscore.experiments.attribution import reference_digest
    model = LinearModel()
    result = estimate(model)
    codec = PayloadCodec(tmp_path)
    loaded = codec.decode(json.loads(json.dumps(codec.encode(result))))
    assert reference_digest(loaded['reference']) == reference_digest(result['reference'])
    with TorchInstrumentation(model).patch(['units'], reference=loaded['reference']):
        assert model(torch.ones(3)).item() == 5


@pytest.mark.parametrize('indices', [[0, 0], [], [True], [9]])
def test_invalid_unit_selection_has_no_hook_leak(indices):
    model = LinearModel()
    with pytest.raises((ValueError, IndexError)):
        estimate(model, targets=[Selection('units', indices)])
    assert_clean(model)


def test_patch_validates_all_targets_before_hook_attachment():
    model = LinearModel()
    with pytest.raises(ValueError, match='Unknown'):
        with TorchInstrumentation(model).patch(['units', 'unknown'], reference=estimate(model)['reference']):
            pass
    assert_clean(model)


@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16])
def test_low_precision_reference_can_be_patched_without_dtype_changes(dtype):
    model = LinearModel().to(dtype=dtype)
    result = estimate(model, reference=torch.tensor([3., 1., 9.], dtype=dtype),
                      comparison=torch.tensor([1., 2., 4.], dtype=dtype))
    assert result['reference']['values']['units'].dtype == np.float32
    with TorchInstrumentation(model).patch(['units'], reference=result['reference']):
        value = model(torch.ones(3, dtype=dtype))
    assert value.dtype == dtype and value.item() == 5
    assert_clean(model)
