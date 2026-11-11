"""LEACE is checked against its reference implementation, offline and in a subject."""
from contextlib import contextmanager
import json

import numpy as np
import pytest
import torch
from concept_erasure import LeaceEraser

from brainscore.experiments import Experiment, SessionProtocol, TorchInstrumentation
from brainscore.run_record import RunRecord
from brainscore.tools import attach_tools, load_tool
from brainscore_core.events import Selection
from brainscore_core.streaming import Session, StreamEvent
from brainscore_core.supported_data_standards.brainio.assemblies import NeuroidAssembly
from tests.test_ablation_tool_loader import FixtureSubject

pytestmark = pytest.mark.unit


def activity(values, units=None):
    values = np.asarray(values)
    units = list(range(values.shape[-1])) if units is None else list(units)
    return NeuroidAssembly(values, dims=('presentation', 'neuroid'), coords={
        'stimulus_id': ('presentation', [f'item-{i}' for i in range(len(values))]),
        'stimulus_number': ('presentation', list(range(len(values)))),
        'unit_index': ('neuroid', units), 'layer': ('neuroid', ['population'] * len(units)),
    })


def discovery(classes=2):
    rng = np.random.RandomState(31)
    indices = rng.randint(classes, size=120)
    values = rng.normal(size=(120, 4))
    values[:, :classes] += np.eye(classes)[indices] * 3
    labels = np.asarray([f'class-{i}' for i in indices])
    return activity(values), labels


def fitted(**configuration):
    features, labels = discovery()
    return load_tool('leace', target='population', **configuration).fit(
        features, labels, provenance={'checkpoint': 'fixture-v1', 'split': 'discovery'})


@pytest.mark.parametrize('classes', [2, 3])
@pytest.mark.parametrize('shrinkage', [True, False])
def test_held_out_transform_matches_reference_without_refitting(classes, shrinkage):
    features, labels = discovery(classes)
    _, encoded = np.unique(labels, return_inverse=True)
    concepts = torch.from_numpy(np.eye(classes)[encoded])
    reference = LeaceEraser.fit(torch.from_numpy(features.values), concepts, shrinkage=shrinkage)
    learner = load_tool('leace', target='population', shrinkage=shrinkage)
    eraser = learner.fit(features, labels)
    held_out = activity(np.random.RandomState(8).normal(size=(17, 4)))
    transformed = eraser.transform(held_out)
    assert not learner.fitted and eraser.fitted
    assert isinstance(transformed, NeuroidAssembly)
    np.testing.assert_allclose(transformed.values, reference(torch.from_numpy(held_out.values)).numpy(),
                               rtol=1e-12, atol=1e-12)
    np.testing.assert_array_equal(transformed['stimulus_id'].values, held_out['stimulus_id'].values)
    np.testing.assert_array_equal(held_out.values, np.random.RandomState(8).normal(size=(17, 4)))


def test_strict_sample_erasure_removes_fitting_cross_covariance():
    features, labels = discovery()
    eraser = load_tool('leace', target='population', shrinkage=False,
                       constrain_cov_trace=False, svd_tol=1e-10).fit(features, labels)
    erased = eraser.transform(features).values
    _, encoded = np.unique(labels, return_inverse=True)
    centered_labels = encoded - encoded.mean()
    assert np.linalg.norm((features.values - features.values.mean(0)).T @ centered_labels) > 1
    np.testing.assert_allclose((erased - erased.mean(0)).T @ centered_labels, 0, atol=1e-10)


def test_artifact_roundtrip_checks_identity_and_preserves_fitting_settings(tmp_path):
    eraser = fitted(shrinkage=False, svd_tol=1e-8)
    artifact = tmp_path / 'eraser.json'
    eraser.save(artifact)
    loaded = load_tool('leace', artifact=artifact, name='loaded', conditions=['lesioned'])
    features, _ = discovery()
    np.testing.assert_array_equal(loaded.transform(features), eraser.transform(features))
    assert loaded.provenance == eraser.provenance and loaded.shrinkage is False
    assert loaded.name == 'loaded' and loaded.conditions == ('lesioned',)
    with pytest.raises(FileExistsError):
        loaded.save(artifact)
    payload = json.loads(artifact.read_text())
    payload['payload']['parameters']['bias'][0] += 1
    artifact.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match='checksum'):
        load_tool('leace', artifact=artifact)


def test_subset_live_transform_composes_with_recording_and_restores(tmp_path):
    subject = FixtureSubject()
    features, labels = discovery()
    features = activity(features.values[:, :2], units=[1, 3])
    eraser = load_tool('leace', target='population').fit(features, labels)
    original = subject.layer.register_forward_hook(lambda *args: None)
    values = np.array([[1., 2., 3., 4.]], dtype=np.float32)
    with attach_tools(subject, tools=[load_tool('record-io'),
                      load_tool('record-activity', targets=['population'], when='before', name='before'),
                      eraser, load_tool('record-activity', targets=['population'], name='after')],
                      instrumentation=TorchInstrumentation({'population': subject.layer}),
                      output_dir=tmp_path / 'run'):
        output = subject.layer(torch.from_numpy(values))
        expected = values.copy()
        expected[:, [1, 3]] = eraser.transform(activity(values[:, [1, 3]], units=[1, 3])).values
        np.testing.assert_allclose(output.numpy(), expected, atol=1e-6)
        np.testing.assert_array_equal(output.numpy()[:, [0, 2]], values[:, [0, 2]])
        # Repeated calls are supported; unlike exact donor patching, this is a reusable transform.
        np.testing.assert_allclose(subject.layer(torch.from_numpy(values)).numpy(), expected, atol=1e-6)
    assert list(subject.layer._forward_hooks) == [original.id]
    np.testing.assert_array_equal(subject.layer(torch.from_numpy(values)).numpy(), values)
    events = list(RunRecord(tmp_path / 'run').events())
    assert any(e['kind'] == 'tool_artifact' and e['source'] == 'leace' for e in events)
    original.remove()


def test_session_conditions_keep_baseline_and_restore_model(tmp_path):
    subject = FixtureSubject()
    eraser = fitted(conditions=['lesioned'])
    values = np.array([[1., 2., 3., 4.]], dtype=np.float32)

    @contextmanager
    def factory(trial):
        yield Session([StreamEvent('observation', values, None)])

    result = Experiment(subject=subject,
        protocol=SessionProtocol('leace', factory, conditions=['baseline', 'lesioned', 'restored'],
                                 input_channels=['observation'], output_channels=['behavior']),
        tools=[load_tool('record-io'), eraser],
        instrumentation=TorchInstrumentation({'population': subject.layer}),
        output_dir=tmp_path / 'run').run()
    outputs = list(result.record.outputs())
    np.testing.assert_array_equal(outputs[0].payload, outputs[2].payload)
    np.testing.assert_allclose(outputs[1].payload, eraser.transform(activity(values)).values, atol=1e-6)
    assert not subject.layer._forward_hooks


@pytest.mark.parametrize('failure', ['interrupt', 'missing_forward', 'shape'])
def test_failed_transform_restores_existing_hooks(tmp_path, failure):
    subject = FixtureSubject()
    original = subject.layer.register_forward_hook(lambda *args: None)
    error = KeyboardInterrupt if failure == 'interrupt' else (RuntimeError if failure == 'missing_forward' else ValueError)
    with pytest.raises(error):
        with attach_tools(subject, tools=[load_tool('record-io'), fitted()],
                          instrumentation=TorchInstrumentation({'population': subject.layer}),
                          output_dir=tmp_path / 'run'):
            if failure == 'interrupt':
                subject.layer(torch.ones(1, 4))
                raise KeyboardInterrupt()
            if failure == 'shape':
                subject.layer(torch.ones(1, 2))
    assert list(subject.layer._forward_hooks) == [original.id]
    assert RunRecord(tmp_path / 'run').manifest['status'] == 'failed'
    original.remove()


def test_unfitted_tool_fails_preflight_and_mismatched_units_fail_offline(tmp_path):
    subject = FixtureSubject()
    with pytest.raises(RuntimeError, match='fit'):
        with attach_tools(subject, tools=[load_tool('leace', target='population')],
                          instrumentation=TorchInstrumentation({'population': subject.layer}),
                          output_dir=tmp_path / 'run'):
            pytest.fail('unfitted eraser entered scope')
    assert not (tmp_path / 'run').exists()
    features, _ = discovery()
    with pytest.raises(ValueError):
        fitted().transform(features.isel(neuroid=[3, 2, 1, 0]))


@pytest.mark.parametrize('axis,shape', [(-1, (1, 2, 4)), (1, (1, 4, 2)), (None, (1, 2, 2))])
def test_live_axes_match_offline_transforms_without_pooling(tmp_path, axis, shape):
    subject = FixtureSubject()
    features, labels = discovery()
    target = Selection('population', axis=axis, expected_shape=shape[1:] if axis is None else None)
    eraser = load_tool('leace', target=target).fit(features, labels)
    values = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
    with attach_tools(subject, tools=[eraser],
                      instrumentation=TorchInstrumentation({'population': subject.layer}),
                      output_dir=tmp_path / 'run'):
        result = subject.layer(torch.from_numpy(values)).numpy()
    positioned = values.reshape(1, -1) if axis is None else np.moveaxis(values, axis, -1)
    expected = eraser.transform(activity(positioned.reshape(-1, 4))).values.reshape(positioned.shape)
    expected = expected.reshape(shape) if axis is None else np.moveaxis(expected, -1, axis)
    np.testing.assert_allclose(result, expected, atol=1e-6)


def test_live_tuple_output_and_gradients_match_reference_and_preserve_rng(tmp_path):
    class TupleLayer(torch.nn.Module):
        def forward(self, value):
            return value, 'untouched-cache'

    features, labels = discovery()
    _, encoded = np.unique(labels, return_inverse=True)
    native = LeaceEraser.fit(torch.from_numpy(features.values), torch.from_numpy(np.eye(2)[encoded]))
    eraser = load_tool('leace', target='population').fit(features, labels)
    subject = FixtureSubject()
    subject.layer = TupleLayer()
    inputs = torch.tensor([[1., 2., 3., 4.]], dtype=torch.float64, requires_grad=True)
    expected = native(inputs)
    expected_gradient = torch.autograd.grad(expected.sum(), inputs)[0]
    random_state = torch.random.get_rng_state().clone()
    with attach_tools(subject, tools=[eraser],
                      instrumentation=TorchInstrumentation({'population': subject.layer}),
                      output_dir=tmp_path / 'run'):
        result, cache = subject.layer(inputs)
        assert cache == 'untouched-cache' and subject.layer.training
        torch.testing.assert_close(result, expected, rtol=1e-12, atol=1e-12)
        torch.testing.assert_close(torch.autograd.grad(result.sum(), inputs)[0], expected_gradient,
                                   rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(torch.random.get_rng_state(), random_state, rtol=0, atol=0)
    torch.testing.assert_close(subject.layer(inputs)[0], inputs, rtol=0, atol=0)


@pytest.mark.parametrize('invalid', ['parameters', 'method', 'version', 'units', 'provenance'])
def test_valid_checksum_does_not_bypass_artifact_validation(tmp_path, invalid):
    from brainscore.tools._activity_data import _checksum
    path = tmp_path / 'eraser.json'
    fitted().save(path)
    saved = json.loads(path.read_text())
    payload = saved['payload']
    if invalid == 'parameters':
        payload['parameters']['proj_right'] = [[]]
    elif invalid == 'method':
        payload['method'] = 'another-method'
    elif invalid == 'version':
        payload['schema_version'] = 99
    elif invalid == 'units':
        payload['target']['indices'] = [0, 0, 2, 3]
    else:
        payload['provenance'] = []
    saved['sha256'] = _checksum(payload)
    path.write_text(json.dumps(saved))
    with pytest.raises(ValueError):
        load_tool('leace', artifact=path)


def test_fitted_target_and_artifact_fitting_settings_cannot_silently_change(tmp_path):
    eraser = fitted()
    path = tmp_path / 'eraser.json'
    eraser.save(path)
    with pytest.raises(TypeError, match='fixed by the artifact'):
        load_tool('leace', artifact=path, shrinkage=False)
    eraser.targets[0].indices = (3, 2, 1, 0)
    features, _ = discovery()
    with pytest.raises(ValueError, match='target changed'):
        eraser.transform(features)


@pytest.mark.parametrize('invalid', ['dtype', 'shape', 'nonfinite', 'not_tensor'])
def test_backend_transform_rejects_invalid_maps_and_detaches(invalid):
    subject = FixtureSubject()
    provider = TorchInstrumentation({'population': subject.layer})

    def apply(value):
        if invalid == 'dtype':
            return value.double()
        if invalid == 'shape':
            return value[..., :1]
        if invalid == 'nonfinite':
            return value * float('nan')
        return None

    values = torch.ones(1, 4)
    with pytest.raises(ValueError):
        with provider.transform(['population'], apply=apply):
            subject.layer(values)
    assert not subject.layer._forward_hooks
    torch.testing.assert_close(values, torch.ones(1, 4), rtol=0, atol=0)


def test_missing_fitter_has_an_actionable_error(monkeypatch):
    import sys
    monkeypatch.setitem(sys.modules, 'concept_erasure', None)
    features, labels = discovery()
    with pytest.raises(ImportError, match=r'brainscore\[interpretability\]'):
        load_tool('leace', target='population').fit(features, labels)


@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16])
def test_low_precision_live_application_matches_cast_reference(tmp_path, dtype):
    features, labels = discovery()
    _, encoded = np.unique(labels, return_inverse=True)
    native = LeaceEraser.fit(torch.from_numpy(features.values), torch.from_numpy(np.eye(2)[encoded]))
    native = LeaceEraser(native.proj_left.to(dtype), native.proj_right.to(dtype), native.bias.to(dtype))
    subject = FixtureSubject()
    values = torch.tensor([[1., 2., 3., 4.]], dtype=dtype)
    expected = native(values)
    with attach_tools(subject, tools=[fitted()],
                      instrumentation=TorchInstrumentation({'population': subject.layer}),
                      output_dir=tmp_path / 'run'):
        output = subject.layer(values)
        assert output.dtype == dtype and output.device == values.device
        torch.testing.assert_close(output, expected, rtol=0, atol=0)
    torch.testing.assert_close(subject.layer(values), values, rtol=0, atol=0)


def test_backend_without_transform_support_fails_preflight(tmp_path):
    class RecordOnly:
        def validate(self, operation, targets):
            raise ValueError(f'Unsupported operation: {operation}')

    with pytest.raises(ValueError, match='transform'):
        with attach_tools(FixtureSubject(), tools=[fitted()], instrumentation=RecordOnly(),
                          output_dir=tmp_path / 'run'):
            pytest.fail('unsupported backend entered scope')
    assert not (tmp_path / 'run').exists()


def test_documented_leace_example_runs(tmp_path):
    import subprocess
    import sys
    from pathlib import Path
    script = Path(__file__).resolve().parents[1] / 'examples' / 'leace.py'
    result = subprocess.run([sys.executable, str(script), '--out', str(tmp_path / 'demo')],
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
    assert 'baseline restored exactly' in result.stdout
    record = RunRecord(tmp_path / 'demo' / 'experiment')
    assert len([e for e in record.events() if e['kind'] == 'probe']) == 6
