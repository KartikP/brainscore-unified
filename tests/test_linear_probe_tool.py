"""A fitted probe is reusable, observational, and composes with other tools."""
from contextlib import contextmanager
import json
import subprocess
import sys

import numpy as np
import pytest
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from brainscore.experiments import (
    Experiment, RecordInputsOutputs,
    SessionProtocol, TorchInstrumentation,
)
from brainscore.run_record import RunRecord
from brainscore.tools import attach_tools, list_tools, load_tool, register_tool
from brainscore_core.contract import Subject
from brainscore_core.events import Selection
from brainscore_core.streaming import Session, StreamEvent
from brainscore_core.supported_data_standards.brainio.assemblies import (
    BehavioralAssembly, NeuroidAssembly,
)

pytestmark = pytest.mark.unit


def activity(values, units=(0, 1), *, layer='population'):
    values = np.asarray(values, dtype=float)
    return NeuroidAssembly(values, dims=('presentation', 'neuroid'), coords={
        'stimulus_id': ('presentation', [f'item-{i}' for i in range(len(values))]),
        'stimulus_number': ('presentation', list(range(len(values)))),
        'unit_index': ('neuroid', list(units)),
        'layer': ('neuroid', [layer] * len(units)),
    })


def discovery():
    values = np.random.RandomState(7).normal(size=(80, 2))
    labels = np.where(values[:, 0] > 0, 'positive', 'negative')
    return activity(values), labels


def fitted(**kwargs):
    features, labels = discovery()
    return load_tool('linear-probe', target='population', **kwargs).fit(
        features, labels, provenance={'checkpoint': 'fixture-revision', 'split': 'discovery'})


class FixtureSubject(Subject):
    identifier = 'probe-fixture'
    in_channels, out_channels = {'observation'}, {'behavior'}

    def __init__(self):
        self.layer = torch.nn.Identity()

    def participate(self, session):
        while (event := session.next_input()) is not None:
            value = self.layer(torch.as_tensor(event.payload, dtype=torch.float32))
            session.respond(StreamEvent('behavior', value.numpy(), event.t_ms))


def protocol():
    @contextmanager
    def factory(trial):
        yield Session([StreamEvent('observation', [[-2., 1.], [2., -1.]], None)])
    return SessionProtocol('probe-check', factory, conditions=['baseline', 'lesioned'],
                           input_channels=['observation'], output_channels=['behavior'])


@pytest.mark.parametrize('classes', [2, 3])
def test_predictions_match_independent_sklearn_reference_on_held_out_data(classes):
    rng = np.random.RandomState(10)
    train = activity(rng.normal(size=(100, 2)))
    labels = np.asarray(['abc'[i % classes] for i in range(len(train))])
    test = activity(rng.normal(size=(13, 2)))
    scaler = StandardScaler().fit(train.values)
    reference = LogisticRegression(C=0.7, max_iter=1000).fit(scaler.transform(train), labels)
    probe = load_tool('linear-probe', target='population', classifier_c=0.7).fit(train, labels)
    result = probe.predict(test)
    assert isinstance(result, BehavioralAssembly)
    np.testing.assert_allclose(result.values, reference.predict_proba(scaler.transform(test)), atol=1e-14)
    np.testing.assert_array_equal(result['stimulus_id'].values, test['stimulus_id'].values)
    np.testing.assert_array_equal(result['choice'].values, reference.classes_)
    assert probe.provenance['n_fitting_examples'] == 100


def test_artifact_roundtrip_and_instance_names(tmp_path):
    probe = fitted(name='reading')
    path = tmp_path / 'probe.json'
    probe.save(path)
    loaded = load_tool('linear-probe', artifact=path, name='second', when='before')
    features, _ = discovery()
    np.testing.assert_array_equal(loaded.predict(features), probe.predict(features))
    assert loaded.name == 'second' and loaded.phase == 'before_intervention'
    assert loaded.provenance == probe.provenance
    with pytest.raises(FileExistsError):
        probe.save(path)
    saved = json.loads(path.read_text())
    saved['payload']['mean'][0] += 1
    path.write_text(json.dumps(saved))
    with pytest.raises(ValueError, match='checksum'):
        load_tool('linear-probe', artifact=path)


@pytest.mark.parametrize('invalid', ['units', 'layer', 'nonfinite', 'missing_units'])
def test_predict_rejects_incompatible_activity(invalid):
    probe = fitted()
    features, _ = discovery()
    if invalid == 'units':
        features = features.isel(neuroid=[1, 0])
    elif invalid == 'layer':
        features = activity(features.values, layer='other')
    elif invalid == 'nonfinite':
        features.values[0, 0] = np.nan
    else:
        import xarray as xr
        features = xr.DataArray(features.values, dims=('presentation', 'neuroid'))
    with pytest.raises(ValueError):
        probe.predict(features)


def test_subset_uses_original_units_and_fit_does_not_mutate_learner():
    features, labels = discovery()
    features = activity(features.values, units=(2, 5))
    learner = load_tool('linear-probe', target='population')
    probe = learner.fit(features, labels)
    assert not learner.fitted and probe.fitted
    assert tuple(probe.target.indices) == (2, 5)
    with pytest.raises(RuntimeError, match='fit'):
        learner.predict(features)


def test_two_probes_recording_and_ablation_share_one_protocol_without_changing_outputs(tmp_path):
    subject = FixtureSubject()
    baseline = Experiment(subject=subject, protocol=protocol(), output_dir=tmp_path / 'baseline').run()
    result = Experiment(
        subject=subject, protocol=protocol(),
        tools=[RecordInputsOutputs(), fitted(name='before', when='before'),
               load_tool('ablate', targets=[Selection('population', [0])], conditions=['lesioned']),
               fitted(name='after'), load_tool('record-activity', targets=['population'])],
        instrumentation=TorchInstrumentation({'population': subject.layer}),
        output_dir=tmp_path / 'probed',
    ).run()
    events = list(result.record.events())
    predictions = [event for event in events if event['kind'] == 'probe']
    assert [(e['source'], e['condition']) for e in predictions] == [
        ('before', 'baseline'), ('after', 'baseline'),
        ('before', 'lesioned'), ('after', 'lesioned')]
    assert all(isinstance(e['payload'], BehavioralAssembly) for e in predictions)
    outputs = [event['payload'].payload for event in events if event['kind'] == 'output']
    np.testing.assert_array_equal(outputs[0], baseline.value[0][0].payload)
    np.testing.assert_array_equal(outputs[1], [[0., 1.], [0., -1.]])
    assert not np.array_equal(predictions[-2]['payload'], predictions[-1]['payload'])
    assert not subject.layer._forward_hooks


def test_direct_scope_reuses_probe_and_preserves_model_even_on_interrupt(tmp_path):
    subject = FixtureSubject()
    probe = fitted()
    provider = TorchInstrumentation({'population': subject.layer})
    original = subject.layer.register_forward_hook(lambda *args: None)
    with pytest.raises(KeyboardInterrupt):
        with attach_tools(subject, tools=[RecordInputsOutputs(), probe],
                          instrumentation=provider, output_dir=tmp_path / 'scope'):
            session = Session([StreamEvent('observation', [[2., 1.]], None)])
            subject.participate(session)
            raise KeyboardInterrupt()
    assert list(subject.layer._forward_hooks) == [original.id]
    record = RunRecord(tmp_path / 'scope')
    assert record.manifest['status'] == 'failed'
    assert any(e['kind'] == 'probe' for e in record.events())
    original.remove()


def test_duplicate_names_fail_before_attachment(tmp_path):
    subject = FixtureSubject()
    with pytest.raises(ValueError, match='unique'):
        with attach_tools(subject, tools=[fitted(), fitted()],
                          instrumentation=TorchInstrumentation({'population': subject.layer}),
                          output_dir=tmp_path / 'scope'):
            pytest.fail('invalid scope was entered')
    assert not (tmp_path / 'scope').exists()
    assert not subject.layer._forward_hooks


def test_registry_is_explicit_and_rejects_silent_replacement():
    assert 'linear-probe' in list_tools()
    with pytest.raises(ValueError, match='registered'):
        register_tool('linear-probe', lambda: None)
    with pytest.raises(KeyError, match='unknown-tool'):
        load_tool('unknown-tool')


@pytest.mark.parametrize('axis,shape', [(-1, (2, 3, 2)), (1, (2, 2, 3)), (None, (2, 1, 2))])
def test_hook_predictions_preserve_positions_and_coordinate_system(tmp_path, axis, shape):
    subject = FixtureSubject()
    features, labels = discovery()
    target = Selection('population', axis=axis, expected_shape=shape[1:] if axis is None else None)
    probe = load_tool('linear-probe', target=target).fit(features, labels)
    values = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
    with attach_tools(subject, tools=[RecordInputsOutputs(), probe],
                      instrumentation=TorchInstrumentation({'population': subject.layer}),
                      output_dir=tmp_path / 'scope'):
        result = subject.layer(torch.from_numpy(values))
        np.testing.assert_array_equal(result.numpy(), values)
    predictions = [e['payload'] for e in RunRecord(tmp_path / 'scope').events() if e['kind'] == 'probe']
    expected = values.reshape(2, -1) if axis is None else np.moveaxis(values, axis, -1)
    expected = expected.reshape(-1, 2)
    reference = probe.predict(activity(expected)).values
    np.testing.assert_allclose(predictions[0].values.reshape(-1, 2), reference)
    assert predictions[0].shape[:-1] == ((2,) if axis is None else (2, 3))
    assert not subject.layer._forward_hooks


@pytest.mark.parametrize('labels', [['one'] * 80, ['a', 'b'], np.arange(80), [''] * 80])
def test_invalid_fitting_labels_fail(labels):
    features, _ = discovery()
    with pytest.raises(ValueError, match='label'):
        load_tool('linear-probe', target='population').fit(features, labels)


@pytest.mark.parametrize('failure', ['invalid_target', 'unfitted', 'bad_order'])
def test_collection_validation_leaves_no_hooks_or_files(tmp_path, failure):
    subject = FixtureSubject()
    tools = [RecordInputsOutputs(), fitted()]
    provider = TorchInstrumentation({'population': subject.layer})
    if failure == 'invalid_target':
        provider = TorchInstrumentation({'other': subject.layer})
    elif failure == 'unfitted':
        tools[1] = load_tool('linear-probe', target='population')
    else:
        tools.append(fitted(name='late-before', when='before'))
    with pytest.raises((ValueError, RuntimeError)):
        with attach_tools(subject, tools=tools, instrumentation=provider, output_dir=tmp_path / 'scope'):
            pytest.fail('invalid tools entered')
    assert not subject.layer._forward_hooks and not (tmp_path / 'scope').exists()


def test_later_tool_attachment_failure_detaches_earlier_probe(tmp_path):
    from brainscore.experiments import Tool

    class Broken(Tool):
        name, phase = 'broken', 'after_intervention'

        @contextmanager
        def attach(self, context):
            raise RuntimeError('attachment failed')
            yield

    subject = FixtureSubject()
    with pytest.raises(RuntimeError, match='attachment failed'):
        with attach_tools(subject, tools=[RecordInputsOutputs(), fitted(), Broken()],
                          instrumentation=TorchInstrumentation({'population': subject.layer}),
                          output_dir=tmp_path / 'scope'):
            pytest.fail('broken attachment entered')
    assert not subject.layer._forward_hooks
    assert RunRecord(tmp_path / 'scope').manifest['status'] == 'failed'


@pytest.mark.parametrize('coordinate', ['neuron_number_in_layer', 'neuroid_num'])
def test_existing_wrapper_unit_coordinates_are_accepted(coordinate):
    import xarray as xr
    features, labels = discovery()
    wrapped = xr.DataArray(features.values, dims=('presentation', 'neuroid'), coords={
        coordinate: ('neuroid', [0, 1]), 'layer': ('neuroid', ['population'] * 2)})
    probe = load_tool('linear-probe', target='population').fit(wrapped, labels)
    np.testing.assert_array_equal(probe.predict(wrapped).values, fitted().predict(features).values)


def test_attached_probe_preserves_training_mode_rng_outputs_and_gradients(tmp_path):
    subject = FixtureSubject()
    subject.layer = torch.nn.Linear(2, 2)
    inputs = torch.tensor([[1., 2.]], requires_grad=True)
    baseline = subject.layer(inputs)
    gradients = torch.autograd.grad(baseline.sum(), [inputs, subject.layer.weight])
    random_state = torch.random.get_rng_state().clone()
    with attach_tools(subject, tools=[fitted()],
                      instrumentation=TorchInstrumentation({'population': subject.layer}),
                      output_dir=tmp_path / 'scope'):
        output = subject.layer(inputs)
        observed_gradients = torch.autograd.grad(output.sum(), [inputs, subject.layer.weight])
        torch.testing.assert_close(output, baseline, rtol=0, atol=0)
        for observed, expected in zip(observed_gradients, gradients):
            torch.testing.assert_close(observed, expected, rtol=0, atol=0)
        assert subject.layer.training
    torch.testing.assert_close(torch.random.get_rng_state(), random_state, rtol=0, atol=0)
    assert not subject.layer._forward_hooks


def test_loading_tool_catalog_does_not_import_heavy_model_dependencies():
    result = subprocess.run([sys.executable, '-c',
        'import sys; from brainscore.tools import list_tools, load_tool; '
        'assert "linear-probe" in list_tools(); '
        'load_tool("linear-probe", target="language"); '
        'assert not ({"torch", "transformers", "sklearn"} & set(sys.modules))'],
        capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr


def test_missing_forward_is_an_explicit_failure_with_hooks_removed(tmp_path):
    subject = FixtureSubject()
    with pytest.raises(RuntimeError, match='observed no activity'):
        with attach_tools(subject, tools=[RecordInputsOutputs(), fitted()],
                          instrumentation=TorchInstrumentation({'population': subject.layer}),
                          output_dir=tmp_path / 'scope'):
            cached_response = np.array([[1., 2.]])
            assert cached_response.shape == (1, 2)
    assert not subject.layer._forward_hooks
    assert RunRecord(tmp_path / 'scope').manifest['status'] == 'failed'


def test_documented_cpu_example_runs_without_downloads(tmp_path):
    from pathlib import Path
    script = Path(__file__).resolve().parents[1] / 'examples' / 'linear_probe.py'
    result = subprocess.run([sys.executable, str(script), '--out', str(tmp_path / 'demo')],
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
    assert 'Restored baseline exactly' in result.stdout
    events = list(RunRecord(tmp_path / 'demo' / 'experiment').events())
    assert len([event for event in events if event['kind'] == 'probe']) == 8
    assert len([event for event in events if event['kind'] == 'activity']) == 8
