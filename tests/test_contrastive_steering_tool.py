"""Contrastive directions and signed doses are explicit, reusable and reversible."""
from contextlib import contextmanager
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import torch

from brainscore.experiments import Experiment, SessionProtocol, TorchInstrumentation
from brainscore.run_record import RunRecord
from brainscore.tools import attach_tools, load_tool
from brainscore_core.streaming import BufferedSession, StreamEvent
from brainscore_core.events import Selection
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


def discovery():
    labels = np.asarray(['positive', 'negative'] * 30)
    values = np.random.RandomState(42).normal(size=(60, 3))
    values[labels == 'positive', 0] += 2.5
    return activity(values), labels


def fitted(**configuration):
    features, labels = discovery()
    return load_tool('contrastive-steering', target='population', **configuration).fit(
        features, labels, positive='positive', negative='negative', provenance={'checkpoint': 'fixture-v1'})


@pytest.mark.parametrize('strength', [-1., 0., 0.5, 1.])
@pytest.mark.parametrize('normalize', [False, True])
def test_held_out_transforms_match_independent_group_means(strength, normalize):
    features, labels = discovery()
    direction = features.values[labels == 'positive'].mean(0) - features.values[labels == 'negative'].mean(0)
    if normalize:
        direction /= np.linalg.norm(direction)
    learner = load_tool('contrastive-steering', target='population', strength=strength)
    tool = learner.fit(features, labels, positive='positive', negative='negative', normalize=normalize)
    held_out = activity(np.random.RandomState(8).normal(size=(9, 3)))
    transformed = tool.transform(held_out)
    assert not learner.fitted and tool.fitted and isinstance(transformed, NeuroidAssembly)
    np.testing.assert_allclose(transformed.values, held_out.values + strength * direction, atol=1e-12)
    np.testing.assert_array_equal(transformed['stimulus_id'].values, held_out['stimulus_id'].values)
    assert transformed.attrs['application_provenance']['configuration'] == {'strength': strength}


def test_artifact_reloads_default_dose_and_accepts_explicit_override(tmp_path):
    original = fitted(strength=0.25)
    path = tmp_path / 'direction.json'
    original.save(path)
    loaded = load_tool('contrastive-steering', artifact=path)
    stronger = load_tool('contrastive-steering', artifact=path, strength=1., name='stronger')
    features, _ = discovery()
    np.testing.assert_array_equal(loaded.transform(features), original.transform(features))
    assert loaded.strength == 0.25 and stronger.strength == 1.
    np.testing.assert_allclose(stronger.transform(features).values - features.values,
                               4 * (loaded.transform(features).values - features.values), atol=1e-12)
    assert loaded.provenance == stronger.provenance


def test_signed_protocol_conditions_and_multiple_tools_restore(tmp_path):
    subject = FixtureSubject()
    path = tmp_path / 'direction.json'
    fitted().save(path)
    values = np.array([[1., 2., 3.]], dtype=np.float32)

    @contextmanager
    def factory(trial):
        yield BufferedSession([StreamEvent('stimulus', values, None)])

    result = Experiment(subject=subject,
        protocol=SessionProtocol('steering', factory,
            conditions=['baseline', 'positive', 'negative', 'restored'],
            input_channels=['stimulus'], output_channels=['behavior']),
        tools=[load_tool('record-io'),
               load_tool('contrastive-steering', artifact=path, strength=1., name='plus', conditions=['positive']),
               load_tool('contrastive-steering', artifact=path, strength=-1., name='minus', conditions=['negative']),
               load_tool('record-activity', targets=['population'])],
        instrumentation=TorchInstrumentation({'population': subject.layer}), output_dir=tmp_path / 'run').run()
    outputs = list(result.record.outputs())
    np.testing.assert_array_equal(outputs[0].payload, outputs[3].payload)
    for index, sign in [(1, 1.), (2, -1.)]:
        expected = load_tool('contrastive-steering', artifact=path, strength=sign).transform(activity(values))
        np.testing.assert_allclose(outputs[index].payload, expected.values, atol=1e-6)
    assert not subject.layer._forward_hooks


def test_application_configuration_is_frozen_for_each_scope(tmp_path):
    subject = FixtureSubject()
    tool = fitted(strength=0.5)
    values = torch.ones(1, 3)
    with attach_tools(subject, tools=[load_tool('record-io'), tool],
                      instrumentation=TorchInstrumentation({'population': subject.layer}),
                      output_dir=tmp_path / 'run'):
        first = subject.layer(values)
        tool.strength = 99.
        second = subject.layer(values)
        torch.testing.assert_close(first, second, rtol=0, atol=0)
    events = list(RunRecord(tmp_path / 'run').events())
    applications = [e for e in events if e['kind'] == 'intervention_application']
    assert all(e['payload']['configuration'] == {'strength': 0.5} for e in applications)
    assert all(e['payload']['strength'] == 0.5 for e in events if e['kind'] in ('intervention_start', 'intervention_end'))


@pytest.mark.parametrize('scale_first', [True, False])
def test_scaling_and_steering_follow_list_order(tmp_path, scale_first):
    subject = FixtureSubject()
    steer = fitted(strength=0.5)
    scale = load_tool('scale-activity', targets=['population'], factor=2.)
    values = np.array([[1., 2., 3.]], dtype=np.float32)
    expected = steer.transform(activity(values * 2)).values if scale_first else steer.transform(activity(values)).values * 2
    tools = [scale, steer] if scale_first else [steer, scale]
    with attach_tools(subject, tools=tools,
                      instrumentation=TorchInstrumentation({'population': subject.layer}),
                      output_dir=tmp_path / 'run'):
        np.testing.assert_allclose(subject.layer(torch.from_numpy(values)).numpy(), expected, atol=1e-6)
    np.testing.assert_array_equal(subject.layer(torch.from_numpy(values)).numpy(), values)


def test_zero_direction_is_valid_until_normalization_is_requested():
    features = activity(np.zeros((4, 3)))
    labels = ['positive', 'negative'] * 2
    learner = load_tool('contrastive-steering', target='population')
    tool = learner.fit(features, labels, positive='positive', negative='negative')
    np.testing.assert_array_equal(tool.transform(features), features)
    with pytest.raises(ValueError, match='zero'):
        learner.fit(features, labels, positive='positive', negative='negative', normalize=True)


@pytest.mark.parametrize('invalid', ['same', 'missing', 'extra', 'normalization'])
def test_invalid_contrasts_are_not_silently_reinterpreted(invalid):
    features, labels = discovery()
    positive, negative, normalize = 'positive', 'negative', False
    if invalid == 'same':
        negative = positive
    elif invalid == 'missing':
        positive = 'absent'
    elif invalid == 'extra':
        labels[0] = 'other'
    else:
        normalize = 'yes'
    with pytest.raises(ValueError):
        load_tool('contrastive-steering', target='population').fit(
            features, labels, positive=positive, negative=negative, normalize=normalize)


@pytest.mark.parametrize('strength', [True, '1', float('nan'), float('inf')])
def test_invalid_doses_fail_during_loading(strength):
    with pytest.raises(ValueError, match='strength'):
        load_tool('contrastive-steering', target='population', strength=strength)


@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16, torch.float64])
def test_live_dtype_and_gradients_preserve_additive_semantics(tmp_path, dtype):
    subject = FixtureSubject()
    features, labels = discovery()
    direction = features.values[labels == 'positive'].mean(0) - features.values[labels == 'negative'].mean(0)
    tool = fitted(strength=0.25)
    values = torch.ones(1, 3, dtype=dtype, requires_grad=True)
    expected = values + torch.tensor(direction, dtype=dtype) * 0.25
    with attach_tools(subject, tools=[tool],
                      instrumentation=TorchInstrumentation({'population': subject.layer}),
                      output_dir=tmp_path / 'run'):
        output = subject.layer(values)
        assert output.dtype == dtype
        torch.testing.assert_close(output, expected, rtol=0, atol=0)
        torch.testing.assert_close(torch.autograd.grad(output.sum(), values)[0], torch.ones_like(values), rtol=0, atol=0)
    torch.testing.assert_close(subject.layer(values), values, rtol=0, atol=0)


def test_valid_checksum_does_not_allow_invalid_direction(tmp_path):
    from brainscore.tools._activity_data import _checksum
    path = tmp_path / 'direction.json'
    fitted().save(path)
    saved = json.loads(path.read_text())
    saved['payload']['parameters']['direction'] = [1.]
    saved['sha256'] = _checksum(saved['payload'])
    path.write_text(json.dumps(saved))
    with pytest.raises(ValueError, match='parameters'):
        load_tool('contrastive-steering', artifact=path)


def test_original_unit_addresses_broadcast_at_every_position_and_cleanup_on_failure(tmp_path):
    subject = FixtureSubject()
    features, labels = discovery()
    selected = activity(features.values[:, [2, 0]], units=[2, 0])
    tool = load_tool('contrastive-steering', target=Selection('population', [2, 0])).fit(
        selected, labels, positive='positive', negative='negative')
    direction = selected.values[labels == 'positive'].mean(0) - selected.values[labels == 'negative'].mean(0)
    values = torch.ones(2, 4, 3)
    expected = values.clone()
    expected[..., [2, 0]] += torch.tensor(direction, dtype=values.dtype)
    with pytest.raises(KeyboardInterrupt):
        with attach_tools(subject, tools=[tool],
                          instrumentation=TorchInstrumentation({'population': subject.layer}),
                          output_dir=tmp_path / 'run'):
            torch.testing.assert_close(subject.layer(values), expected)
            torch.testing.assert_close(subject.layer(values), expected)
            raise KeyboardInterrupt
    torch.testing.assert_close(subject.layer(values), values, rtol=0, atol=0)
    assert not subject.layer._forward_hooks
    with pytest.raises(ValueError, match='units or their order'):
        tool.transform(activity(selected.values, units=[0, 2]))


def test_dose_cannot_be_changed_to_a_nonfinite_value():
    tool = fitted()
    with pytest.raises(ValueError, match='strength'):
        tool.strength = float('nan')
    assert tool.strength == 1.


def test_saved_maps_load_and_apply_without_fitting_or_tensor_libraries(tmp_path):
    features, labels = discovery()
    fitted().save(tmp_path / 'steering.json')
    load_tool('leace', target='population').fit(features, labels).save(tmp_path / 'leace.json')
    script = '''
import sys
import numpy as np
from brainscore.tools import list_tools, load_tool
from brainscore_core.supported_data_standards.brainio.assemblies import NeuroidAssembly
assert 'contrastive-steering' in list_tools()
activity = NeuroidAssembly(np.ones((2, 3)), dims=('presentation', 'neuroid'), coords={
    'stimulus_id': ('presentation', ['one', 'two']),
    'stimulus_number': ('presentation', [0, 1]),
    'layer': ('neuroid', ['population'] * 3), 'unit_index': ('neuroid', [0, 1, 2])})
for method, path in zip(['contrastive-steering', 'leace'], sys.argv[1:]):
    transformed = load_tool(method, artifact=path).transform(activity)
    assert isinstance(transformed, NeuroidAssembly) and transformed.shape == activity.shape
assert not ({'torch', 'transformers', 'sklearn', 'concept_erasure'} & set(sys.modules))
'''
    result = subprocess.run([sys.executable, '-c', script, str(tmp_path / 'steering.json'),
                             str(tmp_path / 'leace.json')], capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr


def test_documented_steering_example_runs(tmp_path):
    script = Path(__file__).resolve().parents[1] / 'examples' / 'contrastive_steering.py'
    result = subprocess.run([sys.executable, str(script), '--out', str(tmp_path / 'demo')],
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
    assert 'baseline restored exactly' in result.stdout
    record = RunRecord(tmp_path / 'demo' / 'experiment')
    assert len([e for e in record.events() if e['kind'] == 'probe']) == 8
