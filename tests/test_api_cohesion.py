"""A learned selection must address the same units when recording and intervening."""
from contextlib import contextmanager, nullcontext
from types import SimpleNamespace
import numpy as np
import pytest
import torch
import xarray as xr
from brainscore.experiments import SessionProtocol, Trial, TorchInstrumentation
from brainscore_core.model_interface import FunctionalSelection, Selection
from brainscore_core.streaming import InMemorySession


@pytest.mark.parametrize('axis,expected_count', [(None, 1), (-1, 6), (1, 12)])
def test_record_and_ablate_share_tensor_coordinates(axis, expected_count):
    module = torch.nn.Identity()
    provider = TorchInstrumentation({'layer': module})
    selection = Selection(layer='layer', indices=[1], axis=axis)
    inputs = torch.arange(1., 25.).reshape(1, 2, 3, 4)
    captured = []
    with provider.record([selection], lambda name, value: captured.append(value['array'])):
        with provider.ablate([selection]):
            output = module(inputs)
    assert (output == 0).sum().item() == expected_count
    # The recorder was attached before the intervention and saw the original values.
    assert captured[0].size == expected_count
    assert not module._forward_hooks
    torch.testing.assert_close(module(inputs), inputs)


def test_localizer_selection_reuses_spatial_units_in_tools():
    shape = (2, 3, 4)
    positions = np.indices(shape).reshape(3, -1)
    values = np.tile(np.arange(24.), (8, 1))
    values[:4, 1] += 10
    values[::2] += 0.1
    assembly = xr.DataArray(values, dims=('presentation', 'neuroid'), coords={
        'label': ('presentation', ['positive'] * 4 + ['negative'] * 4),
        'layer': ('neuroid', ['layer'] * 24),
        **{name: ('neuroid', pos) for name, pos in zip(('channel', 'channel_x', 'channel_y'), positions)},
    })
    subject = SimpleNamespace(start_recording=lambda target: None, process=lambda stimuli: assembly)
    selection = FunctionalSelection(recording_target='layer', localizer_stimuli=None,
                                    contrast=(['positive'], ['negative']), n_units=1).resolve(subject)
    assert selection.indices == [1] and selection.axis is None and selection.expected_shape == shape
    module = torch.nn.Identity()
    provider = TorchInstrumentation({'layer': module})
    captured = []
    with provider.ablate([selection]), provider.record([selection], lambda name, value: captured.append(value['array'])):
        output = module(torch.ones(1, *shape))
    assert (output == 0).sum().item() == 1
    np.testing.assert_equal(captured[0], [[0]])
    with provider.ablate([selection]), pytest.raises(ValueError, match='layouts differ'):
        module(torch.ones(1, 4, 3, 2))
    assert not module._forward_hooks


@pytest.mark.parametrize('conditions', [None, ['normal'], ['normal', 'changed']])
def test_factory_argument_does_not_change_when_conditions_are_added(conditions):
    requests = []
    @contextmanager
    def factory(trial):
        requests.append(trial)
        yield InMemorySession()
    protocol = SessionProtocol(identifier='stable', session_factory=factory, conditions=conditions)
    subject = SimpleNamespace(reset=lambda: None, interact=lambda session: None)
    context = SimpleNamespace(trial=lambda *a, **k: nullcontext(), publish=lambda *a: None)
    protocol.run(subject, context)
    assert all(isinstance(request, Trial) and request.identifier == 0 for request in requests)


def test_identifier_factory_is_an_explicit_compatibility_option():
    requests = []
    @contextmanager
    def factory(identifier):
        requests.append(identifier)
        yield InMemorySession()
    protocol = SessionProtocol(identifier='old', session_factory=factory, factory_argument='identifier')
    protocol.run(SimpleNamespace(reset=lambda: None, interact=lambda session: None),
                 SimpleNamespace(trial=lambda *a, **k: nullcontext(), publish=lambda *a: None))
    assert requests == [0]


@pytest.mark.parametrize('operation', ['record', 'ablate', 'scale'])
def test_noncontiguous_flat_selection_preserves_layout_and_source(operation):
    module = torch.nn.Identity()
    provider = TorchInstrumentation({'layer': module})
    inputs = torch.arange(1., 25.).reshape(1, 2, 3, 4).transpose(1, 3)
    original = inputs.clone()
    target = Selection(layer='layer', indices=[1], axis=None, expected_shape=tuple(inputs.shape[1:]))
    seen = []
    scope = (provider.record([target], lambda name, value: seen.append(value['array']))
             if operation == 'record' else provider.ablate([target]) if operation == 'ablate'
             else provider.scale([target], factor=0.5))
    with scope:
        output = module(inputs)
    torch.testing.assert_close(inputs, original)
    if operation == 'record':
        np.testing.assert_equal(seen[0], original.reshape(1, -1)[:, [1]].numpy())
    else:
        expected = original.reshape(1, -1).clone()
        expected[:, 1] *= 0 if operation == 'ablate' else 0.5
        torch.testing.assert_close(output, expected.reshape(inputs.shape))
    assert not module._forward_hooks


def test_unsupported_coordinate_system_is_rejected_before_attaching_tools():
    target = Selection(layer='layer', indices=[1], axis=None)
    provider = TorchInstrumentation({'layer': torch.nn.Identity()})
    with pytest.raises(ValueError, match='last-axis'):
        provider.validate('attribute', [target])
    from brainscore.experiments.backends.openpi_remote import RemoteOpenPIInstrumentation
    with pytest.raises(ValueError, match='last-axis'):
        RemoteOpenPIInstrumentation._scope('record', [target])


def test_real_vision_extraction_localizer_and_ablation_share_one_feature(tmp_path, monkeypatch):
    from PIL import Image
    from brainscore_core.model_interface import BrainScoreModel
    from brainscore_core.supported_data_standards.brainio.stimuli import StimulusSet
    from brainscore_vision.model_helpers.activations.pytorch import PytorchWrapper

    monkeypatch.setenv('RESULTCACHING_DISABLE', '1')
    network = torch.nn.Sequential(torch.nn.Conv2d(1, 2, kernel_size=1, bias=False)).eval()
    with torch.no_grad():
        network[0].weight.copy_(torch.tensor([1., 2.]).reshape(2, 1, 1, 1))
    paths = {}
    for index, intensity in enumerate((20, 21, 2, 3)):
        pixels = np.full((2, 2), 10, dtype=np.uint8)
        pixels[0, 1] = intensity
        path = tmp_path / f'{index}.png'
        Image.fromarray(pixels).save(path)
        paths[str(index)] = path
    stimuli = StimulusSet({
        'stimulus_id': list(paths),
        'image_path': [str(path) for path in paths.values()],
        'label': ['positive', 'positive', 'negative', 'negative'],
    })
    stimuli.stimulus_paths = paths
    stimuli.identifier = 'spatial-localizer-test'

    def prepare(image_paths):
        arrays = []
        for path in image_paths:
            with Image.open(path) as image:
                arrays.append(np.asarray(image, dtype=np.float32)[None])
        return np.stack(arrays)

    wrapper = PytorchWrapper(
        identifier='spatial-cnn',
        model=network,
        preprocessing=prepare,
    )
    subject = BrainScoreModel(
        identifier='spatial-cnn',
        activations_model={'vision': wrapper},
        region_layer_map={'IT': '0'},
    )
    selected = FunctionalSelection(
        recording_target='IT',
        localizer_stimuli=stimuli,
        contrast=(['positive'], ['negative']),
        n_units=1,
    ).resolve(subject)
    assert selected.axis is None
    assert selected.expected_shape == (2, 2, 2)
    assert selected.indices[0] in (1, 5)
    baseline = subject.process(stimuli)
    provider = TorchInstrumentation(network)
    with provider.ablate([selected]):
        changed = subject.process(stimuli)
    assert np.count_nonzero(changed.values == 0) == len(stimuli)
    assert np.flatnonzero(changed.values[0] == 0).tolist() == selected.indices
    restored = subject.process(stimuli)
    np.testing.assert_array_equal(restored.values, baseline.values)
    assert not network[0]._forward_hooks


def test_experiment_warns_about_unregistered_channels_before_running():
    from brainscore_core.compatibility import CompatibilityWarning, CompatibilityError
    from brainscore_core.contract import Subject

    class Exploratory(Subject):
        identifier = 'exploratory'
        in_channels = {'unregistered_input'}
        out_channels = {'behavior'}

        def interact(self, session):
            pass

    subject = Exploratory()
    protocol = SessionProtocol(
        identifier='exploratory', session_factory=lambda trial: None,
        input_channels=['unregistered_input'], output_channels=['behavior'],
    )
    with pytest.warns(CompatibilityWarning, match='benchmark scoring requires'):
        protocol.validate(subject)
    subject.in_channels = {'behavior'}
    with pytest.raises(CompatibilityError, match='registry marks'):
        protocol.validate(subject)


def test_resnet_notebook_subject_works_in_experiments_and_scoring(tmp_path):
    """Execute the notebook's actual adapter with a tiny network, without weights."""
    import ast
    import json
    from pathlib import Path
    from brainscore import score
    from brainscore_core import Benchmark, Score, Subject
    from brainscore_core.streaming import StreamEvent
    from brainscore.experiments import Experiment
    notebook = json.loads((Path(__file__).parents[1] / 'notebooks/18_resnet_experiment_toolbox.ipynb').read_text())
    source = next(''.join(cell['source']) for cell in notebook['cells']
                  if 'class ImageSubject(Subject):' in ''.join(cell.get('source', [])))
    tree = ast.parse(source)
    adapter = next(node for node in tree.body if isinstance(node, ast.ClassDef))
    from brainscore_core.supported_data_standards.brainio.assemblies import BehavioralAssembly
    namespace = dict(Subject=Subject, torch=torch, np=np, StreamEvent=StreamEvent,
                     BehavioralAssembly=BehavioralAssembly)
    exec(compile(ast.Module(body=[adapter], type_ignores=[]), '<notebook-adapter>', 'exec'), namespace)
    subject = namespace['ImageSubject'](torch.nn.Linear(3, 2).eval())
    inputs = np.array([[1., 2., 3.]], dtype=np.float32)
    def session():
        return InMemorySession(
            inputs=[StreamEvent(channel='vision', payload=inputs.copy(), t_ms=0)],
            requested_output_channels=['behavior'],
        )
    class SessionBenchmark(Benchmark):
        identifier = 'notebook-channel-transfer'
        parent = 'behavioral'
        uses_session = True
        required_input_channels = {'vision'}
        requested_output_channels = {'behavior'}
        def __call__(self, candidate):
            run = session()
            candidate.interact(run)
            return Score(float(run.collect(channel='behavior').sum()))
    result = score(subject, SessionBenchmark(), check_mem=False)
    assert float(result) == pytest.approx(1)
    experiment = Experiment(
        subject=subject,
        protocol=SessionProtocol(
            identifier='notebook-channels', session_factory=lambda trial: nullcontext(session()),
            input_channels=['vision'], output_channels=['behavior'],
        ),
        output_dir=tmp_path / 'experiment',
    )
    assert experiment.run().manifest['status'] == 'complete'


@pytest.mark.parametrize('prefer_path', ['auto', 'generation', 'readout'])
def test_roar_declares_requested_path_and_distinct_score_name(monkeypatch, prefer_path):
    from brainscore.benchmarks.roar_yeatman2021 import benchmark as module
    from brainscore_core.supported_data_standards.brainio.stimuli import StimulusSet
    from brainscore_core.supported_data_standards.brainio.assemblies import BehavioralAssembly
    ids = [f's{i}' for i in range(500)]
    labels = ['real'] * 250 + ['pseudo'] * 250
    stimuli = StimulusSet({'stimulus_id': ids, 'sentence': ids, 'image_label': labels})
    stimuli.identifier = 'synthetic-roar'
    stimuli.stimulus_paths = {sid: sid for sid in ids}
    human = xr.Dataset({
        'data': ('presentation', np.ones(500)),
        'correct': ('presentation', np.ones(500)),
        'stimulus_id': ('presentation', ids),
    })
    monkeypatch.setattr(module, 'load_stimulus_set', lambda identifier: stimuli)
    monkeypatch.setattr(module, 'load_dataset', lambda identifier: human)
    benchmark = module.Yeatman2021LexicalDecision(modality='text', prefer_path=prefer_path)
    suffix = '' if prefer_path == 'auto' else f'-{prefer_path}'
    assert benchmark.identifier == f'Yeatman2021-lexical_decision-text{suffix}'
    tasks = []
    def predict(stimuli):
        return BehavioralAssembly(
            np.array([[1, 0] if label == 'real' else [0, 1] for label in stimuli.image_label]),
            dims=('presentation', 'choice'),
            coords={'stimulus_id': ('presentation', list(stimuli.stimulus_id)),
                    'stimulus_index': ('presentation', np.arange(len(stimuli))),
                    'choice': ('choice', ['real', 'pseudo'])},
        )
    result = benchmark(SimpleNamespace(start_task=tasks.append, process=predict))
    assert tasks[0].prefer_path == prefer_path
    assert float(result) == 1
    assert result.attrs['requested_behavioral_path'] == prefer_path
    assert not set(benchmark._train_stimuli.stimulus_id) & set(benchmark._test_stimuli.stimulus_id)


def test_new_model_template_supports_public_recording_workflow(tmp_path):
    import importlib.util
    from pathlib import Path
    from PIL import Image
    from brainscore_core.supported_data_standards.brainio.stimuli import StimulusSet
    path = Path(__file__).parents[1] / 'templates/new_model/model.py'
    spec = importlib.util.spec_from_file_location('umi_example_model', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    subject = module.get_model()
    image = tmp_path / 'image.png'
    Image.fromarray(np.zeros((64, 64, 3), dtype=np.uint8)).save(image)
    stimuli = StimulusSet({'stimulus_id': ['s0'], 'image_path': [str(image)]})
    stimuli.identifier = 'template-smoke'
    stimuli.stimulus_paths = {'s0': image}
    assert subject.supported_modalities == {'vision'}
    subject.start_recording('IT')
    result = subject.process(stimuli)
    assert result.sizes['presentation'] == 1 and result.sizes['neuroid'] > 0
    # The template's raw-layer example must agree with the public API.
    for target in ('2', ['2']):
        with pytest.warns(UserWarning, match='raw layer path'):
            subject.start_recording(target)
        np.testing.assert_array_equal(subject.process(stimuli).values, result.values)
    subject.reset()
