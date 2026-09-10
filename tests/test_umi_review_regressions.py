"""Offline cross-repository regressions for the 2026-09-08 UMI review.

No weights, datasets or network: tiny CPU models and synthetic assemblies.
"""
import numpy as np
import pytest
import torch
from torch import nn

from brainscore_core.model_interface import (
    BrainScoreModel, CompositeSelector, RandomSelection,
    TaskContext, StateChange, Perturbation, EnvironmentStep,
)
from brainscore_core.supported_data_standards.brainio.assemblies import (
    NeuroidAssembly, BehavioralAssembly,
)
from brainscore_core.supported_data_standards.brainio.stimuli import StimulusSet

pytestmark = pytest.mark.integration


def _stimuli():
    stimuli = StimulusSet({
        'stimulus_id': ['s0', 's1', 's2', 's3'],
        'sentence': ['the cat', 'the dog', 'cat sat', 'dog ran'],
        'image_label': ['real', 'real', 'pseudo', 'pseudo'],
    })
    stimuli.identifier = 'umi-review-synthetic'
    return stimuli


class _Extractor:
    identifier = 'synthetic-extractor'

    def __init__(self, values):
        self.values = values

    def __call__(self, stimuli, layers):
        values = self.values() if callable(self.values) else self.values
        blocks = [np.asarray(values)[:len(stimuli)] for _ in layers]
        width = blocks[0].shape[1]
        return NeuroidAssembly(np.concatenate(blocks, axis=1),
            dims=['presentation', 'neuroid'], coords={
                **{k: ('presentation', list(stimuli[k])) for k in stimuli.columns},
                'layer': ('neuroid', [layer for layer in layers for _ in range(width)]),
                'neuroid_id': ('neuroid', [f'{layer}.{i}' for layer in layers for i in range(width)]),
            })


def test_algonauts_prediction_uses_training_svd(tmp_path, monkeypatch):
    from brainscore.benchmarks.algonauts2025.benchmark import (
        Algonauts2025Friends, Algonauts2025FriendsS7,
    )
    train_features = np.array([[4, 0], [-4, 0], [2, 0], [-2, 0]], dtype=np.float32)
    test_features = np.array([[1, 10], [-1, 10], [1, -10], [-1, -10]], dtype=np.float32)
    train = Algonauts2025Friends(subject=1, stimulus_window=1, hrf_delay=0,
        excluded_samples_start=0, excluded_samples_end=0)
    test = Algonauts2025FriendsS7(subject=1, stimulus_window=1, hrf_delay=0)
    for benchmark in (train, test):
        benchmark.FEATURE_DIM_CAP = 1
        stimuli = _stimuli()
        stimuli.identifier = benchmark._split
        benchmark._assembly = NeuroidAssembly(train_features[:, :1],
            dims=['presentation', 'neuroid'], coords={
                'stimulus_id': ('presentation', ['clip'] * 4),
                'run': ('presentation', ['run'] * 4),
                'neuroid_id': ('neuroid', ['target']),
            })
        monkeypatch.setattr(benchmark, '_expand_to_per_TR_frames', lambda s=stimuli: s)
        monkeypatch.setattr(benchmark, '_align_features_to_assembly', lambda features, *args: features)

    class Candidate:
        def start_recording(self, *args, **kwargs):
            pass

        def process(self, stimuli):
            features = train_features if stimuli.identifier == 'friends' else test_features
            return _Extractor(features)(stimuli, ['L'])

    result = test.generate_predictions(Candidate(), tmp_path, train_benchmark=train, alpha=1e-6)
    np.testing.assert_allclose(result['predictions'].ravel(), [1, -1, 1, -1], atol=1e-4)
    np.testing.assert_array_equal(np.load(result['path']), result['predictions'])


def test_feature_projection_validates_basis_and_preserves_no_compression():
    from brainscore.benchmarks.algonauts2025.projection import FeatureProjection
    projection = FeatureProjection(5)
    features = np.eye(2, dtype=np.float32)
    with pytest.raises(ValueError, match='training data'):
        projection.transform(features)
    np.testing.assert_array_equal(projection.fit_transform(features, ['a', 'b']), features)
    np.testing.assert_array_equal(projection.transform(features, ['a', 'b']), features)
    with pytest.raises(ValueError, match='identities/order'):
        projection.transform(features, ['b', 'a'])
    with pytest.raises(ValueError, match='width'):
        projection.transform(np.ones((2, 3)))


@pytest.fixture
def language_pair(monkeypatch):
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast, GPT2Config, GPT2LMHeadModel
    from brainscore.model_helpers.text_wrapper import TextWrapper
    from brainscore_language.model_helpers.huggingface import HuggingfaceSubject
    from brainscore_language.compat.unified_adapter import LanguageModelAdapter

    vocab = {'[PAD]': 0, '[UNK]': 1, 'the': 2, 'cat': 3, 'dog': 4, 'sat': 5, 'ran': 6}
    tokenizer = Tokenizer(models.WordLevel(vocab=vocab, unk_token='[UNK]'))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=tokenizer, unk_token='[UNK]',
                                       pad_token='[PAD]', model_max_length=64)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(17)
        net = GPT2LMHeadModel(GPT2Config(vocab_size=7, n_positions=64, n_embd=8,
                                       n_layer=1, n_head=2)).eval()
    monkeypatch.setattr('brainscore.model_helpers._device.select_device', lambda: 'cpu')
    monkeypatch.setattr(torch.backends.mps, 'is_available', lambda: False)
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    wrapper = TextWrapper(net, tokenizer, identifier='tiny-new', max_length=64)
    legacy = HuggingfaceSubject('tiny-legacy', {'language_system': 'transformer.h.0'},
                               model=net, tokenizer=tokenizer)
    native = BrainScoreModel('tiny-new', region_layer_map={'language_system': 'transformer.h.0'},
                             preprocessors={'text': wrapper})
    adapter = LanguageModelAdapter(legacy)
    yield native, adapter
    native.reset()
    adapter.reset()


@pytest.mark.parametrize('groups', [None, ['p', 'p', 'p', 'p'], ['a', 'b', 'a', 'b']])
def test_text_context_and_metadata_match_across_wrappers(language_pair, groups):
    from brainscore.benchmarks.lebel2023.benchmark import LeBel2023Encoding
    native, adapter = language_pair
    stimuli = _stimuli()
    if groups is not None:
        stimuli['context_id'] = groups
    for model in (native, adapter):
        model.start_recording('language_system')
    expected, actual = native.process(stimuli), adapter.process(stimuli)
    np.testing.assert_allclose(actual.values, expected.values, atol=1e-6)
    for column in stimuli.columns:
        assert list(actual[column].values) == list(stimuli[column])
    aligner = LeBel2023Encoding.__new__(LeBel2023Encoding)
    np.testing.assert_allclose(aligner._align_features(actual, expected.isel(presentation=[3, 1])),
                               expected.values[[3, 1]], atol=1e-6)


def test_explicit_passage_changes_context_and_legacy_digest_is_preserved(language_pair):
    native, adapter = language_pair
    stimuli = _stimuli().iloc[:2].copy()
    for model in (native, adapter):
        model.start_recording('language_system')
    independent = native.process(stimuli)
    stimuli['context_id'] = 'passage'
    passage = native.process(stimuli)
    assert not np.allclose(passage.values[1], independent.values[1])
    np.testing.assert_allclose(adapter.digest_text(list(stimuli.sentence))['neural'].values,
                               passage.values, atol=1e-6)
    np.testing.assert_allclose(native.digest_text(list(stimuli.sentence))['neural'].values,
                               passage.values, atol=1e-6)


def test_language_reset_clears_real_wrapped_measurements(language_pair):
    _, adapter = language_pair
    adapter.start_recording('language_system')
    adapter.start_task(TaskContext('next_word'))
    adapter.reset()
    assert adapter._legacy.neural_recordings == []
    assert adapter._legacy.behavioral_task is None
    adapter.start_task(TaskContext('next_word'))
    assert isinstance(adapter.process(_stimuli()), BehavioralAssembly)


@pytest.mark.parametrize('composite', [False, True])
def test_shared_layer_names_keep_towers_and_regions_separate(composite):
    selector = CompositeSelector((('block', (1,)),)) if composite else 'block'
    model = BrainScoreModel('two-towers',
        region_layer_map={'IT': selector, 'language_system': selector},
        region_modality_map={'IT': 'vision', 'language_system': 'text'},
        preprocessors={'vision': _Extractor(np.ones((4, 2))),
                       'text': _Extractor(np.full((4, 2), 9.0))})
    stimuli = _stimuli()
    stimuli['image_path'] = ['unused.png'] * 4
    model.start_recording(['IT', 'language_system'])
    actual = model.process(stimuli, multi_modality=True)
    width = 1 if composite else 2
    assert actual.shape == (4, 2 * width)
    assert list(actual.region.values) == ['IT'] * width + ['language_system'] * width
    np.testing.assert_array_equal(actual.values[:, :width], 1)
    np.testing.assert_array_equal(actual.values[:, width:], 9)
    # Single-region recording chooses its declared tower even with both inputs.
    model.start_recording('language_system')
    with pytest.warns(UserWarning, match='multiple supported modalities'):
        np.testing.assert_array_equal(model.process(stimuli).values, 9)
    model.start_recording(['IT', 'language_system'])
    with pytest.raises(ValueError, match='multi_modality=True'):
        model.process(stimuli)
    with pytest.raises(ValueError, match='missing input modalities'):
        model.process(stimuli.drop(columns='sentence'), multi_modality=True)


@pytest.mark.parametrize('layers', [('L',), ('L', 'M')])
def test_localized_and_random_lesions_use_original_subset_indices(layers):
    from brainscore.perturbation import build_pytorch_ablation_fn
    from brainscore.benchmarks.induced_dyslexia.benchmark import InducedDyslexia
    net = nn.Module()
    for layer in layers:
        linear = nn.Linear(5, 5, bias=False)
        with torch.no_grad():
            linear.weight.copy_(torch.eye(5))
        net.add_module(layer, linear)
    x = torch.tensor([[3, 1, 3, 3, 10], [3, 2, 3, 3, 11],
                      [3, 1, 3, 3, 0], [3, 2, 3, 3, 1]], dtype=torch.float32)
    model = BrainScoreModel('subset', model=net,
        region_layer_map={'VWFA': CompositeSelector(tuple((layer, (4, 1)) for layer in layers))},
        preprocessors={'text': _Extractor(lambda: net.L(x).detach().numpy())},
        state_change_fn=build_pytorch_ablation_fn(net))
    localizer = InducedDyslexia(n_units=1, modality='text',
        reading_benchmark=object(), localizer_stimuli=_stimuli())
    selections = localizer._localize(model, list(layers))
    for selected in selections:
        assert selected.indices == [4]
        assert selected.metadata['unit_population'] == [4, 1]
        random = RandomSelection(selected.layer, n_units=2, n_total=2,
                                 population=selected.metadata['unit_population']).resolve(model)
        assert random.indices == [1, 4]
        model.process(StateChange('ablation', target=selected, perturbation=Perturbation('zero')))
        after = getattr(net, selected.layer)(x).detach().numpy()
        assert np.all(after[:, 4] == 0)
        np.testing.assert_array_equal(after[:, 0], x[:, 0].numpy())
    model.reset()
    np.testing.assert_array_equal(net.L(x).detach().numpy(), x.numpy())


def test_reset_and_readout_clear_composite_recording():
    model = BrainScoreModel('composite-reset',
        region_layer_map={'IT': CompositeSelector((('L', (0,)),))},
        preprocessors={'text': _Extractor(np.ones((4, 2)))}, behavioral_readout_layer='L')
    model.start_recording('IT', time_bins=[(0, 1)])
    # Readout extraction must also work while composite recording is configured.
    task = TaskContext('probabilities', fitting_stimuli=_stimuli(), prefer_path='readout')
    model.start_task(task)
    assert isinstance(model.process(_stimuli()), BehavioralAssembly)
    model.reset()
    assert not model._composite_recording and model._time_bins is None
    model.start_task(task)
    assert isinstance(model.process(_stimuli()), BehavioralAssembly)


@pytest.mark.parametrize('bound_method', [False, True])
def test_model_reset_clears_policy_history(bound_method):
    from brainscore.model_helpers.policy_wrapper import PolicyWrapper
    wrapper = PolicyWrapper(lambda observation, history: len(history))
    model = BrainScoreModel('policy', action_fn=wrapper.__call__ if bound_method else wrapper)
    assert model.process(EnvironmentStep(observation={})).action == 0
    model.reset()
    assert model.process(EnvironmentStep(observation={})).action == 0


def test_vision_adapter_reset_reaches_real_commitment():
    from brainscore_vision.compat.unified_adapter import VisionModelAdapter
    from brainscore_vision.model_helpers.brain_transformation import ModelCommitment
    from brainscore_vision.model_helpers.brain_transformation.behavior import (
        BehaviorArbiter, LabelBehavior, ProbabilitiesMapping, OddOneOut,
    )
    from brainscore_vision.model_helpers.brain_transformation.neural import LayerMappedModel
    from brainscore_vision.model_helpers.brain_transformation.temporal import TemporalAligned
    legacy = ModelCommitment.__new__(ModelCommitment)
    legacy.do_behavior = True
    legacy.layer_model = TemporalAligned(LayerMappedModel('toy', None, {'IT': 'L'}))
    label = LabelBehavior('toy', None)
    probabilities = ProbabilitiesMapping('toy', None, 'L')
    odd_one_out = OddOneOut('toy', None, 'L')
    label.current_task, label.choice_labels = 'label', ['cat', 'dog']
    probabilities.current_task = 'probabilities'
    probabilities.classifier._scaler = object()
    old_classifier = probabilities.classifier
    odd_one_out.similarity_measure = 'cosine'
    legacy.behavior_model = BehaviorArbiter({
        'label': label, 'probabilities': probabilities, 'odd_one_out': odd_one_out})
    legacy.behavior_model.current_executor = label
    adapter = VisionModelAdapter(legacy)
    adapter.start_recording('IT')
    adapter.reset()
    assert not legacy.do_behavior
    assert legacy.layer_model._time_bins is None
    assert legacy.layer_model._layer_model.recorded_regions == []
    assert legacy.behavior_model.current_executor is None
    assert label.current_task is None and label.choice_labels is None
    assert probabilities.current_task is None
    assert probabilities.classifier is not old_classifier
    assert probabilities.classifier._scaler is None
    assert odd_one_out.current_task is None and odd_one_out.similarity_measure == 'dot'


def test_pereira_declares_passages_before_extraction():
    from brainscore_language.benchmarks.pereira2018.unified import _Pereira2018ExperimentUnified
    from brainscore_core.text import contextualized_texts
    benchmark = _Pereira2018ExperimentUnified.__new__(_Pereira2018ExperimentUnified)
    benchmark.data = NeuroidAssembly(np.zeros((4, 1)), dims=['presentation', 'neuroid'], coords={
        'stimulus': ('presentation', list(_stimuli().sentence)),
        'stimulus_id': ('presentation', list(_stimuli().stimulus_id)),
        'passage_label': ('presentation', ['p1', 'p1', 'p2', 'p2']),
        'neuroid_id': ('neuroid', ['target']),
    })
    seen = []
    class Candidate:
        def start_recording(self, *args, **kwargs):
            pass
        def process(self, stimuli):
            seen.append(contextualized_texts(stimuli))
            return _Extractor(np.ones((4, 1)))(stimuli, ['L'])
    class ReachedMetric(Exception):
        pass
    def stop_at_metric(prediction, target):
        assert list(prediction.stimulus_id.values) == list(target.stimulus_id.values)
        raise ReachedMetric
    benchmark.metric = stop_at_metric
    with pytest.raises(ReachedMetric):
        benchmark(Candidate())
    assert seen == [['the cat', 'the cat the dog'], ['cat sat', 'cat sat dog ran']]


def test_context_expansion_preserves_stimulus_identifier(monkeypatch):
    from brainscore.model_helpers.text_wrapper import TextWrapper
    wrapper = TextWrapper.__new__(TextWrapper)
    wrapper._layer_aggregation = 'last_token'
    seen = []
    monkeypatch.setattr(wrapper, '_from_texts_cached',
                        lambda texts, layers, identifier: seen.append((texts, identifier)))
    monkeypatch.setattr(wrapper, '_attach_stimulus_set_meta', lambda output, stimuli: output)
    stimuli = _stimuli()
    wrapper._from_stimulus_set(stimuli, ['L'])
    stimuli['context_id'] = 'passage'
    wrapper._from_stimulus_set(stimuli, ['L'])
    stimuli['context_id'] = ['a', 'a', 'b', 'b']
    wrapper._from_stimulus_set(stimuli, ['L'])
    assert len({identifier for _, identifier in seen}) == 1
    assert len({tuple(texts) for texts, _ in seen}) == 3
    assert seen[0][0][1] == 'the dog'
    assert seen[1][0][1] == 'the cat the dog'
    wrapper._layer_aggregation = 'per_token'
    with pytest.raises(ValueError, match='per_token'):
        wrapper._from_stimulus_set(stimuli, ['L'])


def test_provider_reset_attempts_remaining_providers_after_failure():
    class Provider:
        def __init__(self, fail=False):
            self.fail, self.calls = fail, 0
        def reset(self):
            self.calls += 1
            if self.fail:
                raise ValueError('reset failed')
    first, second = Provider(True), Provider()
    model = BrainScoreModel('providers', action_fn=first, generation_fn=second,
                            preprocessors={'text': second})
    with pytest.raises(RuntimeError, match='not clean'):
        model.reset()
    assert first.calls == second.calls == 1
