"""Regions of one extractor share one call, and each channel equals its single-channel output.

An extractor that declares ``independent_layers = True`` runs once per session
(once per row when streamed) for every requested region it records. Each
channel must still equal, bit for bit, the payload of a session that requests
that channel alone: values, dims, coordinates and their order, the neuroid
index, attributes and event meta. Tiny real networks on CPU; no weights.
"""
import numpy as np
import pytest
import torch

from brainscore import BrainScoreModel, CompositeSelector, LayerSelector
from brainscore_core import TaskContext
from brainscore_core.streaming_helpers import stimulus_session
from brainscore_core.supported_data_standards.brainio.assemblies import NeuroidAssembly
from brainscore_core.supported_data_standards.brainio.stimuli import StimulusSet


def assert_same_output(actual, expected):
    assert type(actual) is type(expected)
    assert actual.dims == expected.dims and actual.dtype == expected.dtype
    np.testing.assert_array_equal(actual.values, expected.values)
    assert list(actual.coords) == list(expected.coords)
    for name in expected.coords:
        assert actual[name].dims == expected[name].dims, name
        assert actual[name].dtype == expected[name].dtype, name
    assert set(actual.indexes) == set(expected.indexes)
    for dim, index in expected.indexes.items():
        assert list(actual.indexes[dim].names) == list(index.names), dim
        assert actual.indexes[dim].equals(index), dim
        for mine, theirs in zip(getattr(actual.indexes[dim], 'levels', ()), getattr(index, 'levels', ())):
            assert mine.equals(theirs), dim
    assert actual.attrs.keys() == expected.attrs.keys()
    assert actual.identical(expected)


def events(session, channel):
    return [(e.t_ms, e.meta, e.payload) for e in session.emitted if e.channel == channel]


def assert_each_channel_alone(make, stimuli, regions, streaming, task=None):
    """One session for ``regions`` equals one session per region; returns the calls of each."""
    model, calls = make()
    session = stimulus_session(stimuli, [f'neural:{r}' for r in regions], task=task, streaming=streaming)
    model.participate(session)
    joint_calls = list(calls)
    for region in regions:
        alone_model, _ = make()
        alone = stimulus_session(stimuli, [f'neural:{region}'], task=task, streaming=streaming)
        alone_model.participate(alone)
        mine, theirs = events(session, f'neural:{region}'), events(alone, f'neural:{region}')
        assert len(mine) == len(theirs)
        for (t_a, meta_a, a), (t_b, meta_b, b) in zip(mine, theirs):
            assert t_a == t_b and meta_a == meta_b
            assert_same_output(a, b)
    return joint_calls


def counting(wrapper, name):
    """Count the wrapper's computing calls (``name`` is its uncached extraction method)."""
    calls, inner = [], getattr(wrapper, name)

    def count(*args, **kwargs):
        calls.append(kwargs.get('layers', args[1] if len(args) > 1 else None))
        return inner(*args, **kwargs)
    setattr(wrapper, name, count)
    return calls


# --- TextWrapper on a tiny GPT-2 ---------------------------------------------

TEXT_REGIONS = {
    'V1': LayerSelector('h.0', modality='text'),
    'V4': LayerSelector('h.1', modality='text'),
    'IT': LayerSelector('ln_f', modality='text'),
    'SAME': LayerSelector('h.1', modality='text'),
    'INPUT': LayerSelector('h.1.mlp.c_proj:input', modality='text'),
    'MIX': CompositeSelector((('h.0', (1, 3)), ('ln_f', None)), modality='text'),
}


def text_model(aggregation):
    tokenizers = pytest.importorskip('tokenizers')
    from transformers import GPT2Config, GPT2Model, PreTrainedTokenizerFast
    from brainscore.model_helpers.text_wrapper import TextWrapper
    vocab = {'[PAD]': 0, '[UNK]': 1, 'the': 2, 'cat': 3, 'dog': 4, 'sat': 5, 'ran': 6}
    tokenizer = tokenizers.Tokenizer(tokenizers.models.WordLevel(vocab=vocab, unk_token='[UNK]'))
    tokenizer.pre_tokenizer = tokenizers.pre_tokenizers.Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=tokenizer, unk_token='[UNK]', pad_token='[PAD]',
                                        model_max_length=64)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(17)
        net = GPT2Model(GPT2Config(vocab_size=7, n_positions=64, n_embd=8, n_layer=2, n_head=2)).eval()
    wrapper = TextWrapper(net, tokenizer, identifier=f'tiny-{aggregation}', layer_aggregation=aggregation,
                          max_length=64, batch_size=2, device='cpu')
    calls = counting(wrapper, '_from_texts')
    return BrainScoreModel('tiny-text', extractor={'text': wrapper}, region_layer_map=TEXT_REGIONS), calls


def text_stimuli():
    stimuli = StimulusSet({'stimulus_id': ['a', 'b', 'c'],
                           'sentence': ['the cat sat', 'the dog ran the', 'cat']})
    stimuli.identifier = 'one-call-text'
    return stimuli


@pytest.mark.parametrize('streaming', [False, True], ids=['table', 'stream'])
@pytest.mark.parametrize('aggregation', ['last_token', 'per_token'])
def test_text_regions_share_one_call_and_equal_each_region_alone(aggregation, streaming):
    make = lambda: text_model(aggregation)
    calls = assert_each_channel_alone(make, text_stimuli(), list(TEXT_REGIONS), streaming)
    rows = 3 if streaming else 1
    assert len(calls) == rows                       # one call per table, or per row
    assert set(calls[0]) == {'h.0', 'h.1', 'ln_f', 'h.1.mlp.c_proj:input'}


def test_a_passive_task_still_shares_the_call():
    calls = assert_each_channel_alone(lambda: text_model('last_token'), text_stimuli(), ['V1', 'IT', 'MIX'],
                                      False, task=TaskContext(task_type='passive'))
    assert len(calls) == 1


def test_a_composite_region_in_another_layer_order_gets_its_own_call():
    """A request whose layers the shared output does not list in that order is not cut from it."""
    regions = dict(TEXT_REGIONS, BACK=CompositeSelector((('ln_f', (0,)), ('h.0', None)), modality='text'))

    def make():
        model, calls = text_model('last_token')
        model = BrainScoreModel('tiny-text', extractor=dict(model._extractors), region_layer_map=regions)
        return model, calls
    calls = assert_each_channel_alone(make, text_stimuli(), ['V1', 'IT', 'BACK'], False)
    assert len(calls) == 2 and list(calls[1]) == ['ln_f', 'h.0']


# --- VLMVisionWrapper and VLMVideoWrapper on tiny towers -------------------------

class _Images:
    def __call__(self, images=None, return_tensors='pt'):
        return {'pixel_values': torch.tensor(np.stack(
            [np.asarray(image, dtype=np.float32).transpose(2, 0, 1) / 255 for image in images]))}


@pytest.fixture
def images(tmp_path):
    from PIL import Image
    rng = np.random.default_rng(3)
    paths = []
    for index in range(3):
        path = tmp_path / f'image{index}.png'
        Image.fromarray(rng.integers(0, 255, (8, 8, 3), dtype=np.uint8)).save(path)
        paths.append(str(path))
    stimuli = StimulusSet({'stimulus_id': ['i0', 'i1', 'i2'], 'image_path': paths})
    stimuli.identifier = 'one-call-images'
    stimuli.stimulus_paths = dict(zip(stimuli['stimulus_id'], paths))
    return stimuli


def vlm_vision_model():
    from brainscore.model_helpers.vlm_vision_wrapper import VLMVisionWrapper
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(5)
        net = torch.nn.Sequential(torch.nn.Conv2d(3, 4, 3), torch.nn.ReLU(), torch.nn.Conv2d(4, 2, 3),
                                  torch.nn.Flatten())
    wrapper = VLMVisionWrapper(net, _Images(), identifier='tiny-vlm', batch_size=2, device='cpu')
    calls = counting(wrapper, '_from_paths')
    regions = {'V1': LayerSelector('0', modality='vision'), 'V4': LayerSelector('2', modality='vision'),
               'IT': LayerSelector('3', modality='vision'),
               'MIX': CompositeSelector((('0', (0, 5)), ('3', None)), modality='vision')}
    return BrainScoreModel('tiny-vlm', extractor={'vision': wrapper}, region_layer_map=regions), calls


@pytest.mark.parametrize('streaming', [False, True], ids=['table', 'stream'])
def test_vlm_vision_regions_share_one_call(images, streaming):
    calls = assert_each_channel_alone(vlm_vision_model, images, ['V1', 'V4', 'IT', 'MIX'], streaming)
    assert len(calls) == (3 if streaming else 1)


@pytest.mark.parametrize('streaming', [False, True], ids=['table', 'stream'])
def test_vlm_video_regions_share_one_call(tmp_path, streaming):
    cv2 = pytest.importorskip('cv2')
    from brainscore.model_helpers.vlm_video_wrapper import VLMVideoWrapper
    from tests.test_vlm_video_wrapper import Processor, Tower
    paths = []
    for index, moving in enumerate((True, False)):
        path = tmp_path / f'clip{index}.avi'
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'MJPG'), 4.0, (16, 16))
        for step in range(8):
            frame = np.zeros((16, 16, 3), np.uint8)
            frame[:, (2 * step if moving else 0):(2 * step if moving else 0) + 4] = 255
            writer.write(frame)
        writer.release()
        paths.append(str(path))
    clips = StimulusSet({'stimulus_id': ['c0', 'c1'], 'video_path': paths})
    regions = {'V1': LayerSelector('blocks.0.0', modality='video'),
               'IT': LayerSelector('blocks.0.2', modality='video'),
               'MIX': CompositeSelector((('blocks.0.0', (1,)), ('blocks.0.2', None)), modality='video')}

    def make():
        calls = []
        wrapper = VLMVideoWrapper(Tower(), Processor(), identifier='tiny-vlm-video')
        inner = wrapper.__call__
        class Counted:          # counts the calls; declares what the wrapper declares
            independent_layers = VLMVideoWrapper.independent_layers
            identifier = backbone_id = 'tiny-vlm-video'
            def __call__(self, stimuli, layers, **kwargs):
                calls.append(list(layers))
                return inner(stimuli, layers, **kwargs)
        return BrainScoreModel('tiny-vlm-video', extractor={'video': Counted()}, region_layer_map=regions), calls
    calls = assert_each_channel_alone(make, clips, ['V1', 'IT', 'MIX'], streaming)
    assert len(calls) == (2 if streaming else 1)


def test_the_video_wrapper_does_not_declare_independent_layers():
    """VideoWrapper numbers ``neuroid_num`` over all layers of one call, so it keeps one call per region."""
    from brainscore.model_helpers.video_wrapper import VideoWrapper
    from brainscore.model_helpers.audio_wrapper import AudioWrapper
    assert getattr(VideoWrapper, 'independent_layers', False) is False
    assert getattr(AudioWrapper, 'independent_layers', False) is False


def test_the_declaration_keeps_the_activation_cache_key():
    """The class source is part of the cache key; the declaration sits outside it."""
    import inspect
    from brainscore.model_helpers.text_wrapper import TextWrapper
    from brainscore.model_helpers.vlm_vision_wrapper import VLMVisionWrapper
    for wrapper in (TextWrapper, VLMVisionWrapper):
        assert wrapper.independent_layers is True
        assert 'independent_layers' not in inspect.getsource(wrapper)


# --- the rules of issue #80, with a declared extractor -------------------------

IDS = ['a', 'b']


class Declared:
    """A declared extractor: two units per layer, values from the layer and the stimulus."""

    independent_layers = True
    identifier = backbone_id = 'declared'

    def __init__(self, calls, region=None):
        self.calls, self.region = calls, region

    def __call__(self, stimuli, layers, **kwargs):
        self.calls.append(list(layers))
        ids = list(stimuli['stimulus_id'])
        values = np.concatenate([np.full((len(ids), 2), float(sum(map(ord, layer))), dtype=np.float32)
                                 + np.arange(2, dtype=np.float32) for layer in layers], axis=1)
        output = NeuroidAssembly(values, dims=['presentation', 'neuroid'], coords={
            'stimulus_id': ('presentation', ids),
            'neuroid_id': ('neuroid', [f'{layer}.{u}' for layer in layers for u in range(2)]),
            'layer': ('neuroid', [layer for layer in layers for _ in range(2)])})
        if self.region:
            output = output.assign_coords(region=('neuroid', [self.region] * output.sizes['neuroid']))
        return output


VISUAL = {'V1': LayerSelector('early', modality='vision'), 'IT': LayerSelector('late', modality='vision')}


def images_table():
    return StimulusSet({'stimulus_id': IDS, 'image_path': ['a.png', 'b.png']})


def test_an_extractor_owned_region_coordinate_stays_as_alone():
    def make():
        calls = []
        return BrainScoreModel('owned', extractor={'vision': Declared(calls, region='source-owned')},
                               region_layer_map=VISUAL), calls
    calls = assert_each_channel_alone(make, images_table(), ['V1', 'IT'], False)
    assert calls == [['early', 'late']]


def test_a_capability_that_changes_each_answer_still_runs_per_channel():
    from brainscore.capabilities import Capability, register_capability
    from brainscore.capabilities.registry import capability_registry
    from brainscore.capabilities.neural import NeuralEncodingCapability
    answered = []

    class Center(Capability):
        identifier = 'test-one-call-center'
        order = 1
        def enabled_for(self, model): return model.identifier == 'center-declared'
        def handles(self, model, event, **kwargs): return isinstance(event, StimulusSet)
        def process(self, model, event, **kwargs):
            answered.append(list(model._recorder.recording_regions))
            result = NeuralEncodingCapability().process(model, event, **kwargs)
            return result - result.mean('neuroid')
    register_capability(Center())
    try:
        def make():
            calls = []
            return BrainScoreModel('center-declared', extractor={'vision': Declared(calls)},
                                   region_layer_map=VISUAL), calls
        calls = assert_each_channel_alone(make, images_table(), ['V1', 'IT'], False)
    finally:
        capability_registry.pop(Center.identifier)
    assert calls == [['early', 'late']]
    assert answered[:2] == [['V1'], ['IT']]           # the capability ran once per channel


def test_two_extractors_take_one_call_each():
    """Sharing follows the extractor that dispatch selects, never the modality alone."""
    vision_calls, audio_calls = [], []
    regions = {**VISUAL, 'A1': LayerSelector('early', modality='audio'),
               'A2': LayerSelector('late', modality='audio')}
    model = BrainScoreModel('two', extractor={'vision': Declared(vision_calls), 'audio': Declared(audio_calls)},
                            region_layer_map=regions)
    table = StimulusSet({'stimulus_id': IDS, 'image_path': ['a.png', 'b.png'], 'audio_path': ['a.wav', 'b.wav']})
    session = stimulus_session(table, [f'neural:{r}' for r in regions], task=TaskContext(task_type='passive'))
    model.participate(session)
    assert vision_calls == [['early', 'late']] and audio_calls == [['early', 'late']]


def test_an_undeclared_extractor_keeps_one_call_per_region():
    calls = []

    def plain(stimuli, layers):
        return Declared(calls)(stimuli, layers)
    model = BrainScoreModel('plain', extractor={'vision': plain}, region_layer_map=VISUAL)
    model.participate(stimulus_session(images_table(), ['neural:V1', 'neural:IT']))
    assert calls == [['early'], ['late']]
