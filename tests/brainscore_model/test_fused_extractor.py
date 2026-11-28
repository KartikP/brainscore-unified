"""A fused extractor: one forward pass over several input types, split by input."""
import numpy as np
import pytest

from brainscore.brainscore_model import BrainScoreModel
from brainscore.selection import LayerSelector
from brainscore_core.streaming_helpers import stimulus_session
from brainscore_core.supported_data_standards.brainio.assemblies import NeuroidAssembly
from brainscore_core.supported_data_standards.brainio.stimuli import StimulusSet

pytestmark = pytest.mark.unit


class Fused:
    """Sees the image and the sentence of each stimulus together; returns one step per input."""
    identifier = 'fused'

    def __init__(self):
        self.calls = []

    def __call__(self, stimuli, layers, **kwargs):
        self.calls.append(sorted(c for c in stimuli.columns if c != 'stimulus_id'))
        ids = list(stimuli['stimulus_id'])
        values = np.stack([[[1.0, 2.0], [3.0, 4.0]] for _ in ids])       # parts: image, text
        return NeuroidAssembly(values, dims=['presentation', 'part', 'neuroid'], coords={
            'stimulus_id': ('presentation', ids), 'part_name': ('part', ['image', 'text']),
            'part_index': ('part', [0, 1]),
            'neuroid_id': ('neuroid', ['n0', 'n1']), 'layer': ('neuroid', [layers[0]] * 2)})


def test_a_fused_extractor_declares_each_input_and_runs_once():
    fused = Fused()
    model = BrainScoreModel('fused', extractor={('vision', 'text'): fused},
                            region_layer_map={'A': 'layers.3'})
    assert {'text', 'vision:image'} <= model.in_channels
    both = StimulusSet({'stimulus_id': ['a', 'b'], 'image_path': ['a.png', 'b.png'],
                        'sentence': ['one', 'two']})
    both.identifier = 'fused-both'
    session = stimulus_session(both, ['neural:A'])
    model.participate(session)
    recorded = session.collect('neural:A')
    assert len(fused.calls) == 1 and fused.calls[0] == ['image_path', 'sentence']
    assert recorded.dims == ('presentation', 'part', 'neuroid')
    assert list(recorded['part_name'].values) == ['image', 'text']


def test_single_input_extractors_still_take_single_input_stimuli():
    fused, vision_calls = Fused(), []

    def vision(stimuli, layers, **kwargs):
        vision_calls.append(1)
        ids = list(stimuli['stimulus_id'])
        return NeuroidAssembly(np.ones((len(ids), 2)), dims=['presentation', 'neuroid'], coords={
            'stimulus_id': ('presentation', ids), 'neuroid_id': ('neuroid', ['v0', 'v1']),
            'layer': ('neuroid', [layers[0]] * 2)})
    model = BrainScoreModel('mixed', extractor={'vision': vision, ('vision', 'text'): fused},
                            region_layer_map={'A': LayerSelector('layers.3', modality='vision')})
    images = StimulusSet({'stimulus_id': ['a'], 'image_path': ['a.png']})
    images.identifier = 'images-only'
    session = stimulus_session(images, ['neural:A'])
    model.participate(session)
    assert session.collect('neural:A').dims == ('presentation', 'neuroid')


@pytest.mark.parametrize('second, channel', [('text', 'text'), ('audio', 'audio')])
def test_a_fused_video_extractor_declares_the_video_channel(second, channel):
    fused = Fused()
    model = BrainScoreModel('fused-video', extractor={('video', second): fused},
                            region_layer_map={'A': 'layers.3'})
    assert {channel, 'vision:video'} <= model.in_channels
    assert 'vision:image' not in model.in_channels


def test_a_fused_video_extractor_gets_the_clip_column():
    fused = Fused()
    model = BrainScoreModel('fused-clip', extractor={('video', 'text'): fused},
                            region_layer_map={'A': 'layers.3'})
    both = StimulusSet({'stimulus_id': ['a'], 'video_path': ['a.mp4'], 'sentence': ['one']})
    both.identifier = 'fused-clip-both'
    session = stimulus_session(both, ['neural:A'])
    model.participate(session)
    assert fused.calls == [['sentence', 'video_path']]


def test_required_channels_must_match_the_declared_visual_form():
    def extractor(stimuli, layers, **kwargs):
        raise AssertionError('not called')
    for key, required in (('vision', 'vision:video'), ('video', 'vision:image')):
        with pytest.raises(ValueError, match='vision:'):
            BrainScoreModel('m', extractor={key: extractor}, required_channels={required},
                            region_layer_map={'A': 'layers.3'})


class Layered:
    """A fused extractor for the clip and the sound whose layers are independent: one value row per layer."""
    identifier = 'layered'
    independent_layers = True

    def __init__(self):
        self.calls = []

    def layer_names(self):
        return ['a', 'b', 'c', 'd']

    def __call__(self, stimuli, layers, **kwargs):
        self.calls.append(list(layers))
        ids = list(stimuli['stimulus_id'])
        values = np.concatenate([np.full((len(ids), 2), 1.0 + 'abcd'.index(layer)) + np.arange(len(ids))[:, None]
                                 for layer in layers], axis=1)
        return NeuroidAssembly(values, dims=['presentation', 'neuroid'], coords={
            'stimulus_id': ('presentation', ids),
            'neuroid_id': ('neuroid', [f'{layer}.{i}' for layer in layers for i in range(2)]),
            'layer': ('neuroid', [layer for layer in layers for _ in range(2)])})


def _clip_and_sound(plan, monkeypatch):
    monkeypatch.setenv('BRAINSCORE_REGIONS_PER_PASS', plan)
    layered = Layered()
    model = BrainScoreModel('layered', extractor={('video', 'audio'): layered}, region_layer_map={
        'V1': LayerSelector('a', modality=('video', 'audio')), 'IT': LayerSelector('b', modality=('video', 'audio')),
        'A1': LayerSelector('c', modality='audio')})
    both = StimulusSet({'stimulus_id': ['x', 'y'], 'video_path': ['x.mp4', 'y.mp4'],
                        'audio_path': ['x.wav', 'y.wav']})
    both.identifier = 'clip-and-sound'
    session = stimulus_session(both, ['neural:V1', 'neural:IT', 'neural:A1'])
    model.participate(session)
    return layered, {region: session.collect(f'neural:{region}') for region in ('V1', 'IT', 'A1')}


def test_a_fused_extractor_with_independent_layers_runs_one_pass_for_all_regions(monkeypatch):
    shared, recorded = _clip_and_sound('all', monkeypatch)
    alone, expected = _clip_and_sound('one', monkeypatch)
    assert shared.calls == [['a', 'b', 'c']] and len(alone.calls) == 3
    for region, assembly in expected.items():   # each region gets the layers and values of its own pass
        assert list(recorded[region]['layer'].values) == list(assembly['layer'].values)
        np.testing.assert_array_equal(recorded[region].values, assembly.values)
