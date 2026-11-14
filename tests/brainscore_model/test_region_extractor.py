"""The extractor of a region comes from the wiring: the one whose layers include it.

A region names its extractor only when its layer name is in extractors of two
inputs, or in an extractor that cannot list its layers.
"""
import numpy as np
import pytest

from brainscore import BrainScoreModel, LayerSelector
from brainscore_core.compatibility import CompatibilityError, check_compatibility
from brainscore_core.streaming_helpers import stimulus_session
from brainscore_core.supported_data_standards.brainio.assemblies import NeuroidAssembly
from brainscore_core.supported_data_standards.brainio.stimuli import StimulusSet

pytestmark = pytest.mark.unit

rng = lambda prefix, n: [f'{prefix}.{i}' for i in range(n)]


class Tower:
    """An extractor that lists its layers and answers one value per stimulus."""

    def __init__(self, layers, value=1.0):
        self._layers, self.value, self.calls = list(layers), value, []

    def layer_names(self):
        return list(self._layers)

    def __call__(self, stimuli, layers, **kwargs):
        ids = list(stimuli['stimulus_id'])
        self.calls.append(list(layers))
        return NeuroidAssembly(
            np.full((len(ids), 2 * len(layers)), self.value), dims=['presentation', 'neuroid'],
            coords={'stimulus_id': ('presentation', ids),
                    'neuroid_id': ('neuroid', [f'{l}.{u}' for l in layers for u in range(2)]),
                    'layer': ('neuroid', [l for l in layers for _ in range(2)])})


class Benchmark:
    identifier = 'stand-in'
    trial_memory = 'none'

    def __init__(self, inputs, outputs):
        self.required_input_channels = frozenset(inputs)
        self.requested_output_channels = frozenset(outputs)


# Fakes with the wiring of the registrations (towers list the layer names of the real networks).
QWEN_VISION, QWEN_TEXT = ['merger', *rng('blocks', 32)], ['norm', *rng('layers', 36)]
WAV2VEC2 = [*rng('feature_extractor.conv_layers', 7), *rng('encoder.layers', 12)]
VJEPA1 = ['backbone', *rng('backbone.blocks', 24)]
CLIP_VISION, CLIP_TEXT = rng('encoder.layers', 12), rng('encoder.layers', 12)
WIRINGS = {
    'qwen2.5-vl-3b': ({'vision': QWEN_VISION, 'text': QWEN_TEXT},
                      {'V1': ('blocks.2', 'vision'), 'IT': ('blocks.28', 'vision'),
                       'language_system': ('layers.28', 'text')}),
    'qwen3.6-27b': ({'vision': QWEN_VISION, 'video': QWEN_VISION, 'text': QWEN_TEXT},
                    {'IT': ('blocks.23', 'vision'), 'language_system': ('layers.30', 'text')}),
    'blip2-opt-2.7b': ({'vision': rng('encoder.layers', 39), 'text': rng('layers', 32)},
                       {'V1': ('encoder.layers.4', 'vision'), 'IT': ('encoder.layers.34', 'vision'),
                        'language_system': ('layers.28', 'text')}),
    'qwen2.5-vl-wav2vec2': ({'vision': QWEN_VISION, 'audio': WAV2VEC2},
                            {'V4': ('blocks.14', 'vision'), 'A1': ('encoder.layers.6', 'audio')}),
    'vjepa1-wav2vec2': ({'video': VJEPA1, 'audio': WAV2VEC2},
                        {'IT': ('backbone.blocks.16', 'video'), 'A1': ('encoder.layers.6', 'audio')}),
}


def _model(wiring, named):
    towers, regions = WIRINGS[wiring]
    extractors = {key: Tower(layers) for key, layers in towers.items()}
    return BrainScoreModel(wiring, region_layer_map={
        region: LayerSelector(layer, modality=modality) if named else layer
        for region, (layer, modality) in regions.items()}, extractor=extractors)


@pytest.mark.parametrize('wiring', sorted(WIRINGS))
def test_a_region_without_a_named_extractor_is_declared_as_a_named_one(wiring):
    named, derived = _model(wiring, named=True), _model(wiring, named=False)
    assert derived.output_inputs == named.output_inputs
    assert (derived.in_channels, derived.out_channels) == (named.in_channels, named.out_channels)
    for region in WIRINGS[wiring][1]:
        image = Benchmark({'vision:image'}, {f'neural:{region}'})
        try:
            expected = check_compatibility(named, image).kind
        except CompatibilityError:
            expected = 'na'
        try:
            got = check_compatibility(derived, image).kind
        except CompatibilityError:
            got = 'na'
        assert got == expected


def test_a_layer_name_in_two_inputs_needs_a_named_extractor():
    towers = {'vision': Tower(CLIP_VISION), 'text': Tower(CLIP_TEXT)}
    with pytest.raises(ValueError, match=r"\['IT'\].*names its extractor"):
        BrainScoreModel('clip-like', region_layer_map={'IT': 'encoder.layers.10'}, extractor=towers)
    model = BrainScoreModel('clip-like', extractor=towers, region_layer_map={
        'IT': LayerSelector('encoder.layers.10', modality='vision'),
        'language_system': LayerSelector('encoder.layers.10', modality='text')})
    assert model.output_inputs == {'neural:IT': frozenset({'vision:image'}),
                                   'neural:language_system': frozenset({'text'})}


def test_an_extractor_that_cannot_list_its_layers_needs_a_named_extractor():
    def unlisted(stimuli, layers):
        raise AssertionError('not run')
    with pytest.raises(ValueError, match='names its extractor'):
        BrainScoreModel('m', region_layer_map={'A1': 'encoder.layers.6'},
                        extractor={'vision': unlisted, 'audio': Tower(WAV2VEC2)})


def test_the_run_records_a_region_from_the_extractor_that_has_its_layer():
    vision, audio = Tower(QWEN_VISION, 1.0), Tower(WAV2VEC2, 2.0)
    model = BrainScoreModel('picture-and-sound', extractor={'vision': vision, 'audio': audio},
                            region_layer_map={'V4': 'blocks.14', 'A1': 'encoder.layers.6'})
    stimuli = StimulusSet({'stimulus_id': ['s0', 's1'], 'image_file_name': ['a.png', 'b.png']})
    stimuli.identifier = 'pictures'
    session = stimulus_session(stimuli, ['neural:V4'])
    model.participate(session)
    recorded = session.collect('neural:V4')
    assert np.all(recorded.values == 1.0)
    assert vision.calls == [['blocks.14']] and audio.calls == []
