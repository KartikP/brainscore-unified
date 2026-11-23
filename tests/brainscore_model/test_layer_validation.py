"""Layer paths are checked against the extractor's declared layer names before any forward pass."""
import pytest

from brainscore.brainscore_model import BrainScoreModel
from brainscore.selection import CompositeSelector, LayerSelector


class Declared:
    """An extractor that declares its layer namespace; it must never be called here."""

    def __init__(self, names):
        self.names = names
        self.identifier = 'declared'

    def layer_names(self):
        return list(self.names)

    def __call__(self, stimuli, layers):
        raise AssertionError('validation must not run a forward pass')


VISION = Declared(['conv1', 'layer1', 'layer2', 'layer4', 'logits'])
TEXT = Declared(['encoder.layers.0', 'encoder.layers.1'])


def _model(region_layer_map, **kwargs):
    return BrainScoreModel('toy', extractor={'vision': VISION}, region_layer_map=region_layer_map, **kwargs)


def test_typo_in_region_layer_map_fails_at_construction_with_close_matches():
    with pytest.raises(ValueError, match=r"region_layer_map\['IT'\].*'layr4'.*Close matches: \['layer4'"):
        _model({'IT': 'layr4'})


def test_region_layer_map_setters_are_validated():
    model = _model({'IT': 'layer4'})
    with pytest.raises(ValueError, match='layr2'):
        model.region_layer_map = {'V1': 'layr2'}
    assert dict(model.region_layer_map) == {'IT': 'layer4'}




def test_composite_selector_layers_are_each_checked():
    with pytest.raises(ValueError, match="'layr2'"):
        _model({'IT': CompositeSelector(layers=(('layer4', None), ('layr2', (0,))))})


def test_routed_region_is_checked_against_its_own_tower():
    extractor = {'vision': VISION, 'text': TEXT}
    model = BrainScoreModel('two', extractor=extractor, region_layer_map={
        'IT': LayerSelector('layer4', modality='vision'),
        'language': LayerSelector('encoder.layers.1', modality='text')})
    assert model.region_layer_map['language'].modality == 'text'
    with pytest.raises(ValueError, match="'text' extractor"):
        BrainScoreModel('two', extractor=extractor,
                        region_layer_map={'language': LayerSelector('layer4', modality='text')})
    with pytest.raises(ValueError, match="'vision' extractor"):
        BrainScoreModel('two', extractor=extractor,
                        region_layer_map={'IT': LayerSelector('encoder.layers.1', modality='vision')})


