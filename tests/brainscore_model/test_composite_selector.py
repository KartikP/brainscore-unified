"""Tests for the three layer-mapping types: standard, 'all' (whole-brain), and
the v1.5 CompositeSelector (units gathered across multiple layers as one region).
"""
import numpy as np
import xarray as xr

from brainscore import BrainScoreModel, CompositeSelector


def make_stub_preprocessor(return_value='default_output'):
    """A 'vision' preprocessor placeholder; the activations_model does the
    real extraction, so this is only here to register the modality."""
    def preprocessor(model, stimuli, *, recording_layer=None, **kwargs):
        return return_value
    return preprocessor


class StubStimulusSet:
    """Minimal stand-in for StimulusSet with column names."""

    def __init__(self, columns):
        self.columns = columns


class _MultiUnitActivations:
    """Activations stub returning ``units_per_layer`` neuroids per layer, each
    tagged with its 'layer'. Lets composite tests index within a layer."""

    def __init__(self, units_per_layer=4):
        self.units_per_layer = units_per_layer
        self.calls = []

    def __call__(self, stimuli, layers=None, **kwargs):
        layers = list(layers) if layers else ['default']
        self.calls.append(list(layers))
        k = self.units_per_layer
        layer_coord = [L for L in layers for _ in range(k)]
        n = len(layer_coord)
        return xr.DataArray(
            np.zeros((2, n)),
            dims=('presentation', 'neuroid'),
            coords={'layer': ('neuroid', np.array(layer_coord)),
                    'neuroid_id': ('neuroid', np.arange(n))},
        )


def _model(region_layer_map, units_per_layer=4):
    return BrainScoreModel(
        identifier='composite-test', model=None,
        region_layer_map=region_layer_map,
        preprocessors={'vision': make_stub_preprocessor()},
        activations_model=_MultiUnitActivations(units_per_layer),
    )


def _stim():
    return StubStimulusSet(columns=['image_file_name'])


class TestStandardMapping:
    def test_single_region_one_layer(self):
        m = _model({'IT': 'layer.10'})
        m.start_recording('IT')
        a = m.process(_stim())
        assert 'region' not in a.coords            # single-region keeps legacy shape
        assert a.sizes['neuroid'] == 4             # all units of layer.10


class TestAllMapping:
    def test_a_list_of_every_region_records_each(self):
        m = _model({'V4': 'layer.8', 'IT': 'layer.10'})
        m.start_recording(['V4', 'IT'])
        a = m.process(_stim())
        assert 'region' in a.coords
        assert set(a['region'].values.tolist()) == {'V4', 'IT'}
        assert a.sizes['neuroid'] == 8             # 4 per layer, both layers


class TestCompositeMapping:
    def test_recording_expands_composite_layers(self):
        sel = CompositeSelector(layers=(('layer.5', None), ('layer.8', None)))
        m = _model({'Vc': sel})
        m.start_recording('Vc')
        assert m._composite_recording is True
        assert m._recording_layers == ['layer.5', 'layer.8']

    def test_composite_all_units(self):
        sel = CompositeSelector(layers=(('layer.5', None), ('layer.8', None)))
        m = _model({'Vc': sel})
        m.start_recording('Vc')
        a = m.process(_stim())
        assert set(a['region'].values.tolist()) == {'Vc'}
        assert a.sizes['neuroid'] == 8
        assert set(a['layer'].values.tolist()) == {'layer.5', 'layer.8'}

    def test_composite_indexed_units(self):
        sel = CompositeSelector(layers=(('layer.5', (0, 1)), ('layer.8', (3,))))
        m = _model({'Vc': sel}, units_per_layer=4)
        m.start_recording('Vc')
        a = m.process(_stim())
        assert a.sizes['neuroid'] == 3
        layers = a['layer'].values.tolist()
        assert layers.count('layer.5') == 2
        assert layers.count('layer.8') == 1
        assert a['region'].values.tolist() == ['Vc', 'Vc', 'Vc']

    def test_composite_mixed_with_standard_region(self):
        m = _model({'IT': 'layer.10',
                    'Vc': CompositeSelector(layers=(('layer.5', (0,)), ('layer.8', (1,))))})
        m.start_recording(['IT', 'Vc'])
        a = m.process(_stim())
        regions = a['region'].values.tolist()
        assert set(regions) == {'IT', 'Vc'}
        assert regions.count('IT') == 4   # all units of layer.10
        assert regions.count('Vc') == 2   # one unit each from layer.5, layer.8


# -- each selector answers for its region; the model never checks its class ----

def _text_model(region_layer_map):
    from brainscore_core.supported_data_standards.brainio.assemblies import NeuroidAssembly

    class Units:
        """Four units per layer; unit ``i`` of layer ``L`` reads ``10 * L + i``."""
        identifier = backbone_id = 'four-units'

        def __call__(self, stimuli, layers, **kwargs):
            values = np.array([[10.0 * int(layer[-1]) + i for layer in layers for i in range(4)]
                               for _ in range(len(stimuli))])
            return NeuroidAssembly(values, dims=['presentation', 'neuroid'], coords={
                'stimulus_id': ('presentation', list(stimuli['stimulus_id'])),
                'layer': ('neuroid', [layer for layer in layers for _ in range(4)]),
                'neuroid_id': ('neuroid', [f'{layer}.{i}' for layer in layers for i in range(4)])})

    return BrainScoreModel('selector-answers', extractor={'text': Units()},
                           region_layer_map=region_layer_map)


def _record(model, region):
    from brainscore_core.streaming_helpers import neural_response
    from brainscore_core.supported_data_standards.brainio.stimuli import StimulusSet
    stimuli = StimulusSet({'stimulus_id': ['a', 'b'], 'sentence': ['x', 'y']})
    stimuli.identifier = 'selector-answers'
    return neural_response(model, stimuli, [f'neural:{region}'])


class TestSelectorAnswers:
    def test_one_layer_selector(self):
        from brainscore import LayerSelector
        selector = LayerSelector('layer.1')
        assert selector.layer_paths == ('layer.1',)
        assert selector.required_layers == ('layer.1',)
        assert not selector.gathers_units and not selector.keeps_unit_subset
        marker = object()
        assert selector.gather_units(marker) is marker

    def test_composite_selector_keeps_the_empty_subset_rule(self):
        selector = CompositeSelector(layers=(('layer.1', (2, 0)), ('layer.2', ()), ('layer.3', None)))
        assert selector.layer_paths == ('layer.1', 'layer.2', 'layer.3')
        assert selector.required_layers == ('layer.1', 'layer.3')   # an empty subset is not required
        assert selector.gathers_units and selector.keeps_unit_subset
        assert not CompositeSelector(layers=(('layer.1', None),)).keeps_unit_subset

    def test_composite_recording_keeps_layer_and_unit_order(self):
        selector = CompositeSelector(layers=(('layer.2', (3, 1)), ('layer.1', None)))
        recorded = _record(_text_model({'R': selector}), 'R')
        assert recorded['layer'].values.tolist() == ['layer.2'] * 2 + ['layer.1'] * 4
        assert recorded['unit_index'].values.tolist() == [3, 1, 0, 1, 2, 3]
        assert recorded.values[0].tolist() == [23.0, 21.0, 10.0, 11.0, 12.0, 13.0]
        assert recorded['region'].values.tolist() == ['R'] * 6

    def test_a_new_selector_kind_needs_no_model_change(self):
        from brainscore import UnitSelector

        class EvenUnits(UnitSelector):
            """Units 0 and 2 of two layers: a third selector kind."""
            gathers_units = True

            @property
            def layer_path(self):
                return 'layer.1'

            @property
            def layer_paths(self):
                return ('layer.1', 'layer.2')

            def gather_units(self, assembly):
                return assembly.isel(neuroid=np.flatnonzero(assembly['unit_index'].values % 2 == 0))

        recorded = _record(_text_model({'R': EvenUnits()}), 'R')
        assert recorded.values[0].tolist() == [10.0, 12.0, 20.0, 22.0]
        assert recorded['unit_index'].values.tolist() == [0, 2, 0, 2]
