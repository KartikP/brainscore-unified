import pytest
import warnings

from brainscore import BrainScoreModel, UnitSelector, LayerSelector
from brainscore_core.contract import TaskContext


# -- Helpers ------------------------------------------------------------------

def make_stub_extractor(return_value='default_output'):
    """Create a stub extractor callable that records its call args."""
    def extractor(stimuli, layers=None, **kwargs):
        extractor.call_args = {
            'stimuli': stimuli,
            'layers': layers,
        }
        return return_value
    extractor.call_args = None
    return extractor


def _noop_extractor(stimuli, layers=None, **kwargs):
    return None


class StubActivationsModel:
    """Stub extractor that records calls and returns a fixed value."""

    def __init__(self, return_value='activations_output'):
        self._return_value = return_value
        self.call_args = None

    def __call__(self, stimuli, layers=None, **kwargs):
        self.call_args = {
            'stimuli': stimuli,
            'layers': layers,
        }
        return self._return_value


class StubStimulusSet:
    """Minimal stand-in for StimulusSet with column names."""

    def __init__(self, columns):
        self.columns = columns


# -- TaskContext tests --------------------------------------------------------

class TestTaskContext:

    def test_defaults(self):
        ctx = TaskContext(task_type='odd_one_out')
        assert ctx.label_set is None
        assert ctx.fitting_stimuli is None
        assert ctx.instruction is None
        assert ctx.metadata == {}

    def test_metadata_default_is_independent(self):
        ctx1 = TaskContext(task_type='a')
        ctx2 = TaskContext(task_type='b')
        ctx1.metadata['key'] = 'value'
        assert 'key' not in ctx2.metadata


# -- BrainScoreModel construction tests ---------------------------------------

class TestBrainScoreModelConstruction:

    def test_basic_construction(self):
        m = BrainScoreModel(
            identifier='resnet-50',
            model='fake_model',
            region_layer_map={'V1': 'layer1', 'IT': 'layer4'},
            extractor={'vision': _noop_extractor},
        )
        assert m.identifier == 'resnet-50'
        assert m.region_layer_map == {'V1': 'layer1', 'IT': 'layer4'}
        assert m.in_channels == {'vision:image', 'control'}

    @staticmethod
    def _model(region_layer_map=None):
        return BrainScoreModel(
            identifier='test',
            model=None,
            region_layer_map=region_layer_map or {'V1': 'layer1'},
            extractor={'vision': _noop_extractor},
        )

    def test_region_layer_map_rejects_mutation(self):
        """Editing the map in place must raise, not quietly do nothing.

        This previously returned a fresh dict, so an in-place edit changed a
        throwaway copy: no error, no effect. A layer sweep written that way
        produced an identical score for every layer, because every run read the
        layer the model was actually built with.
        """
        m = self._model()
        with pytest.raises(TypeError):
            m.region_layer_map['V1'] = 'hacked'
        assert m.region_layer_map['V1'] == 'layer1'

    def test_region_layer_map_is_still_readable_like_a_dict(self):
        m = self._model({'V1': 'layer1', 'IT': 'layer4'})
        assert m.region_layer_map['IT'] == 'layer4'
        assert set(m.region_layer_map) == {'V1', 'IT'}
        assert dict(m.region_layer_map) == {'V1': 'layer1', 'IT': 'layer4'}
        assert 'V1' in m.region_layer_map

    def test_assigning_a_whole_map_works(self):
        m = self._model()
        m.region_layer_map = {'IT': 'layer4'}
        assert dict(m.region_layer_map) == {'IT': 'layer4'}

    def test_setter_keeps_selectors_and_layer_paths_in_step(self):
        """The two derived views must not drift — the reason direct edits are
        refused in the first place."""
        m = self._model()
        m.region_layer_map = {'IT': 'layer4'}
        assert m.region_layer_map['IT'] == 'layer4'
        assert m.region_layer_selectors['IT'].layer_path == 'layer4'
        assert set(m.region_layer_map) == set(m.region_layer_selectors)

    def test_recording_follows_a_reassigned_layer(self):
        """The end the wart broke: changing the map must change what is read."""
        extractor = StubActivationsModel()
        m = BrainScoreModel('test', region_layer_map={'IT': 'layer1'},
                            extractor={'vision': extractor})
        m.region_layer_map = {'IT': 'layer9'}
        m.start_recording('IT')
        m.process(StubStimulusSet(['image_file_name']))
        assert extractor.call_args['layers'] == ['layer9']

    def test_multiple_modalities(self):
        m = BrainScoreModel(
            identifier='vlm',
            model=None,
            region_layer_map={},
            extractor={
                'vision': _noop_extractor,
                'text': lambda stimuli, layers=None, **kw: None,
            },
        )
        assert m.in_channels == {'vision:image', 'text', 'control'}

    def test_with_extractor(self):
        act_model = StubActivationsModel()
        m = BrainScoreModel(
            identifier='resnet',
            model=None,
            region_layer_map={'IT': 'layer4'},
            extractor={'vision': act_model},
        )
        assert m.extractor['vision'] is act_model

    def test_visual_degrees_default(self):
        m = BrainScoreModel(
            identifier='test',
            model=None,
            region_layer_map={},
            extractor={'vision': _noop_extractor},
        )
        assert m.visual_degrees() == 8

    def test_visual_degrees_custom(self):
        m = BrainScoreModel(
            identifier='test',
            model=None,
            region_layer_map={},
            extractor={'vision': _noop_extractor},
            visual_degrees=12,
        )
        assert m.visual_degrees() == 12


# -- region_layer_map: Union[str, UnitSelector] forward-compat ----------------

class TestRegionLayerMapSelectors:
    """The value type on ``region_layer_map`` is ``Union[str, UnitSelector]``.

    Bare strings are auto-promoted to ``LayerSelector(name=str)``. Future
    selector subclasses (multi-layer, indexed) are additive."""

    def test_bare_string_is_promoted_to_layer_selector(self):
        m = BrainScoreModel(
            identifier='m', model=None,
            region_layer_map={'IT': 'encoder.layers.10'},
            extractor={'vision': _noop_extractor},
        )
        sel = m.region_layer_selectors['IT']
        assert isinstance(sel, LayerSelector)
        assert sel.name == 'encoder.layers.10'
        assert sel.layer_path == 'encoder.layers.10'

    def test_explicit_layer_selector_is_accepted(self):
        m = BrainScoreModel(
            identifier='m', model=None,
            region_layer_map={'IT': LayerSelector(name='encoder.layers.10')},
            extractor={'vision': _noop_extractor},
        )
        assert m.region_layer_map == {'IT': 'encoder.layers.10'}
        assert m.region_layer_selectors['IT'] == LayerSelector('encoder.layers.10')

    def test_string_and_selector_are_equivalent(self):
        a = BrainScoreModel(
            identifier='a', model=None,
            region_layer_map={'IT': 'x.y.z'},
            extractor={'vision': _noop_extractor},
        )
        b = BrainScoreModel(
            identifier='b', model=None,
            region_layer_map={'IT': LayerSelector('x.y.z')},
            extractor={'vision': _noop_extractor},
        )
        assert a.region_layer_map == b.region_layer_map
        assert a.region_layer_selectors == b.region_layer_selectors

    def test_invalid_value_type_raises(self):
        with pytest.raises(TypeError, match='str or UnitSelector'):
            BrainScoreModel(
                identifier='m', model=None,
                region_layer_map={'IT': 123},  # noqa
                extractor={'vision': _noop_extractor},
            )

    def test_region_layer_selectors_returns_copy(self):
        m = BrainScoreModel(
            identifier='m', model=None,
            region_layer_map={'IT': 'layer.10'},
            extractor={'vision': _noop_extractor},
        )
        view = m.region_layer_selectors
        view['IT'] = LayerSelector('hacked')
        assert m.region_layer_selectors['IT'].name == 'layer.10'

    def test_unit_selector_is_abstract(self):
        with pytest.raises(TypeError):
            UnitSelector()  # noqa

    def test_custom_selector_subclass_resolves_layer_path(self):
        """Future subclasses just need a ``layer_path`` property."""
        class FakeMultiLayer(UnitSelector):
            def __init__(self, names):
                self._names = names
            @property
            def layer_path(self) -> str:
                return self._names[0]  # canonical / default layer
        sel = FakeMultiLayer(['l.2', 'l.3', 'l.4'])
        m = BrainScoreModel(
            identifier='m', model=None,
            region_layer_map={'V1': sel},
            extractor={'vision': _noop_extractor},
        )
        assert m.region_layer_map['V1'] is sel
        assert m.region_layer_selectors['V1'] is sel


# -- BrainScoreModel dispatch tests ------------------------------------------

class TestBrainScoreModelDispatch:

    def test_vision_routes_to_vision_extractor(self):
        """Vision stimuli route through the vision extractor."""
        act_model = StubActivationsModel(return_value='vision_assembly')
        m = BrainScoreModel(
            identifier='resnet',
            model='torch_model',
            region_layer_map={'IT': 'layer4'},
            extractor={'vision': act_model},
        )
        m.start_recording('IT')
        stimuli = StubStimulusSet(columns=['image_file_name', 'stimulus_id'])
        result = m.process(stimuli)
        assert result == 'vision_assembly'
        assert act_model.call_args['stimuli'] is stimuli
        assert act_model.call_args['layers'] == ['layer4']

    def test_process_without_start_recording_raises(self):
        """A run that records no region must fail loudly, not return 0 neuroids."""
        m = BrainScoreModel(
            identifier='resnet',
            model='torch_model',
            region_layer_map={'IT': 'layer4'},
            extractor={'vision': StubActivationsModel()},
        )
        stimuli = StubStimulusSet(columns=['image_file_name', 'stimulus_id'])
        with pytest.raises(ValueError, match="records no region"):
            m.process(stimuli)  # no start_recording called

    def test_start_recording_unknown_region_warns(self):
        """A typo'd region name warns instead of silently becoming a layer path."""
        m = BrainScoreModel(
            identifier='resnet',
            model='torch_model',
            region_layer_map={'IT': 'layer4'},
            extractor={'vision': StubActivationsModel()},
        )
        with pytest.warns(UserWarning, match="not a region"):
            m.start_recording('ITT')  # typo of 'IT'

    def test_multi_modality_process_without_start_recording_raises(self):
        """The forgotten-start_recording guard must also cover multi_modality=True."""
        m = BrainScoreModel(
            identifier='vlm',
            model='torch_model',
            region_layer_map={'IT': LayerSelector('layer4', modality='vision')},  # regions defined
            extractor={'vision': StubActivationsModel(),
                       'text': make_stub_extractor()},
        )
        stimuli = StubStimulusSet(columns=['image_file_name', 'sentence'])
        with pytest.raises(ValueError, match="records no region"):
            m.process(stimuli, multi_modality=True)

    def test_selected_plain_modality_not_false_positived(self):
        """The guard must check the SELECTED modality, not ANY detected one: a
        layer-aware non-selected tower must not turn a selected input with no
        extractor into a 'records no region' error."""
        class _LayerAwareAudio:
            identifier = 'audio-wrapper'

            def __call__(self, stimuli, layers=None, **kwargs):
                return 'audio_out'

        m = BrainScoreModel(
            identifier='mm',
            model='torch_model',
            region_layer_map={'A1': LayerSelector('layer', modality='audio')},  # none active
            in_channels={'text'},          # declared input, no extractor
            extractor={'audio': _LayerAwareAudio()},  # layer-aware -> extraction
        )
        stimuli = StubStimulusSet(columns=['sentence', 'audio_file_name'])
        # text wins MODALITY_PRIORITY; the error names the missing text
        # extractor, not the audio tower's missing recording
        with pytest.raises(NotImplementedError, match='no activation extractor'):
            m.process(stimuli)

    def test_text_routes_to_text_extractor(self):
        """Text stimuli route to the text extractor."""
        text_proc = make_stub_extractor(return_value='text_assembly')
        m = BrainScoreModel(
            identifier='clip',
            model='clip_model',
            region_layer_map={'language_system': LayerSelector(
                'text_model.encoder.layers.10', modality='text')},
            extractor={
                'vision': StubActivationsModel(),
                'text': text_proc,
            },
        )
        m.start_recording('language_system')
        stimuli = StubStimulusSet(columns=['sentence', 'stimulus_id'])
        result = m.process(stimuli)
        assert result == 'text_assembly'
        assert text_proc.call_args['stimuli'] is stimuli
        assert text_proc.call_args['layers'] == ['text_model.encoder.layers.10']

    def test_recording_layer_resolved_from_region_map(self):
        """start_recording resolves brain region to model layer."""
        act_model = StubActivationsModel()
        m = BrainScoreModel(
            identifier='resnet',
            model='m',
            region_layer_map={'IT': 'layer4'},
            extractor={'vision': act_model},
        )
        m.start_recording('IT')
        stimuli = StubStimulusSet(columns=['image_file_name'])
        m.process(stimuli)
        assert act_model.call_args['layers'] == ['layer4']

    def test_no_modality_columns_raises(self):
        m = BrainScoreModel(
            identifier='test',
            model=None,
            region_layer_map={},
            extractor={'vision': make_stub_extractor()},
        )
        stimuli = StubStimulusSet(columns=['unknown_column'])
        with pytest.raises(ValueError, match="No column in this stimulus set"):
            m.process(stimuli)

    def test_paths_only_stimulus_set_error_names_the_column_to_add(self):
        import pandas as pd
        from brainscore_core.supported_data_standards.brainio.stimuli import StimulusSet
        m = BrainScoreModel(identifier='test', extractor={'vision': lambda stimuli, layers: None})
        stimuli = StimulusSet(pd.DataFrame({'stimulus_id': ['a']}))
        stimuli.stimulus_paths = {'a': '/tmp/a.png'}
        with pytest.raises(ValueError, match=r"No column.*image_path.*stimulus_paths dict alone"):
            m.process(stimuli)

    def test_single_modality_detected_from_multi_extractor_model(self):
        """If only one modality is in the stimuli, route to that modality."""
        act_model = StubActivationsModel(return_value='vis_only')
        text_proc = make_stub_extractor()
        m = BrainScoreModel(
            identifier='vlm',
            model=None,
            region_layer_map={},
            extractor={
                'vision': act_model,
                'text': text_proc,
            },
        )
        stimuli = StubStimulusSet(columns=['image_file_name'])
        result = m.process(stimuli)
        assert result == 'vis_only'
        assert text_proc.call_args is None


# -- Behavioral path dispatch tests ------------------------------------------

class TestBehavioralPathDispatch:
    """The parts a model is built with decide its behavioral path; a path
    that cannot run errors clearly."""

    def _make_dual_path_model(self, has_generation=True, has_readout=True):
        """A model with both behavioral_readout_layer AND answer set,
        so that dispatch has a real choice to make."""
        answer = ({'answer': lambda stim_row, instruction, label_set: label_set[0]}
                  if has_generation else {})
        return BrainScoreModel(
            identifier='dual-path-model',
            model=None,
            region_layer_map={'IT': 'layer4'},
            extractor={'vision': make_stub_extractor()},
            behavioral_readout_layer='layer4' if has_readout else None,
            **answer,
        )

    def _fitting_stub(self):
        """Minimal fitting_stimuli stub — has columns + label getter so the
        readout fitting code can pretend to fit."""
        return StubStimulusSet(columns=['image_file_name', 'stimulus_id', 'label'])

    def test_readout_only_model_without_fitting_stimuli_raises(self):
        """A model built without answer= reads out; without fitting_stimuli it errors clearly."""
        m = self._make_dual_path_model(has_generation=False)
        ctx = TaskContext(
            task_type='probabilities',
            label_set=['real', 'pseudo'],
            instruction="Is this real?",
            fitting_stimuli=None,           # missing
        )
        with pytest.raises(ValueError, match="readout needs TaskContext.fitting_stimuli"):
            m.start_task(ctx)

    def test_unknown_task_type_raises(self):
        """A typo such as 'choice' must not silently run as a passive task."""
        m = self._make_dual_path_model()
        ctx = TaskContext(task_type='choice', label_set=['real', 'pseudo'],
                          instruction="Is this real?")
        with pytest.raises(ValueError, match="Unknown task_type 'choice'"):
            m.start_task(ctx)

    def test_passive_task_type_still_accepted(self):
        m = self._make_dual_path_model()
        m.start_task(TaskContext(task_type='passive'))  # no error

    def test_no_dual_path_warning_without_readout_layer(self):
        """An answer-only model given fitting stimuli has one path, not two."""
        m = self._make_dual_path_model(has_readout=False)
        ctx = TaskContext(task_type='probabilities', label_set=['real', 'pseudo'],
                          instruction="Is this real?",
                          fitting_stimuli=self._fitting_stub())
        with warnings.catch_warnings():
            warnings.simplefilter('error', UserWarning)
            m.start_task(ctx)

    def test_dual_path_warning_with_readout_layer(self):
        m = self._make_dual_path_model()
        ctx = TaskContext(task_type='probabilities', label_set=['real', 'pseudo'],
                          instruction="Is this real?",
                          fitting_stimuli=self._fitting_stub())
        with pytest.warns(UserWarning, match='Both generation and fitted readout'):
            m.start_task(ctx)


# -- BrainScoreModel state tests ---------------------------------------------

class TestBrainScoreModelState:

    def test_start_recording_resolves_region(self):
        extractor = StubActivationsModel()
        m = BrainScoreModel(
            identifier='test',
            model=None,
            region_layer_map={'V1': 'layer1', 'IT': 'layer4'},
            extractor={'vision': extractor},
        )
        m.start_recording('IT')
        m.process(StubStimulusSet(['image_file_name']))
        assert extractor.call_args['layers'] == ['layer4']

    def test_start_recording_passes_through_unknown_region(self):
        extractor = StubActivationsModel()
        m = BrainScoreModel(
            identifier='test',
            model=None,
            region_layer_map={'V1': 'layer1'},
            extractor={'vision': extractor},
        )
        m.start_recording('layer3.2')
        m.process(StubStimulusSet(['image_file_name']))
        assert extractor.call_args['layers'] == ['layer3.2']

    def test_reset_clears_state(self):
        m = BrainScoreModel(
            identifier='test',
            model=None,
            region_layer_map={'IT': 'layer4'},
            extractor={'vision': make_stub_extractor()},
        )
        m.start_recording('IT')
        # Use a non-behavioral task_type so start_task doesn't require
        # a viable behavioral path. Goal here is to verify reset() — not
        # the dispatch logic. (Behavioral task_types like 'classification'
        # require fitting_stimuli or instruction+answer — see
        # TestPreferPathDispatch for that coverage.)
        m.start_task(TaskContext(task_type='passive'))
        m.reset()
        with pytest.raises(ValueError, match='records no region'):
            m.process(StubStimulusSet(['image_file_name']))
        assert m._behavioral.task is None


# -- Multi-region start_recording tests (#36) --------------------------------

class _XArrayActivationsModel:
    """Activations stub returning an xarray DataArray keyed by layer.

    Returns one neuroid per layer in `layers`, with the layer name on the
    neuroid coord. Lets multi-region dispatch tests inspect the resulting
    'region' coord that BrainScoreModel adds.
    """

    def __init__(self):
        self.call_args = None

    def __call__(self, stimuli, layers=None, **kwargs):
        import numpy as np
        from brainscore_core.supported_data_standards.brainio.assemblies import NeuroidAssembly
        self.call_args = {'stimuli': stimuli, 'layers': layers}
        # Default to one neuroid even when caller passes empty layers,
        # mimicking real wrappers that extract at a default location.
        layers = list(layers) if layers else ['default_layer']
        n_neuroid = len(layers)
        n_pres = 2
        data = np.zeros((n_pres, n_neuroid))
        return NeuroidAssembly(
            data,
            dims=('presentation', 'neuroid'),
            coords={
                'layer': ('neuroid', np.array(layers)),
                'neuroid_id': ('neuroid', np.arange(n_neuroid)),
            },
        )


class _NeuroidAssemblyActivationsModel:
    """Returns BrainIO's MultiIndex-backed assembly in a chosen layer order."""

    def __init__(self, output_layers):
        self.output_layers = list(output_layers)

    def __call__(self, stimuli, layers=None, **kwargs):
        del stimuli, layers, kwargs
        import numpy as np
        from brainscore_core.supported_data_standards.brainio.assemblies import (
            NeuroidAssembly,
        )

        n_neuroid = len(self.output_layers)
        return NeuroidAssembly(
            np.zeros((2, n_neuroid)),
            dims=('presentation', 'neuroid'),
            coords={
                'layer': ('neuroid', np.array(self.output_layers)),
                'neuroid_id': ('neuroid', np.arange(n_neuroid)),
            },
        )


class _UnattributedActivationsModel:
    def __call__(self, stimuli, layers=None, **kwargs):
        del stimuli, layers, kwargs
        import xarray as xr
        import numpy as np

        return xr.DataArray(
            np.zeros((2, 2)),
            dims=('presentation', 'neuroid'),
            coords={'neuroid_id': ('neuroid', np.arange(2))},
        )


class TestMultiRegionStartRecording:

    def test_provenance_follows_layer_coord_not_position(self):
        act = _NeuroidAssemblyActivationsModel(
            ['late_layer', 'early_layer']
        )
        m = BrainScoreModel(
            identifier='test',
            model=None,
            region_layer_map={
                'early': 'early_layer',
                'late': 'late_layer',
            },
            extractor={'vision': act},
        )
        m.start_recording(['early', 'late'])

        assembly = m.process(StubStimulusSet(columns=['image_file_name']))

        provenance = dict(zip(assembly['layer'].values.tolist(),
                              assembly['region'].values.tolist()))
        assert provenance == {
            'early_layer': 'early',
            'late_layer': 'late',
        }

    def test_multi_layer_output_without_layer_coord_raises(self):
        m = BrainScoreModel(
            identifier='test',
            model=None,
            region_layer_map={
                'early': 'early_layer',
                'late': 'late_layer',
            },
            extractor={'vision': _UnattributedActivationsModel()},
        )
        m.start_recording(['early', 'late'])

        with pytest.raises(ValueError, match="no per-neuroid 'layer' coordinate"):
            m.process(StubStimulusSet(columns=['image_file_name']))

    def test_single_region_string_backward_compat(self):
        """String target keeps old behavior; no 'region' coord added."""
        act = _XArrayActivationsModel()
        m = BrainScoreModel(
            identifier='test',
            model=None,
            region_layer_map={'V4': 'layer.8', 'IT': 'layer.10'},
            extractor={'vision': act},
        )
        m.start_recording('IT')
        stimuli = StubStimulusSet(columns=['image_file_name'])
        assembly = m.process(stimuli)
        assert 'region' not in assembly.coords
        assert act.call_args['layers'] == ['layer.10']

    def test_multi_region_list_adds_region_coord(self):
        """List target produces a 'region' coord on neuroids and extracts both layers."""
        act = _XArrayActivationsModel()
        m = BrainScoreModel(
            identifier='test',
            model=None,
            region_layer_map={'V4': 'layer.8', 'IT': 'layer.10'},
            extractor={'vision': act},
        )
        m.start_recording(['V4', 'IT'])
        stimuli = StubStimulusSet(columns=['image_file_name'])
        assembly = m.process(stimuli)
        assert 'region' in assembly.coords
        assert set(assembly['region'].values.tolist()) == {'V4', 'IT'}
        # Single forward pass for both layers
        assert act.call_args['layers'] == ['layer.8', 'layer.10']

    def test_overlapping_regions_share_layer(self):
        """Two regions mapping to the same layer extract once; both tagged."""
        act = _XArrayActivationsModel()
        m = BrainScoreModel(
            identifier='test',
            model=None,
            region_layer_map={'V4': 'layer.8', 'IT': 'layer.8'},
            extractor={'vision': act},
        )
        m.start_recording(['V4', 'IT'])
        stimuli = StubStimulusSet(columns=['image_file_name'])
        assembly = m.process(stimuli)
        # Single layer present once; region coord joins both regions
        assert act.call_args['layers'] == ['layer.8']
        assert 'V4|IT' in assembly['region'].values.tolist()

    def test_unknown_name_in_list_records_raw_layer(self):
        """Wrapping a layer name in a list keeps its meaning."""
        act = _XArrayActivationsModel()
        m = BrainScoreModel(
            identifier='test',
            model=None,
            region_layer_map={'V4': 'layer.8'},
            extractor={'vision': act},
        )
        with pytest.warns(UserWarning, match="raw layer"):
            m.start_recording(['V4', 'NotARegion'])
        m.process(StubStimulusSet(columns=['image_file_name']))
        assert act.call_args['layers'] == ['layer.8', 'NotARegion']

    def test_reset_clears_multi_region_state(self):
        """reset() clears the multi-region attributes alongside the singular one."""
        m = BrainScoreModel(
            identifier='test',
            model=None,
            region_layer_map={'V4': 'layer.8', 'IT': 'layer.10'},
            extractor={'vision': _XArrayActivationsModel()},
        )
        m.start_recording(['V4', 'IT'])
        m.reset()
        with pytest.raises(ValueError, match='records no region'):
            m.process(StubStimulusSet(columns=['image_file_name']))

    def test_start_recording_a_list_of_every_region(self):
        """A list of every mapped region records them all in one extraction."""
        act = _XArrayActivationsModel()
        m = BrainScoreModel(
            identifier='test', model=None,
            region_layer_map={
                'V1': 'layer.2', 'V4': 'layer.8',
                'IT': 'layer.10', 'language_network': 'layer.20',
            },
            extractor={'vision': act},
        )
        m.start_recording(['V1', 'V4', 'IT', 'language_network'])
        stimuli = StubStimulusSet(columns=['image_file_name'])
        assembly = m.process(stimuli)
        assert act.call_args['layers'] == ['layer.2', 'layer.8', 'layer.10', 'layer.20']
        assert assembly['region'].values.tolist() == ['V1', 'V4', 'IT', 'language_network']


# -- Multi-modality dispatch tests -------------------------------------------

class _XArrayPreprocessor:
    """Layer-aware text/audio preprocessor stub (mimics TextWrapper).

    Returns an xarray DataArray keyed by layer with one neuroid per layer,
    so multi-modality dispatch can concat assemblies from this stub with
    activations-model output.
    """

    identifier = 'stub_extractor'

    def __init__(self, label='text'):
        self.label = label
        self.call_args = None

    def __call__(self, stimuli, layers=None, **kwargs):
        import numpy as np
        from brainscore_core.supported_data_standards.brainio.assemblies import NeuroidAssembly
        self.call_args = {'stimuli': stimuli, 'layers': layers}
        layers = list(layers or [f'{self.label}_layer'])
        n_neuroid = len(layers)
        n_pres = 2
        data = np.zeros((n_pres, n_neuroid))
        return NeuroidAssembly(
            data,
            dims=('presentation', 'neuroid'),
            coords={
                'layer': ('neuroid', np.array(layers)),
                'neuroid_id': (
                    'neuroid',
                    np.array([f'{self.label}_n{i}' for i in range(n_neuroid)]),
                ),
            },
        )


_RICH_PRESENTATION_LEVELS = [
    'stimulus_id',
    'image_label',
    'word',
    'sentence',
    'realpseudo',
    'numeric_label',
    'image_file_name',
]

_RICH_PRESENTATION_VALUES = {
    'stimulus_id': ['roar_0000', 'roar_0001'],
    'image_label': ['real', 'pseudo'],
    'word': ['able', 'ablood'],
    'sentence': ['able', 'ablood'],
    'realpseudo': ['real', 'pseudo'],
    'numeric_label': [1, 0],
    'image_file_name': ['roar_0000.png', 'roar_0001.png'],
}


def _rich_presentation_index(levels=None):
    import pandas as pd

    levels = levels or _RICH_PRESENTATION_LEVELS
    return pd.MultiIndex.from_arrays(
        [_RICH_PRESENTATION_VALUES[level] for level in levels],
        names=levels,
    )


def _rich_presentation_coords(exclude=()):
    exclude = set(exclude)
    return {
        name: ('presentation', values)
        for name, values in _RICH_PRESENTATION_VALUES.items()
        if name not in exclude
    }


class _RichPresentationExtractor:
    """Layer-aware extractor with configurable presentation coord packaging."""

    identifier = 'rich_presentation_extractor'

    def __init__(self, label, base_value, presentation_mode):
        self.label = label
        self.base_value = base_value
        self.presentation_mode = presentation_mode
        self.call_args = None

    def __call__(self, stimuli, layers=None, **kwargs):
        import numpy as np
        import xarray as xr

        self.call_args = {'stimuli': stimuli, 'layers': layers}
        layers = list(layers or [f'{self.label}_layer'])
        n_presentation = len(_RICH_PRESENTATION_VALUES['stimulus_id'])
        n_neuroid = len(layers)
        data = (
            self.base_value
            + np.arange(n_presentation * n_neuroid).reshape(
                n_presentation, n_neuroid
            )
        )
        coords = {
            'layer': ('neuroid', np.array(layers)),
            'neuroid_id': (
                'neuroid',
                np.array([
                    f'{self.label}_n{i}' for i in range(n_neuroid)
                ]),
            ),
        }
        if self.presentation_mode == 'multiindex':
            coords['presentation'] = _rich_presentation_index()
        elif self.presentation_mode == 'partial_index':
            coords['presentation'] = _rich_presentation_index(['stimulus_id'])
            coords.update(_rich_presentation_coords(exclude={'stimulus_id'}))
        elif self.presentation_mode == 'coords':
            coords.update(_rich_presentation_coords())
        else:
            raise ValueError(
                f'unknown presentation_mode={self.presentation_mode!r}'
            )
        return xr.DataArray(
            data,
            dims=('presentation', 'neuroid'),
            coords=coords,
        )


class TestMultiModalityDispatch:

    def test_default_single_modality_backward_compat(self):
        """Without multi_modality=True, multimodal stim warns and then picks
        one modality via MODALITY_PRIORITY (vision wins)."""
        act = _XArrayActivationsModel()
        text_proc = _XArrayPreprocessor(label='text')
        m = BrainScoreModel(
            identifier='test',
            model=None,
            region_layer_map={'IT': LayerSelector('layer.10', modality='vision')},
            extractor={'vision': act, 'text': text_proc},
        )
        m.start_recording('IT')
        stimuli = StubStimulusSet(columns=['image_file_name', 'sentence'])
        with pytest.warns(UserWarning, match="multiple supported modalities"):
            assembly = m.process(stimuli)
        # Vision wins; text wrapper not invoked
        assert text_proc.call_args is None
        assert 'modality' not in assembly.coords

    def test_single_modality_default_does_not_warn(self):
        """Single-modality stimuli keep the exact default dispatch path."""
        act = _XArrayActivationsModel()
        text_proc = _XArrayPreprocessor(label='text')
        m = BrainScoreModel(
            identifier='test',
            model=None,
            region_layer_map={'IT': LayerSelector('layer.10', modality='vision')},
            extractor={'vision': act, 'text': text_proc},
        )
        m.start_recording('IT')
        stimuli = StubStimulusSet(columns=['image_file_name'])
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            assembly = m.process(stimuli)
        assert not [
            warning for warning in caught
            if "multiple supported modalities" in str(warning.message)
        ]
        assert text_proc.call_args is None
        assert 'modality' not in assembly.coords

    def test_multi_modality_invokes_all_wrappers(self):
        """multi_modality=True extracts from every wrapper whose modality
        is present in the stimulus set."""
        act = _XArrayActivationsModel()
        text_proc = _XArrayPreprocessor(label='text')
        m = BrainScoreModel(
            identifier='test',
            model=None,
            region_layer_map={},
            extractor={'vision': act, 'text': text_proc},
        )
        stimuli = StubStimulusSet(columns=['image_file_name', 'sentence'])
        assembly = m.process(stimuli, multi_modality=True)
        # Both wrappers fired
        assert act.call_args is not None
        assert text_proc.call_args is not None
        # Output carries a modality coord
        assert 'modality' in assembly.coords
        assert set(assembly['modality'].values.tolist()) == {'vision', 'text'}

    def test_explicit_multi_modality_does_not_warn(self):
        """multi_modality=True is an explicit choice, not an ambiguous fallback."""
        act = _XArrayActivationsModel()
        text_proc = _XArrayPreprocessor(label='text')
        m = BrainScoreModel(
            identifier='test',
            model=None,
            region_layer_map={},
            extractor={'vision': act, 'text': text_proc},
        )
        stimuli = StubStimulusSet(columns=['image_file_name', 'sentence'])
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            assembly = m.process(stimuli, multi_modality=True)
        assert not [
            warning for warning in caught
            if "multiple supported modalities" in str(warning.message)
        ]
        assert 'modality' in assembly.coords

    def test_multi_modality_falls_back_to_single_when_one_modality(self):
        """multi_modality=True is a no-op when only one modality is
        actually detected — leaves the assembly without a modality coord."""
        act = _XArrayActivationsModel()
        text_proc = _XArrayPreprocessor(label='text')
        m = BrainScoreModel(
            identifier='test',
            model=None,
            region_layer_map={'IT': LayerSelector('layer.10', modality='vision')},
            extractor={'vision': act, 'text': text_proc},
        )
        m.start_recording('IT')
        # Only image column — text wrapper should NOT fire even with multi_modality
        stimuli = StubStimulusSet(columns=['image_file_name'])
        assembly = m.process(stimuli, multi_modality=True)
        assert text_proc.call_args is None
        assert 'modality' not in assembly.coords

    def test_multi_modality_skips_unsupported_modalities(self):
        """When stim has a modality the model has no extractor for, the
        single-modality detector already filters it. multi_modality fans
        out only over supported-and-detected modalities."""
        act = _XArrayActivationsModel()
        text_proc = _XArrayPreprocessor(label='text')
        m = BrainScoreModel(
            identifier='test',
            model=None,
            region_layer_map={},
            # Model has no audio extractor
            extractor={'vision': act, 'text': text_proc},
        )
        # audio_path feeds the audio channel, but this model has no audio extractor
        stimuli = StubStimulusSet(columns=['image_file_name', 'sentence', 'audio_path'])
        assembly = m.process(stimuli, multi_modality=True)
        modalities = set(assembly['modality'].values.tolist())
        assert modalities == {'vision', 'text'}  # audio dropped silently

    def test_multi_modality_concat_neuroid_count(self):
        """Output neuroid count is the sum across modalities."""
        act = _XArrayActivationsModel()
        text_proc = _XArrayPreprocessor(label='text')
        m = BrainScoreModel(
            identifier='test',
            model=None,
            region_layer_map={'IT': LayerSelector('layer.10', modality='vision')},
            extractor={'vision': act, 'text': text_proc},
        )
        m.start_recording('IT')
        stimuli = StubStimulusSet(columns=['image_file_name', 'sentence'])
        single = m.process(stimuli)  # vision only — 1 neuroid (1 layer)
        assert single.sizes['neuroid'] == 1
        # A text region adds the text tower: one neuroid per tower.
        m.region_layer_map = {'IT': LayerSelector('layer.10', modality='vision'),
                              'language': LayerSelector('layer.20', modality='text')}
        m.start_recording(['IT', 'language'])
        multi = m.process(stimuli, multi_modality=True)
        assert multi.sizes['neuroid'] == 2

    def _make_rich_presentation_model(self, vision_mode, text_mode):
        vision_act = _RichPresentationExtractor(
            label='vision', base_value=1.0,
            presentation_mode=vision_mode,
        )
        text_extractor = _RichPresentationExtractor(
            label='text', base_value=101.0,
            presentation_mode=text_mode,
        )
        model = BrainScoreModel(
            identifier='test',
            model=None,
            region_layer_map={},
            extractor={
                'vision': vision_act,
                'text': text_extractor,
            },
        )
        return model, vision_act, text_extractor

    def test_multi_modality_reconciles_rich_presentation_multiindex(self):
        """Rich presentation metadata that used to collide now fans out."""
        import xarray as xr

        stimuli = StubStimulusSet(columns=['image_file_name', 'sentence'])

        colliding, vision_act, text_extractor = (
            self._make_rich_presentation_model(
                vision_mode='multiindex',
                text_mode='partial_index',
            )
        )
        assembly = colliding.process(stimuli, multi_modality=True)

        baseline, _, _ = self._make_rich_presentation_model(
            vision_mode='coords',
            text_mode='coords',
        )
        expected = baseline.process(stimuli, multi_modality=True)

        assert vision_act.call_args is not None
        assert text_extractor.call_args is not None
        assert set(assembly['modality'].values.tolist()) == {'vision', 'text'}
        assert assembly.sizes['neuroid'] == 2
        assert assembly.indexes['presentation'].names == _RICH_PRESENTATION_LEVELS
        assert int((assembly['modality'] == 'vision').sum()) == 1
        assert int((assembly['modality'] == 'text').sum()) == 1
        xr.testing.assert_identical(assembly, expected)


# -- Cross-tower layer routing tests -----------------------------------------


class TestCrossTowerRouting:
    """The modality on each region's LayerSelector filters layers per wrapper,
    so each tower extracts only its own regions' layers in multi-modality
    dispatch."""

    def _make_cross_tower_model(self, routed=True):
        """Two wrappers + region_layer_map across both towers."""
        act = _XArrayActivationsModel()
        text_proc = _XArrayPreprocessor(label='text')
        if routed:
            region_layer_map = {
                'IT': LayerSelector('vision_model.layer.10', modality='vision'),
                'language_network': LayerSelector('text_model.layer.20', modality='text'),
            }
        else:
            region_layer_map = {
                'IT': 'vision_model.layer.10',
                'language_network': 'text_model.layer.20',
            }
        return BrainScoreModel(
            identifier='cross-tower',
            model=None,
            region_layer_map=region_layer_map,
            extractor={
                'vision': act,
                'text': text_proc,
            },
        ), act, text_proc

    def test_unknown_modality_in_selector_raises(self):
        with pytest.raises(ValueError, match="no extractor"):
            BrainScoreModel(
                identifier='test',
                model=None,
                region_layer_map={'IT': LayerSelector('layer.10', modality='no_such_modality')},
                extractor={'vision': make_stub_extractor()},
            )

    def test_cross_tower_routing_filters_per_wrapper(self):
        """With each region routed to a modality, each wrapper gets only its
        own modality's layer."""
        m, act, text_proc = self._make_cross_tower_model()
        m.start_recording(['IT', 'language_network'])
        stimuli = StubStimulusSet(columns=['image_file_name', 'sentence'])
        m.process(stimuli, multi_modality=True)
        # Vision wrapper only saw IT's layer
        assert act.call_args['layers'] == ['vision_model.layer.10']
        # Text wrapper only saw language_network's layer
        assert text_proc.call_args['layers'] == ['text_model.layer.20']

    def test_cross_tower_assembly_carries_modality_coord(self):
        m, act, text_proc = self._make_cross_tower_model()
        m.start_recording(['IT', 'language_network'])
        stimuli = StubStimulusSet(columns=['image_file_name', 'sentence'])
        assembly = m.process(stimuli, multi_modality=True)
        modalities = set(assembly['modality'].values.tolist())
        assert modalities == {'vision', 'text'}

    def test_without_region_modalities_the_model_is_refused(self):
        """A model with two extractors names the one each region is recorded
        from. Without it, a region would be recorded from whichever tower gets
        input, and pre-flight could not check it."""
        with pytest.raises(ValueError, match='several extractors'):
            self._make_cross_tower_model(routed=False)

    def test_cross_tower_skips_wrapper_with_no_layers(self):
        """If recording is single-region but multi_modality=True, the
        non-matching wrapper should be skipped entirely (filtered
        layers = empty)."""
        m, act, text_proc = self._make_cross_tower_model()
        m.start_recording('IT')  # only vision tower
        stimuli = StubStimulusSet(columns=['image_file_name', 'sentence'])
        assembly = m.process(stimuli, multi_modality=True)
        # Text wrapper not invoked
        assert text_proc.call_args is None
        # Output has only 'vision' modality entries
        modalities = set(assembly['modality'].values.tolist())
        assert modalities == {'vision'}

    def test_cross_tower_single_modality_path_unchanged(self):
        """Single-modality dispatch ignores the region modalities (used
        only by multi_modality fan-out)."""
        m, act, text_proc = self._make_cross_tower_model()
        m.start_recording('IT')
        stimuli = StubStimulusSet(columns=['image_file_name'])
        assembly = m.process(stimuli)
        # Vision tower only — single-modality always uses pick_modality
        assert act.call_args is not None
        assert text_proc.call_args is None
        assert 'modality' not in assembly.coords


# -- BrainScoreModel column detection tests -----------------------------------

class TestBrainScoreModelColumnDetection:

    def _make_model(self, *modalities):
        extractors = {m: make_stub_extractor(f'{m}_out') for m in modalities}
        return BrainScoreModel(
            identifier='test',
            model=None,
            region_layer_map={},
            extractor=extractors,
        )

    def test_video_model_declares_the_video_channel(self):
        m = self._make_model('video')
        assert m.in_channels == {'vision:video', 'control'}

    def test_video_extractor_resolves_under_canonical_vision(self):
        # a video-keyed extractor records a video column, detected as 'vision'
        m = BrainScoreModel('test', extractor={'video': make_stub_extractor('video_out')},
                            region_layer_map={'IT': 'layer'})
        m.start_recording('IT')
        assert m.process(StubStimulusSet(['video_path'])) == 'video_out'

    def test_canonical_modality_folds_video_into_vision(self):
        from brainscore_core.io_catalog import canonical_modality
        assert canonical_modality('video') == 'vision'
        assert canonical_modality('vision') == 'vision'
        assert canonical_modality('audio') == 'audio'

    def test_video_extractor_with_vision_region_map_constructs(self):
        # a 'video' extractor with a region selector saying 'vision' and a
        # required video channel must construct (video canonicalizes to vision)
        m = BrainScoreModel(
            identifier='v', model=None,
            region_layer_map={'IT': LayerSelector('layer4', modality='vision')},
            extractor={'video': make_stub_extractor('out')},
            required_channels={'vision:video'},
        )
        assert m.required_channels == {'vision:video'}
