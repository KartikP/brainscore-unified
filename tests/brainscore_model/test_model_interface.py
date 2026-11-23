import pytest
import warnings

from brainscore import BrainScoreModel, UnitSelector, LayerSelector
from brainscore_core.compatibility import ExecutionNotFeasibleError
from brainscore_core.contract import TaskContext
from brainscore_core.responses import Choice
from brainscore_core.streaming_helpers import stimulus_session


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


def _stimuli(columns):
    """Two stimuli with one value per named input column."""
    from brainscore_core.supported_data_standards.brainio.stimuli import StimulusSet
    data = {'stimulus_id': ['roar_0000', 'roar_0001']}
    data.update({column: [f'{column}0', f'{column}1'] for column in columns})
    stimuli = StimulusSet(data)
    stimuli.identifier = 'model-interface-' + '-'.join(columns)
    return stimuli


def _record(model, columns, *regions):
    """The payload of each requested region, from one session on ``_stimuli(columns)``."""
    session = stimulus_session(_stimuli(columns), [f'neural:{region}' for region in regions])
    model.participate(session)
    return [session.collect(f'neural:{region}') for region in regions]


def _answer(model, task, columns=('image_file_name',)):
    """Run one behavior:choice session under ``task``."""
    model.participate(stimulus_session(_stimuli(columns), ['behavior:choice'], task))


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
        extractor = _XArrayActivationsModel()
        m = BrainScoreModel('test', region_layer_map={'IT': 'layer1'},
                            extractor={'vision': extractor})
        m.region_layer_map = {'IT': 'layer9'}
        _record(m, ['image_file_name'], 'IT')
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

    def test_a_neural_session_records_the_layer_of_its_region(self):
        """A neural:IT session records the layer the region map names, through participate."""
        from brainscore_core.streaming_helpers import stimulus_session
        from tests.brainscore_model.test_behavior_paths import STIMULI, _features
        calls = []

        def extract(stimuli, layers, **kwargs):
            calls.append(list(layers))
            return _features(stimuli, layers, **kwargs)
        m = BrainScoreModel(identifier='resnet', region_layer_map={'IT': 'layer4'}, extractor={'text': extract})
        session = stimulus_session(STIMULI, ['neural:IT'])
        m.participate(session)
        assert calls == [['layer4']]
        recorded = session.collect('neural:IT')
        assert list(recorded['stimulus_id'].values) == list(STIMULI['stimulus_id'])
        assert set(recorded['layer'].values) == {'layer4'}


    def test_text_routes_to_text_extractor(self):
        """Text stimuli route to the text extractor."""
        text_proc = _XArrayPreprocessor(label='text')
        m = BrainScoreModel(
            identifier='clip',
            model='clip_model',
            region_layer_map={'language_system': LayerSelector(
                'text_model.encoder.layers.10', modality='text')},
            extractor={
                'vision': _XArrayActivationsModel(),
                'text': text_proc,
            },
        )
        _record(m, ['sentence'], 'language_system')
        assert list(text_proc.call_args['stimuli']['sentence']) == ['sentence0', 'sentence1']
        assert text_proc.call_args['layers'] == ['text_model.encoder.layers.10']

    def test_recording_layer_resolved_from_region_map(self):
        """A session's region resolves to the model layer."""
        act_model = _XArrayActivationsModel()
        m = BrainScoreModel(
            identifier='resnet',
            model='m',
            region_layer_map={'IT': 'layer4'},
            extractor={'vision': act_model},
        )
        _record(m, ['image_file_name'], 'IT')
        assert act_model.call_args['layers'] == ['layer4']

    def test_no_modality_columns_raises(self):
        m = BrainScoreModel(
            identifier='test',
            model=None,
            region_layer_map={'IT': 'layer4'},
            extractor={'vision': _XArrayActivationsModel()},
        )
        with pytest.raises(ValueError, match="No column in this stimulus set"):
            _record(m, ['unknown_column'], 'IT')

    def test_paths_only_stimulus_set_error_names_the_column_to_add(self):
        import pandas as pd
        from brainscore_core.streaming_helpers import stimulus_session
        from brainscore_core.supported_data_standards.brainio.stimuli import StimulusSet
        m = BrainScoreModel(identifier='test', region_layer_map={'IT': 'layer4'},
                            extractor={'vision': _XArrayActivationsModel()})
        stimuli = StimulusSet(pd.DataFrame({'stimulus_id': ['a']}))
        stimuli.identifier = 'paths-only'
        stimuli.stimulus_paths = {'a': '/tmp/a.png'}
        with pytest.raises(ValueError, match=r"No column.*image_path.*stimulus_paths dict alone"):
            m.participate(stimulus_session(stimuli, ['neural:IT']))

    def test_single_modality_detected_from_multi_extractor_model(self):
        """If only one modality is in the stimuli, route to that modality."""
        act_model = _XArrayActivationsModel()
        text_proc = _XArrayPreprocessor(label='text')
        m = BrainScoreModel(
            identifier='vlm',
            model=None,
            region_layer_map={'IT': LayerSelector('layer4', modality='vision')},
            extractor={
                'vision': act_model,
                'text': text_proc,
            },
        )
        _record(m, ['image_file_name'], 'IT')
        assert act_model.call_args is not None
        assert text_proc.call_args is None


# -- Behavioral path dispatch tests ------------------------------------------

class TestBehavioralPathDispatch:
    """The parts a model is built with decide its behavioral path; a path
    that cannot run errors clearly."""

    def _make_dual_path_model(self, has_generation=True, has_readout=True):
        """A model with both behavioral_readout_layer AND answer set,
        so that dispatch has a real choice to make."""
        answer = ({'answer': lambda stimulus_row, instruction, label_set: label_set[0]}
                  if has_generation else {})
        return BrainScoreModel(
            identifier='dual-path-model',
            model=None,
            region_layer_map={'IT': 'layer4'},
            extractor={'vision': _XArrayActivationsModel()},
            behavioral_readout_layer='layer4' if has_readout else None,
            **answer,
            **({'answer_owners': {}} if answer and has_readout else {}),
            **({'answer_owners': {'behavior:choice': ('readout', 'argmax')}}
               if has_readout and not answer else {}),
        )


    def test_readout_only_model_without_fitting_stimuli_raises(self):
        """A model built without answer= reads out; without fitting_stimuli it errors clearly."""
        m = self._make_dual_path_model(has_generation=False)
        ctx = TaskContext(response=Choice(['real', 'pseudo']), instruction="Is this real?",
                          fitting_stimuli=None)           # missing
        with pytest.raises(ExecutionNotFeasibleError, match="readout needs .*fitting_stimuli"):
            _answer(m, ctx)

    def test_unknown_task_type_raises(self):
        """A typo such as 'choice' must not silently run as a passive task."""
        m = self._make_dual_path_model()
        ctx = TaskContext(task_type='choice', label_set=['real', 'pseudo'],
                          instruction="Is this real?")
        with pytest.raises(ExecutionNotFeasibleError, match="task_type 'choice' states no response"):
            _answer(m, ctx)

    def test_passive_task_type_still_accepted(self):
        m = self._make_dual_path_model()
        session = stimulus_session(_stimuli(['image_file_name']), ['neural:IT'],
                                   TaskContext(task_type='passive'))
        m.participate(session)  # no error

    def test_no_dual_path_warning_without_readout_layer(self):
        """An answer-only model given fitting stimuli has one path, not two."""
        m = self._make_dual_path_model(has_readout=False)
        fitting = _stimuli(['image_file_name', 'label'])
        ctx = TaskContext(response=Choice(['real', 'pseudo']), instruction="Is this real?",
                          fitting_stimuli=fitting)
        with warnings.catch_warnings():
            warnings.simplefilter('error', UserWarning)
            _answer(m, ctx)

    def test_an_answer_and_a_readout_build_without_a_statement_and_declare_bare_behavior(self):
        model = BrainScoreModel(identifier='dual-path-model', region_layer_map={'IT': 'layer4'},
                                extractor={'vision': make_stub_extractor()}, behavioral_readout_layer='layer4',
                                answer=lambda stimulus_row, instruction, label_set: label_set[0])
        assert 'behavior' in model.out_channels


# -- BrainScoreModel state tests ---------------------------------------------

class TestBrainScoreModelState:

    def test_a_session_region_resolves_to_its_layer(self):
        extractor = _XArrayActivationsModel()
        m = BrainScoreModel(
            identifier='test',
            model=None,
            region_layer_map={'V1': 'layer1', 'IT': 'layer4'},
            extractor={'vision': extractor},
        )
        _record(m, ['image_file_name'], 'IT')
        assert extractor.call_args['layers'] == ['layer4']


# -- Extractor stubs that return an assembly -----------------------------------

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


class TestMultiModalityDispatch:

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
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            assembly, = _record(m, ['image_file_name'], 'IT')
        assert not [
            warning for warning in caught
            if "multiple supported modalities" in str(warning.message)
        ]
        assert text_proc.call_args is None
        assert 'modality' not in assembly.coords


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
        _record(m, ['image_file_name', 'sentence'], 'IT', 'language_network')
        # Vision wrapper only saw IT's layer
        assert act.call_args['layers'] == ['vision_model.layer.10']
        # Text wrapper only saw language_network's layer
        assert text_proc.call_args['layers'] == ['text_model.layer.20']

    def test_cross_tower_assembly_carries_modality_coord(self):
        m, act, text_proc = self._make_cross_tower_model()
        it, language = _record(m, ['image_file_name', 'sentence'], 'IT', 'language_network')
        assert set(it['modality'].values.tolist()) == {'vision'}
        assert set(language['modality'].values.tolist()) == {'text'}

    def test_without_region_modalities_the_model_is_refused(self):
        """A model with two extractors names the one each region is recorded
        from. Without it, a region would be recorded from whichever tower gets
        input, and pre-flight could not check it."""
        with pytest.raises(ValueError, match='names its extractor'):
            self._make_cross_tower_model(routed=False)

    def test_cross_tower_skips_wrapper_with_no_layers(self):
        """A session that requests only the vision region, on stimuli with
        both inputs, does not call the text wrapper."""
        m, act, text_proc = self._make_cross_tower_model()
        assembly, = _record(m, ['image_file_name', 'sentence'], 'IT')
        # Text wrapper not invoked
        assert text_proc.call_args is None
        # Output has only 'vision' modality entries
        modalities = set(assembly['modality'].values.tolist())
        assert modalities == {'vision'}

    def test_cross_tower_single_modality_path_unchanged(self):
        """Single-modality stimuli use the one tower they feed."""
        m, act, text_proc = self._make_cross_tower_model()
        assembly, = _record(m, ['image_file_name'], 'IT')
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
        extractor = _XArrayActivationsModel()
        m = BrainScoreModel('test', extractor={'video': extractor},
                            region_layer_map={'IT': 'layer'})
        _record(m, ['video_path'], 'IT')
        assert extractor.call_args['layers'] == ['layer']

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
