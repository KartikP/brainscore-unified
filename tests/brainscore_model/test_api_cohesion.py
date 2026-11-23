"""Transfer the same configuration between extraction, routing and sessions."""
import numpy as np
import pytest
import xarray as xr
from brainscore import BrainScoreModel, LayerSelector, CompositeSelector
from brainscore_core.streaming import Session, StreamEvent
from brainscore_core.streaming_helpers import stimulus_session
from brainscore_core.supported_data_standards.brainio.assemblies import NeuroidAssembly
from brainscore_core.supported_data_standards.brainio.stimuli import StimulusSet


class Extractor:
    # No identifier attribute: extraction is an explicit role, not a name heuristic.
    def __init__(self, value=1):
        self.value = value
        self.calls = []

    def __call__(self, stimuli, *, layers):
        self.calls.append(layers)
        return NeuroidAssembly(
            np.full((len(stimuli), 4 * len(layers)), self.value),
            dims=('presentation', 'neuroid'),
            coords={'layer': ('neuroid', np.repeat(layers, 4)),
                    'neuroid_id': ('neuroid', np.arange(4 * len(layers)))},
        )


def _record(model, columns, *regions):
    """The payload of each requested region, from one session on a one-row table."""
    stimuli = StimulusSet({'stimulus_id': ['s0'], **{column: ['input'] for column in columns}})
    stimuli.identifier = 'api-cohesion-' + '-'.join(columns)
    session = stimulus_session(stimuli, [f'neural:{region}' for region in regions])
    model.participate(session)
    return [session.collect(f'neural:{region}') for region in regions]


def _answer(model, task):
    """The behavior:choice answer of one session on a one-sentence table."""
    stimuli = StimulusSet({'stimulus_id': ['s0'], 'sentence': ['hello']})
    stimuli.identifier = 'api-cohesion-answer'
    session = stimulus_session(stimuli, ['behavior:choice'], task)
    model.participate(session)
    return session.collect('behavior:choice')


def test_composite_mapping_round_trip_preserves_units_and_routing():
    selector = CompositeSelector(layers=(('first', (1,)), ('second', (2,))), modality='text')
    model = BrainScoreModel(identifier='roundtrip', region_layer_map={'language': selector},
                            extractor={'text': Extractor()})
    model.region_layer_map = model.region_layer_map
    assert model.region_layer_map['language'] == selector
    result, = _record(model, ['sentence'], 'language')
    assert result.sizes['neuroid'] == 2
    assert result.layer.values.tolist() == ['first', 'second']
    assert result.unit_index.values.tolist() == [1, 2]


def test_mapping_replacement_drops_removed_routes_and_is_atomic():
    model = BrainScoreModel(identifier='routes',
                            region_layer_map={'IT': LayerSelector(name='x', modality='vision')},
                            extractor={'vision': Extractor()})
    before = model.region_layer_map
    with pytest.raises(ValueError, match='no extractor'):
        model.region_layer_map = {'A1': LayerSelector(name='x', modality='audio')}
    assert model.region_layer_map == before
    model.region_layer_map = {'V4': 'y'}
    assert dict(model.region_layer_map) == {'V4': 'y'}
    assert model.region_layer_selectors['V4'].modality is None


@pytest.mark.parametrize('modality,column', [('vision', 'image_path'), ('text', 'sentence'),
                                           ('audio', 'audio_path'), ('video', 'video_path')])
def test_every_modality_uses_same_extractor_signature(modality, column):
    extractor = Extractor()
    model = BrainScoreModel(identifier='same-api', region_layer_map={'region': 'layer'},
                            extractor={modality: extractor})
    result, = _record(model, [column], 'region')
    assert result.sizes['neuroid'] == 4
    assert extractor.calls == [['layer']]


def test_two_towers_with_same_relative_layer_route_by_selector():
    model = BrainScoreModel(identifier='towers', region_layer_map={
        'IT': LayerSelector(name='encoder.1', modality='vision'),
        'language': LayerSelector(name='encoder.1', modality='text'),
    }, extractor={'vision': Extractor(1), 'text': Extractor(2)})
    it, language = _record(model, ['image_path', 'sentence'], 'IT', 'language')
    assert it.values.tolist() == [[1] * 4]
    assert language.values.tolist() == [[2] * 4]



def test_common_session_collects_neural_output_without_helper_class():
    model = BrainScoreModel(identifier='session', region_layer_map={'IT': 'x'},
                            extractor={'vision': Extractor()})
    session = stimulus_session(StimulusSet({'image_path': ['x']}), ('neural:IT',))
    model.participate(session)
    assert session.collect().shape == (1, 4)


def test_common_session_keeps_catalog_events_and_refuses_unknown_channels():
    session = Session([StreamEvent('text', 'hello', 0)])
    session.respond(StreamEvent('motor', 7, 0))          # a harness-defined payload
    assert session.collect()[0].payload == 7
    with pytest.raises(ValueError, match='failed the channel catalog'):
        session.respond(StreamEvent('custom', 7, 0))     # not in the catalog


def test_explicit_source_is_independent_of_behavioral_task_configuration():
    from brainscore_core.contract import TaskContext
    from brainscore_core.supported_data_standards.brainio.assemblies import BehavioralAssembly
    seen = []
    def generate(model, stimuli, task):
        seen.extend(stimuli.sentence)
        return BehavioralAssembly(np.ones((len(stimuli),)), dims=['presentation'],
                                  coords={'stimulus_id': ('presentation', list(stimuli.stimulus_id))})
    task = TaskContext(task_type='label', instruction='choose', label_set=['yes'],
                       metadata={'stimulus_set': StimulusSet({'sentence': ['wrong']})})
    explicit = StimulusSet({'sentence': ['actual'], 'stimulus_id': ['1']})
    # Check the driver directly, so this test isolates task/source composition.
    from brainscore_core.streaming_helpers import drive_behavior_session
    from types import SimpleNamespace
    subject = SimpleNamespace(start_task=lambda context: None,
                              process=lambda stimuli: generate(None, stimuli, task))
    session = stimulus_session(explicit, ['behavior'], task=task)
    drive_behavior_session(subject, session)
    assert seen == ['actual']
    assert session.collect().shape == (1,)


def test_shared_extractor_resets_once_across_modalities():
    extractor = Extractor()
    resets = []
    extractor.reset = lambda: resets.append(True)
    model = BrainScoreModel(identifier='shared', extractor={'vision': extractor, 'text': extractor})
    model.reset()
    assert resets == [True]


def test_variadic_legacy_wrapper_is_normalized_without_identifier_discovery():
    class Wrapper:
        def __call__(self, *args, **kwargs):
            return Extractor()(*args, **kwargs)
    model = BrainScoreModel(identifier='compatibility', extractor={'vision': Wrapper()},
                            region_layer_map={'IT': 'layer'})
    result, = _record(model, ['image_path'], 'IT')
    assert result.shape == (1, 4)


def test_neural_events_work_without_a_stimulus_table():
    model = BrainScoreModel(identifier='events', region_layer_map={'language': 'x'},
                            extractor={'text': Extractor()})
    session = Session([StreamEvent(channel='text', payload='hello', t_ms=0)], ['neural:language'])
    model.participate(session)
    assert session.collect().shape == (1, 4)


def test_generation_declares_instruction_input_for_replay():
    model = BrainScoreModel(identifier='answers', answer=lambda *args: 'yes')
    assert 'instruction' in model.in_channels


@pytest.mark.parametrize('streaming', [False, True])
def test_common_session_collects_batch_and_streamed_assemblies(streaming):
    model = BrainScoreModel(identifier='delivery', region_layer_map={'language': 'x'},
                            extractor={'text': Extractor()})
    session = stimulus_session(StimulusSet({'sentence': ['one', 'two'], 'stimulus_id': ['1', '2']}),
                            ['neural:language'], streaming=streaming)
    model.participate(session)
    assert session.collect().shape == (2, 4)



def test_external_selector_routing_does_not_require_a_dataclass():
    from brainscore import UnitSelector
    class ExternalSelector(UnitSelector):
        modality = 'text'

        @property
        def layer_path(self):
            return 'custom.layer'
    selector = ExternalSelector()
    model = BrainScoreModel(
        identifier='external',
        region_layer_map={'region': selector},
        extractor={'text': Extractor(), 'vision': Extractor()},
    )
    assert isinstance(model.region_layer_map['region'], ExternalSelector)
    model.region_layer_map = model.region_layer_map
    assert model.region_layer_selectors['region'].modality == 'text'
    assert model.output_inputs == {'neural:region': frozenset({'text'})}
    result, = _record(model, ['sentence'], 'region')
    assert result.shape == (1, 4)


def test_generation_inputs_need_no_placeholder_extractors():
    from brainscore_core.contract import TaskContext
    model = BrainScoreModel(
        identifier='output-only', in_channels={'text'},
        answer=lambda stimulus_row, instruction, label_set: label_set[0],
    )
    assert model.in_channels == {'text', 'instruction', 'control'}
    assert model.region_layer_map == {}
    from brainscore_core.streaming_helpers import behavioral_response
    stimuli = StimulusSet({'stimulus_id': ['s0'], 'sentence': ['hello']})
    stimuli.identifier = 'output-only'
    from brainscore_core.responses import Choice
    result = behavioral_response(model, stimuli, TaskContext(instruction='Choose yes.',
                                                             response=Choice(['yes', 'no'])))
    assert result.values.tolist() == [[1.0, 0.0]]


@pytest.mark.parametrize('parts', ['answer and readout', 'readout'])
def test_the_model_parts_choose_the_behavioral_path(parts):
    """The path is a property of the model: a model built without answer= uses its readout."""
    import warnings
    from brainscore_core.contract import TaskContext
    from brainscore_core.responses import Choice
    answer = {'answer': lambda **kwargs: 'yes'} if parts == 'answer and readout' else {}
    model = BrainScoreModel(
        identifier='dual-path', in_channels={'text'}, behavioral_readout_layer='layer',
        extractor={'text': Extractor()}, **answer,
        answer_owners={} if answer
                           else {'behavior:choice': ('readout', 'argmax')},
    )
    fitting = StimulusSet({'stimulus_id': ['a', 'b'], 'sentence': ['hello', 'bye'],
                           'label': ['yes', 'no']})
    fitting.identifier = 'dual-path-fitting'
    with warnings.catch_warnings(record=True) as notices:
        warnings.simplefilter('always')
        result = _answer(model, TaskContext(instruction='Choose yes.', response=Choice(['yes', 'no']),
                                            fitting_stimuli=fitting))
    assert not [notice for notice in notices if issubclass(notice.category, UserWarning)]
    assert result.attrs['response']['method'] == ('generation' if answer else 'readout')


def test_named_readout_layer_without_extractor_is_not_a_second_path():
    import warnings
    from brainscore_core.contract import TaskContext
    from brainscore_core.responses import Choice
    model = BrainScoreModel(
        identifier='answer-only', in_channels={'text'},
        answer=lambda **kwargs: 'yes', behavioral_readout_layer='features',
    )
    fitting = StimulusSet({'stimulus_id': ['a'], 'sentence': ['hello'], 'label': ['yes']})
    fitting.identifier = 'answer-only-fitting'
    with warnings.catch_warnings():
        warnings.simplefilter('error', UserWarning)
        result = _answer(model, TaskContext(instruction='Choose yes.', response=Choice(['yes', 'no']),
                                            fitting_stimuli=fitting))
    assert result.attrs['response']['method'] == 'generation'


def test_channel_registration_policy_differs_only_for_exploration():
    from brainscore_core.compatibility import (
        validate_channels, CompatibilityWarning,
    )
    from tests.unregistered import unregistered_family
    family = unregistered_family()
    with pytest.warns(CompatibilityWarning, match='register_channel'):
        validate_channels({family}, 'input', 'experiment', allow_unregistered=True)
    with pytest.raises(ValueError, match=family):
        validate_channels({family}, 'input', 'benchmark')
    with pytest.raises(ValueError, match='registry marks'):
        validate_channels({'behavior'}, 'input', 'experiment', allow_unregistered=True)


