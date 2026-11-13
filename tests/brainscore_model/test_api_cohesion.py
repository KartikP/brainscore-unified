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


def test_composite_mapping_round_trip_preserves_units_and_routing():
    selector = CompositeSelector(layers=(('first', (1,)), ('second', (2,))), modality='text')
    model = BrainScoreModel(identifier='roundtrip', region_layer_map={'language': selector},
                            extractor={'text': Extractor()})
    model.region_layer_map = model.region_layer_map
    assert model.region_layer_map['language'] == selector
    model.start_recording('language')
    result = model.process(StimulusSet({'sentence': ['hello']}))
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
    model.start_recording('region')
    assert model.process(StimulusSet({column: ['input']})).sizes['neuroid'] == 4
    assert extractor.calls == [['layer']]


def test_two_towers_with_same_relative_layer_route_by_selector():
    model = BrainScoreModel(identifier='towers', region_layer_map={
        'IT': LayerSelector(name='encoder.1', modality='vision'),
        'language': LayerSelector(name='encoder.1', modality='text'),
    }, extractor={'vision': Extractor(1), 'text': Extractor(2)})
    model.start_recording(['IT', 'language'])
    result = model.process(StimulusSet({'image_path': ['image'], 'sentence': ['text']}), multi_modality=True)
    assert result.sel(neuroid=result.region == 'IT').values.tolist() == [[1] * 4]
    assert result.sel(neuroid=result.region == 'language').values.tolist() == [[2] * 4]


@pytest.mark.parametrize('target', ['raw', ['raw']])
def test_raw_layer_has_same_meaning_alone_or_in_list(target):
    model = BrainScoreModel(identifier='raw', extractor={'text': Extractor()})
    with pytest.warns(UserWarning, match='raw layer'):
        model.start_recording(target)
    assert model.process(StimulusSet({'sentence': ['x']})).layer.values.tolist() == ['raw'] * 4


def test_common_session_collects_neural_output_without_helper_class():
    model = BrainScoreModel(identifier='session', region_layer_map={'IT': 'x'},
                            extractor={'vision': Extractor()})
    session = stimulus_session(StimulusSet({'image_path': ['x']}), ('neural:IT',))
    model.participate(session)
    assert session.collect().shape == (1, 4)


def test_common_session_keeps_custom_events():
    session = Session([StreamEvent('text', 'hello', 0)])
    session.respond(StreamEvent('custom', 7, 0))
    assert session.collect()[0].payload == 7


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
    model.start_recording('IT')
    assert model.process(StimulusSet({'image_path': ['x']})).shape == (1, 4)


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
                            ['neural:language'])
    session.streaming = streaming
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
    model.start_recording('region')
    assert model.process(StimulusSet({'sentence': ['x']})).shape == (1, 4)


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
    result = behavioral_response(model, stimuli, TaskContext(
        task_type='probabilities', instruction='Choose yes.', label_set=['yes'],
    ))
    assert result.values.tolist() == [[1.0]]
    # Without a session, process() has no extractor to answer with.
    with pytest.raises(NotImplementedError, match='no activation extractor'):
        model.process(StimulusSet({'stimulus_id': ['s0'], 'sentence': ['hello']}))


@pytest.mark.parametrize('parts', ['answer and readout', 'readout'])
def test_the_model_parts_choose_the_behavioral_path(monkeypatch, parts):
    """The path is a property of the model: a model built without answer= uses its readout."""
    import warnings
    from brainscore_core.contract import TaskContext
    answer = {'answer': lambda **kwargs: 'yes'} if parts == 'answer and readout' else {}
    model = BrainScoreModel(
        identifier='dual-path', in_channels={'text'}, behavioral_readout_layer='layer',
        extractor={'text': Extractor()}, **answer,
    )
    fitted = []
    monkeypatch.setattr(model._behavioral, 'fit_behavioral_readout', lambda stimuli: fitted.append(stimuli))
    stimuli = StimulusSet({'sentence': ['hello']})
    with warnings.catch_warnings(record=True) as notices:
        warnings.simplefilter('always')
        model.start_task(TaskContext(
            task_type='probabilities', instruction='Choose yes.', label_set=['yes'],
            fitting_stimuli=stimuli,
        ))
    assert len(notices) == (1 if answer else 0)
    if notices:
        assert 'not fitted' in str(notices[0].message)
    assert bool(fitted) == (not answer)
    assert model._behavioral.response_path == ('generation' if answer else 'readout')


def test_named_readout_layer_without_extractor_is_not_a_second_path():
    import warnings
    from brainscore_core.contract import TaskContext
    model = BrainScoreModel(
        identifier='answer-only', in_channels={'text'},
        answer=lambda **kwargs: 'yes', behavioral_readout_layer='features',
    )
    with warnings.catch_warnings():
        warnings.simplefilter('error', UserWarning)
        model.start_task(TaskContext(
            task_type='probabilities', instruction='Choose yes.', label_set=['yes'],
            fitting_stimuli=StimulusSet({'sentence': ['hello']}),
        ))
    assert model._behavioral.response_path == 'generation'


def test_channel_registration_policy_differs_only_for_exploration():
    from brainscore_core.compatibility import (
        validate_channels, CompatibilityError, CompatibilityWarning,
    )
    with pytest.warns(CompatibilityWarning, match='register_channel'):
        validate_channels({'custom_signal'}, 'input', 'experiment', allow_unregistered=True)
    with pytest.raises(CompatibilityError, match='unknown channel'):
        validate_channels({'custom_signal'}, 'input', 'benchmark')
    with pytest.raises(CompatibilityError, match='registry marks'):
        validate_channels({'behavior'}, 'input', 'experiment', allow_unregistered=True)


