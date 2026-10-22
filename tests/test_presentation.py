"""Optional presentation changes sensory inputs without changing session semantics."""
from contextlib import contextmanager
import numpy as np
import pytest
from brainscore.presentation import Presentation, Screen, Speaker
from brainscore_core.streaming import BufferedSession, StreamEvent
from brainscore_core.contract import Subject
from brainscore.experiments import Experiment, SessionProtocol, RecordInputsOutputs, replay_sessions, compare_outputs
from brainscore.experiments.replay import read_events

pytestmark = pytest.mark.unit


def test_screen_places_contrast_adjusted_image_without_mutating_source():
    image = np.array([[0., 1.]], dtype=np.float32)
    canvas = Screen(height=3, width=4, top=1, left=1, contrast=0.5)(image)
    expected = np.full((3, 4), 0.5, dtype=np.float32)
    expected[1, 1:3] = [0.25, 0.75]
    np.testing.assert_array_equal(canvas, expected)
    np.testing.assert_array_equal(image, [[0., 1.]])


@pytest.mark.parametrize('image', [np.zeros((2, 2), dtype=np.uint8), np.array([[np.nan]]),
                                   np.array([[2.]]), np.ones((3, 3)), np.empty((0, 1))])
def test_screen_rejects_invalid_or_oversized_inputs(image):
    with pytest.raises(ValueError):
        Screen(height=2, width=2)(image)


def test_speaker_preserves_audio_timebase_and_metadata():
    audio = {'samples': np.array([[1., -1.], [0.5, -0.5]]), 'sample_rate': 16000,
             'labels': ['original']}
    output = Speaker(gain=0.25)(audio)
    np.testing.assert_array_equal(output['samples'], [[0.25, -0.25], [0.125, -0.125]])
    assert output['sample_rate'] == 16000
    output['labels'].append('changed')
    assert audio['labels'] == ['original']
    np.testing.assert_array_equal(audio['samples'], [[1., -1.], [0.5, -0.5]])


@pytest.mark.parametrize('audio', [{'samples': [1.]}, {'samples': [1.], 'sample_rate': 0},
                                  {'samples': [np.inf], 'sample_rate': 1},
                                  {'samples': [2.], 'sample_rate': 1},
                                  {'samples': [1], 'sample_rate': 1}])
def test_speaker_rejects_invalid_audio(audio):
    with pytest.raises(ValueError):
        Speaker()(audio)


@pytest.mark.parametrize('make', [lambda: Speaker(gain=-1), lambda: Speaker(gain=2),
                                  lambda: Screen(height=0, width=2),
                                  lambda: Screen(height=2, width=2, contrast=2),
                                  lambda: Screen(height=True, width=2)])
def test_invalid_configuration_fails_before_a_session(make):
    with pytest.raises(ValueError):
        make()


def test_ordered_composition_records_settings_and_preserves_event_identity():
    event = StreamEvent(channel='vision', payload=np.ones((1, 1)), t_ms=150,
                        meta={'stimulus_id': 'one'})
    present = Presentation(components={'vision': [Screen(height=2, width=2),
                                                 Screen(height=3, width=3, top=1, left=1)]})
    output = present(event)
    assert output.payload[1, 1] == 1
    assert output.payload.shape == (3, 3)
    assert output.channel == event.channel and output.t_ms == 150
    assert output.meta['stimulus_id'] == 'one'
    assert len(output.meta['presentation']) == 2
    assert event.meta == {'stimulus_id': 'one'} and event.payload.shape == (1, 1)


def test_external_component_needs_no_registration_or_subclass():
    class Invert:
        def describe(self):
            return {'component': 'Invert', 'version': 1}
        def __call__(self, image):
            return 1 - image
    event = StreamEvent(channel='vision', payload=np.array([[0.2]]), t_ms=None)
    output = Presentation(components={'vision': [Invert()]})(event)
    np.testing.assert_allclose(output.payload, [[0.8]])
    assert output.meta['presentation'][0]['component'] == 'Invert'


def test_unconfigured_channels_are_independent_and_unchanged():
    event = StreamEvent(channel='text', payload={'text': ['hello']}, t_ms=None)
    output = Presentation(components={'audio': [Speaker()]})(event)
    assert output == event
    output.payload['text'].append('changed')
    assert event.payload['text'] == ['hello']


def test_record_and_replay_use_presented_inputs_without_applying_transforms_twice(tmp_path):
    class Observer(Subject):
        identifier = 'presentation-observer'
        in_channels = {'vision', 'audio'}
        out_channels = {'behavior'}
        def interact(self, session):
            while (event := session.next_input()) is not None:
                value = event.payload if event.channel == 'vision' else event.payload['samples']
                session.respond(StreamEvent(channel='behavior', payload=float(np.sum(value)), t_ms=event.t_ms))

    present = Presentation(components={'vision': [Screen(height=2, width=2, contrast=0.5)],
                                       'audio': [Speaker(gain=0.25)]})
    @contextmanager
    def factory(trial):
        yield BufferedSession(inputs=map(present, [
            StreamEvent(channel='vision', payload=np.ones((1, 1)), t_ms=0),
            StreamEvent(channel='audio', payload={'samples': np.ones(2), 'sample_rate': 16000}, t_ms=0),
        ]))
    baseline = Experiment(subject=Observer(), protocol=SessionProtocol(
        identifier='presentation', session_factory=factory,
        input_channels=('vision', 'audio'), output_channels=('behavior',)),
        tools=[RecordInputsOutputs()], output_dir=tmp_path/'baseline').run()
    replay = Experiment(subject=Observer(), protocol=replay_sessions(baseline.directory),
                        tools=[RecordInputsOutputs()], output_dir=tmp_path/'replay').run()
    assert compare_outputs(baseline.directory, replay.directory)['equal']
    inputs = [entry['payload'] for entry in read_events(baseline.directory) if entry['kind'] == 'input']
    assert inputs[0].payload.shape == (2, 2)
    assert inputs[0].meta['presentation'][0]['contrast'] == 0.5
    np.testing.assert_array_equal(inputs[1].payload['samples'], [0.25, 0.25])


def test_rgb_channels_and_dtype_are_preserved():
    image = np.array([[[0., 0.5, 1.]]], dtype=np.float32)
    output = Screen(height=2, width=2, background=0., contrast=0.5)(image)
    assert output.shape == (2, 2, 3) and output.dtype == np.float32
    np.testing.assert_array_equal(output[0, 0], [0., 0.25, 0.5])


def test_failed_component_cannot_mutate_original_event():
    class FailingComponent:
        def describe(self):
            return {'component': 'failure'}
        def __call__(self, image):
            image[:] = 0
            raise RuntimeError('failed')
    event = StreamEvent(channel='vision', payload=np.ones((1, 1)), t_ms=0)
    with pytest.raises(RuntimeError, match='failed'):
        Presentation(components={'vision': [FailingComponent()]})(event)
    np.testing.assert_array_equal(event.payload, [[1.]])
    assert event.meta == {}


def test_component_settings_must_be_recordable_before_execution():
    class InvalidDescription:
        def describe(self):
            return {'gain': float('nan')}
        def __call__(self, payload):
            pytest.fail('must reject settings before running the component')
    with pytest.raises(ValueError):
        Presentation(components={'audio': [InvalidDescription()]})
    with pytest.raises(TypeError):
        Presentation(components={'audio': [lambda payload: payload]})
