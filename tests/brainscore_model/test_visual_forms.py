"""A subject declares the visual forms it takes natively: stills, clips, or both."""
import numpy as np
import pytest

from brainscore.brainscore_model import BrainScoreModel
from brainscore.selection import LayerSelector
from brainscore_core.streaming_helpers import neural_response
from brainscore_core.supported_data_standards.brainio.assemblies import NeuroidAssembly
from brainscore_core.supported_data_standards.brainio.stimuli import StimulusSet

pytestmark = pytest.mark.unit


def _extractor(value):
    def extract(stimuli, layers, **kwargs):
        ids = list(stimuli['stimulus_id'])
        return NeuroidAssembly(np.full((len(ids), 2), value), dims=['presentation', 'neuroid'],
                               coords={'stimulus_id': ('presentation', ids),
                                       'neuroid_id': ('neuroid', ['n0', 'n1']),
                                       'layer': ('neuroid', [layers[0]] * 2)})
    return extract


def _stimuli(column):
    stimuli = StimulusSet({'stimulus_id': ['s0', 's1'], column: ['a', 'b']})
    stimuli.identifier = f'forms-{column}'
    return stimuli


@pytest.mark.parametrize('keys, forms', [(('vision',), {'vision:image'}),
                                         (('video',), {'vision:video'}),
                                         (('vision', 'video'), {'vision:image', 'vision:video'})])
def test_forms_follow_the_extractor_keys(keys, forms):
    model = BrainScoreModel('m', extractor={key: _extractor(1.0) for key in keys})
    assert 'vision' not in model.in_channels
    assert {c for c in model.in_channels if c.startswith('vision:')} == forms


def test_a_subject_with_both_forms_takes_each_stimulus_natively():
    model = BrainScoreModel('m', extractor={'vision': _extractor(1.0), 'video': _extractor(2.0)},
                            region_layer_map={'IT': LayerSelector('blocks.1', modality='vision')})
    assert neural_response(model, _stimuli('image_path'), record='IT').values.max() == 1.0
    assert neural_response(model, _stimuli('video_path'), record='IT').values.max() == 2.0


def test_models_without_an_extractor_declare_the_inputs_they_accept():
    model = BrainScoreModel('judge', supported_modalities={'vision'},
                            answer=lambda stimulus_row, instruction, label_set: label_set[0])
    assert 'vision:image' in model.in_channels


class _Recorder:
    """record(layers) stand-in; a clip recording has two time steps."""

    def __init__(self, value, timed):
        self.value, self.timed, self.hooks = value, timed, []

    def __call__(self, stimuli, layers, **kwargs):
        raise AssertionError('a task presents through reply, not the extractor')

    def record(self, layers):
        from contextlib import contextmanager

        @contextmanager
        def recording():
            owner = self

            class Take:
                @staticmethod
                def take():
                    coords = {'stimulus_id': ('presentation', ['?']),
                              'neuroid_id': ('neuroid', ['n0', 'n1']),
                              'layer': ('neuroid', [layers[0]] * 2)}
                    if not owner.timed:
                        return NeuroidAssembly(np.full((1, 2), owner.value), coords=coords,
                                               dims=['presentation', 'neuroid'])
                    coords.update(time_bin_start_ms=('time_bin', [0.0, 1000.0]),
                                  time_bin_end_ms=('time_bin', [1000.0, 2000.0]))
                    values = np.stack([np.full(2, owner.value), np.full(2, owner.value + 1)])[None]
                    return NeuroidAssembly(values, coords=coords,
                                           dims=['presentation', 'time_bin', 'neuroid'])
            yield Take()
        return recording()


def test_a_clip_recorded_during_a_task_uses_the_clip_extractor_and_keeps_time():
    from brainscore_core.model_interface import TaskContext
    from brainscore_core.responses import Choice
    from brainscore_core.streaming_helpers import stimulus_session
    model = BrainScoreModel('m', extractor={'vision': _Recorder(1.0, False), 'video': _Recorder(5.0, True)},
                            region_layer_map={'IT': LayerSelector('blocks.1', modality='vision')},
                            reply=lambda prompt: 'left')
    task = TaskContext(instruction='Which way?', response=Choice(['left', 'right']))
    session = stimulus_session(_stimuli('video_path'), ['behavior:choice', 'neural:IT'], task_context=task)
    model.participate(session)
    recorded = session.collect('neural:IT')
    assert recorded.dims == ('presentation', 'time_bin', 'neuroid')
    np.testing.assert_array_equal(recorded.values[0, :, 0], [5.0, 6.0])
    assert list(recorded['time_bin_start_ms'].values) == [0.0, 1000.0]


def test_a_multi_image_trial_is_recorded_per_image():
    from contextlib import contextmanager
    from brainscore_core.model_interface import TaskContext
    from brainscore_core.responses import Choice
    from brainscore_core.streaming_helpers import stimulus_session

    class PerImage:
        def __call__(self, stimuli, layers, **kwargs):
            raise AssertionError('recorded during the reply')

        @contextmanager
        def record(self, layers):
            class Take:
                @staticmethod
                def take():
                    values = np.arange(6, dtype=float).reshape(1, 3, 2)   # 3 images, 2 units
                    return NeuroidAssembly(values, dims=['presentation', 'part', 'neuroid'], coords={
                        'stimulus_id': ('presentation', ['?']), 'part_index': ('part', [0, 1, 2]),
                        'neuroid_id': ('neuroid', ['n0', 'n1']), 'layer': ('neuroid', [layers[0]] * 2)})
            yield Take()

    model = BrainScoreModel('m', extractor={'vision': PerImage()},
                            region_layer_map={'IT': LayerSelector('blocks.1', modality='vision')},
                            reply=lambda prompt: 'yes')
    trials = StimulusSet({'stimulus_id': ['t1'], 'rules': ['r.png'], 'word_1': ['a.png'], 'word_2': ['b.png']})
    trials.identifier = 'sequence'
    task = TaskContext(response=Choice(['yes', 'no']), present=('image:rules', 'image:word_1', 'image:word_2'))
    session = stimulus_session(trials, ['behavior:choice', 'neural:IT'], task_context=task)
    model.participate(session)
    recorded = session.collect('neural:IT')
    assert recorded.dims == ('presentation', 'part', 'neuroid')
    assert list(recorded['part_name'].values) == ['rules', 'word_1', 'word_2']


@pytest.mark.parametrize('keys, forms, required', [
    (('vision',), {'vision:image'}, set()),
    (('video',), {'vision:video'}, {'vision:video'}),
    (('vision', 'video'), {'vision:image', 'vision:video'}, set())])
def test_declared_inputs_are_exactly_the_forms_taken(keys, forms, required):
    model = BrainScoreModel('m', extractor={key: _extractor(1.0) for key in keys},
                            required_modalities={'video'} if required else None)
    assert {c for c in model.in_channels if c.startswith('vision')} == forms
    assert 'vision' not in model.in_channels
    assert model.required_channels == required


@pytest.mark.parametrize('keys, compatible', [(('vision',), True), (('video',), True),
                                              (('vision', 'video'), True), (('text',), False)])
def test_a_benchmark_declaring_both_forms_is_compatible_with_each(keys, compatible):
    from types import SimpleNamespace
    from brainscore_core import check_compatibility
    from brainscore_core.compatibility import CompatibilityError
    model = BrainScoreModel('m', region_layer_map={'IT': 'layer'},
                            extractor={key: _extractor(1.0) for key in keys})
    benchmark = SimpleNamespace(identifier='both-forms',
                                required_input_channels={'vision:image', 'vision:video'},
                                requested_output_channels={'neural:IT'})
    if compatible:
        check_compatibility(model, benchmark)
    else:
        with pytest.raises(CompatibilityError):
            check_compatibility(model, benchmark)


def test_a_benchmark_with_a_bare_vision_channel_is_refused():
    from types import SimpleNamespace
    from brainscore_core import check_compatibility
    from brainscore_core.compatibility import CompatibilityError
    model = BrainScoreModel('m', region_layer_map={'IT': 'layer'}, extractor={'vision': _extractor(1.0)})
    benchmark = SimpleNamespace(identifier='bare', required_input_channels={'vision'},
                                requested_output_channels={'neural:IT'})
    with pytest.raises(CompatibilityError):
        check_compatibility(model, benchmark)
