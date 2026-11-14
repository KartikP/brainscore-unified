"""``required_channels`` is a fact only of one wired part that takes several inputs.

Each separate extractor runs on its own input, and pre-flight already needs one
declared input, so a hand-typed set over separate extractors could only block
pairs that would run. The constructor refuses it.
"""
import pytest

from brainscore import BrainScoreModel, LayerSelector
from brainscore_core.compatibility import CompatibilityError, check_compatibility


class Tower:
    """An extractor that lists its layers and is never run here."""

    def __init__(self, *layers):
        self._layers = layers

    def layer_names(self):
        return list(self._layers)

    def __call__(self, stimuli, layers):
        raise AssertionError('pre-flight must not run an extractor')


class Benchmark:
    identifier = 'picture-benchmark'
    trial_memory = 'none'

    def __init__(self, inputs, outputs):
        self.required_input_channels = frozenset(inputs)
        self.requested_output_channels = frozenset(outputs)


def _picture_and_sound():
    return BrainScoreModel('picture-and-sound',
                           region_layer_map={'V4': LayerSelector('blocks.2', modality='vision'),
                                             'A1': LayerSelector('encoder.6', modality='audio')},
                           extractor={'vision': Tower('blocks.2'), 'audio': Tower('encoder.6')})


def test_a_set_over_separate_extractors_is_refused():
    with pytest.raises(ValueError, match='one part'):
        BrainScoreModel('m', extractor={'vision': Tower('a'), 'audio': Tower('b')},
                        required_channels={'vision:image', 'audio'})


@pytest.mark.parametrize('key, channel', [('text', 'text'), ('video', 'vision:video'),
                                          ('vision', 'vision:image')])
def test_a_set_that_repeats_the_input_of_one_extractor_is_refused(key, channel):
    with pytest.raises(ValueError, match='only repeats'):
        BrainScoreModel('m', extractor={key: Tower('a')}, required_channels={channel})


def test_the_parts_without_an_extractor_are_one_part():
    with pytest.raises(ValueError, match='only repeats'):
        BrainScoreModel('m', reply=lambda prompt: '', in_channels={'text'},
                        required_channels={'text'})
    model = BrainScoreModel('m', reply=lambda prompt: '', in_channels={'text', 'vision:image'},
                            required_channels={'vision:image'})
    assert model.required_channels == {'vision:image'}
    with pytest.raises(ValueError, match='one part'):
        BrainScoreModel('m', reply=lambda prompt: '', in_channels={'text'},
                        extractor={'vision': Tower('a')},
                        required_channels={'vision:image', 'text'})


def test_one_extractor_that_takes_several_inputs_can_require_them():
    joint = Tower('fusion.3')
    model = BrainScoreModel('joint', region_layer_map={'STS': 'fusion.3'},
                            extractor={('video', 'audio'): joint},
                            required_channels={'vision:video', 'audio'})
    assert model.required_channels == {'vision:video', 'audio'}
    with pytest.raises(CompatibilityError, match='hard-requires'):
        check_compatibility(model, Benchmark({'vision:video'}, {'neural:STS'}))


def test_a_picture_and_sound_model_takes_part_in_a_picture_benchmark():
    model = _picture_and_sound()
    assert model.required_channels == set()
    match = check_compatibility(model, Benchmark({'vision:image'}, {'neural:V4'}))
    assert match.kind == 'full'
    with pytest.raises(CompatibilityError, match='only from the inputs'):
        check_compatibility(model, Benchmark({'vision:image'}, {'neural:A1'}))


def test_a_reply_that_needs_the_picture_keeps_its_requirement_beside_a_vision_extractor():
    # The reply reads the picture and the text of each trial together; it cannot answer
    # without the picture, so it requires it. The vision extractor, a separate part that
    # also takes pictures, does not take that input away from the reply.
    model = BrainScoreModel('reader', reply=lambda prompt: 'yes',
                            in_channels={'text', 'vision:image'},
                            extractor={'vision': Tower('blocks.2')},
                            required_channels={'vision:image'})
    assert model.required_channels == {'vision:image'}


def test_a_joint_extractor_requires_all_its_inputs_beside_a_separate_sound_extractor():
    joint, separate = Tower('fusion.3'), Tower('encoder.6')
    model = BrainScoreModel('joint-and-sound',
                            region_layer_map={'STS': LayerSelector('fusion.3', modality='video')},
                            extractor={('video', 'audio', 'text'): joint, 'audio': separate},
                            required_channels={'vision:video', 'audio', 'text'})
    assert model.required_channels == {'vision:video', 'audio', 'text'}


def test_in_channels_without_a_part_cannot_require_an_input():
    with pytest.raises(ValueError, match='one wired part'):
        BrainScoreModel('phantom', in_channels={'text', 'vision:image'},
                        required_channels={'vision:image'})


def test_a_policy_is_one_part_with_its_inputs():
    model = BrainScoreModel('agent', policy=lambda row: {}, policy_in_channels=('observation', 'text'),
                            required_channels={'observation'})
    assert model.required_channels == {'observation'}
