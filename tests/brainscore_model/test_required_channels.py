"""``required_channels``: leave it out unless one network cannot run without some inputs.

The constructor checks only that each listed channel is declared (as before the
registration clean-up); whether the network really needs them is the
registrant's claim. A model made of separate extractors requires nothing, so a
picture-with-sound model takes part in a picture benchmark for its picture regions.
"""
import numpy as np
import pytest

from brainscore import BrainScoreModel, LayerSelector
from brainscore.capabilities import Capability, capability_registry, register_capability
from brainscore_core.compatibility import CompatibilityError, check_compatibility
from brainscore_core.streaming import StreamEvent
from brainscore_core.streaming import Session


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


def test_a_picture_and_sound_model_takes_part_in_a_picture_benchmark():
    model = _picture_and_sound()
    assert model.required_channels == set()
    match = check_compatibility(model, Benchmark({'vision:image'}, {'neural:V4'}))
    assert match.kind == 'full'
    with pytest.raises(CompatibilityError, match='only from the inputs'):
        check_compatibility(model, Benchmark({'vision:image'}, {'neural:A1'}))


def test_a_reply_that_needs_the_picture_keeps_its_requirement_beside_a_vision_extractor():
    # The reply reads the picture and the text of each trial together and cannot answer
    # without the picture. A vision extractor beside it does not change that claim.
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


def test_a_requirement_outside_the_declared_inputs_is_refused():
    with pytest.raises(ValueError, match='not all declared'):
        BrainScoreModel('m', extractor={'vision': Tower('a')}, required_channels={'audio'})
    with pytest.raises(ValueError, match='not all declared'):
        BrainScoreModel('phantom', required_channels={'text'})


def test_a_joint_extractor_requires_all_its_inputs_and_preflight_needs_them():
    joint = Tower('fusion.3')
    model = BrainScoreModel('joint', region_layer_map={'STS': 'fusion.3'},
                            extractor={('video', 'audio'): joint},
                            required_channels={'vision:video', 'audio'})
    assert model.required_channels == {'vision:video', 'audio'}
    with pytest.raises(CompatibilityError, match='hard-requires'):
        check_compatibility(model, Benchmark({'vision:video'}, {'neural:STS'}))


@pytest.fixture
def reader_capability():
    previous = dict(capability_registry)

    class Reader(Capability):
        """Reads text and pictures of each trial; answers on behavior."""
        identifier = 'test-reader'
        input_channels = {'text', 'vision:image'}
        output_channels = {'behavior'}

        def enabled_for(self, subject):
            return bool(subject.capability_config.get(self.identifier))

        def supports_session(self, subject, channels):
            return set(channels) == {'behavior'}

        def participate(self, subject, session):
            while (event := session.next_input()) is not None:
                session.respond(StreamEvent('behavior', np.array([1.0, 0.0]), event.t_ms))

    register_capability(Reader())
    yield
    capability_registry.clear()
    capability_registry.update(previous)


def test_a_capability_that_takes_several_inputs_can_require_one_and_runs(reader_capability):
    model = BrainScoreModel('reader', in_channels={'text', 'vision:image'},
                            required_channels={'text'},
                            capability_config={'test-reader': True})
    assert model.required_channels == {'text'}
    session = Session([StreamEvent('text', 'a word', 0)], ('behavior',))
    model.participate(session)
    assert len(session.collect('behavior')) == 1
