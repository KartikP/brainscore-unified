"""The constructor takes its parts as keywords; each part declares its channels."""
import pytest

from brainscore import BrainScoreModel


def _f(*a, **k):
    return None


def test_parts_after_the_region_map_are_not_positional():
    with pytest.raises(TypeError):
        BrainScoreModel('id', None, {}, {'vision': _f})


def test_extractor_is_a_mapping():
    with pytest.raises(TypeError, match='mapping'):
        BrainScoreModel('id', extractor=_f)


def test_each_part_declares_its_channels():
    model = BrainScoreModel('id', policy=_f, intervention=_f, answer=_f,
                            in_channels={'text'})
    assert {'motor', 'perturbation', 'behavior:choice'} <= model.out_channels
    assert {'text', 'instruction', 'observation', 'lesion'} <= model.in_channels


def test_identifier_only_construction_works():
    # The docstring says only identifier is always required.
    model = BrainScoreModel('id')
    assert model.identifier == 'id'
    assert dict(model.region_layer_map) == {}
    assert dict(model.extractor) == {}
    assert model.in_channels == {'control'}


@pytest.mark.parametrize('word', ['in_channels', 'required_channels'])
@pytest.mark.parametrize('name, channel', [('vision', 'vision:image'), ('video', 'vision:video')])
def test_channel_arguments_refuse_a_modality_name(word, name, channel):
    with pytest.raises(ValueError, match=f"use '{channel}'"):
        BrainScoreModel('id', reply=_f, **{word: {name}})


def test_in_channels_and_required_channels_are_declared_as_given():
    model = BrainScoreModel('id', reply=_f, in_channels={'text', 'vision:video'},
                            required_channels={'vision:video'})
    assert {'text', 'vision:video'} <= model.in_channels
    assert model.required_channels == {'vision:video'}
