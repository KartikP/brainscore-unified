"""The constructor takes one name per part, as keywords, and no older name."""
import inspect

import pytest

from brainscore import BrainScoreModel


def _f(*a, **k):
    return None


PARTS = ('extractor', 'behavioral_readout_layer', 'answer', 'reply', 'likelihood', 'policy',
         'policy_inputs', 'policy_outputs', 'intervention', 'memory', 'supported_modalities',
         'required_modalities', 'visual_degrees', 'capability_config')


def test_signature_names_each_part_once_and_parts_are_keywords():
    params = inspect.signature(BrainScoreModel.__init__).parameters
    assert list(params) == ['self', 'identifier', 'model', 'region_layer_map', *PARTS]
    for name in PARTS:
        assert params[name].kind == inspect.Parameter.KEYWORD_ONLY, name


@pytest.mark.parametrize('name', ['preprocessors', 'activations_model', 'generation_fn',
                                  'action_fn', 'state_change_fn', 'region_modality_map',
                                  'backbone_id'])
def test_removed_names_are_refused(name):
    with pytest.raises(TypeError, match=name):
        BrainScoreModel('id', **{name: _f})


def test_parts_after_the_region_map_are_not_positional():
    with pytest.raises(TypeError):
        BrainScoreModel('id', None, {}, {'vision': _f})


def test_extractor_is_a_mapping():
    with pytest.raises(TypeError, match='mapping'):
        BrainScoreModel('id', extractor=_f)


def test_each_part_declares_its_channels():
    model = BrainScoreModel('id', policy=_f, intervention=_f, answer=_f,
                            supported_modalities={'text'})
    assert {'motor', 'perturbation', 'behavior:choice'} <= model.out_channels
    assert {'text', 'instruction', 'observation', 'lesion'} <= model.in_channels


def test_constructor_and_process_are_documented():
    assert (BrainScoreModel.__init__.__doc__ or '').strip()
    assert (BrainScoreModel.process.__doc__ or '').strip()


def test_identifier_only_construction_works():
    # The docstring says only identifier is always required.
    model = BrainScoreModel('id')
    assert model.identifier == 'id'
    assert dict(model.region_layer_map) == {}
    assert dict(model.extractor) == {}
    assert model.supported_modalities == set()
