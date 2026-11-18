"""A picture-and-sound combination has the picture tower of its picture-only registration.

For each combination, the family's one loader and one tower builder build its
picture (or video) extractor, with the arguments of the picture-only
registration; only the labels (identifier, the legacy wrapper's stored-path
name) differ. No weights: the loaders and builders are replaced by recorders,
and the wiring is read.
"""
import importlib
import inspect
import sys
from unittest.mock import MagicMock

import pytest

pytestmark = pytest.mark.unit

# (family module, loader, tower builder, picture-only identifier, combination identifiers)
FAMILIES = [
    ('blip2_opt_2_7b', 'load_blip2', 'picture_extractor', 'blip2-opt-2.7b', ['blip2-wav2vec2']),
    ('clip_vit_b_32', 'load_clip', 'picture_extractor', 'clip-vit-b-32', ['clip-wav2vec2', 'clip-random-wav2vec2']),
    ('clip_vit_b_32', 'load_clip', 'picture_extractor', 'random-vit-b-32',
     ['random-clip-wav2vec2', 'random-clip-random-wav2vec2']),
    ('qwen25_vl_3b', 'load_qwen', 'picture_extractor', 'qwen2.5-vl-3b', ['qwen2.5-vl-wav2vec2']),
    ('vjepa_v1', 'load_backbone', 'video_extractor', 'vjepa1-vitl',
     ['vjepa1-wav2vec2', 'vjepa1-random-wav2vec2']),
]
MODULES = {'blip2-opt-2.7b': 'blip2_opt_2_7b', 'clip-vit-b-32': 'clip_vit_b_32', 'random-vit-b-32': 'random_vit_b_32',
           'qwen2.5-vl-3b': 'qwen25_vl_3b', 'vjepa1-vitl': 'vjepa_v1', 'blip2-wav2vec2': 'multimodal_av_blip2_wav2vec2',
           'qwen2.5-vl-wav2vec2': 'multimodal_av_qwen_wav2vec2'}
LABELS = {'identifier', 'backbone_id'}


def _module(identifier):
    name = MODULES.get(identifier) or ('multimodal_av_clip_wav2vec2' if 'clip' in identifier
                                       else 'multimodal_av_vjepa_wav2vec2')
    return importlib.import_module(f'brainscore.models.{name}.model')


def _replace_everywhere(monkeypatch, original, replacement):
    """Replace ``original`` in every loaded registration and helper module that binds it."""
    for name, module in list(sys.modules.items()):
        if name.startswith(('brainscore.models', 'brainscore.model_helpers')) and module is not None:
            for attribute, value in list(vars(module).items()):
                if value is original:
                    monkeypatch.setattr(module, attribute, replacement)


@pytest.fixture
def recorded(monkeypatch):
    """Run ``get_model`` with recorders; returns the calls of the family loader and tower builder."""
    originals = {}

    def run(family, loader_name, tower_name, identifier):
        _module(identifier)
        family_module = importlib.import_module(f'brainscore.models.{family}.model')
        calls = []
        for name in (loader_name, tower_name):          # the real functions, before a first run replaced them
            originals.setdefault((family, name), getattr(family_module, name))
        load_signature = inspect.signature(originals[family, loader_name])
        tower_signature = inspect.signature(originals[family, tower_name])

        def bound(signature, args, kwargs):
            arguments = signature.bind(*args, **kwargs)
            arguments.apply_defaults()
            return dict(arguments.arguments)

        def loader(*args, **kwargs):
            loaded = (MagicMock(name='network'), MagicMock(name='processor'))
            calls.append(('load', bound(load_signature, args, kwargs), loaded))
            return {'load_qwen': (*loaded, {}), 'load_backbone': loaded[0]}.get(loader_name, loaded)

        def tower(*args, **kwargs):
            calls.append(('tower', bound(tower_signature, args, kwargs), None))
            return (MagicMock(name='extractor'), MagicMock(name='model')) if tower_name == 'video_extractor' \
                else MagicMock(name='extractor')

        from brainscore import BrainScoreModel
        from brainscore.models import _shared
        from brainscore.model_helpers import replies, text_wrapper
        for original, replacement in (
                (getattr(family_module, loader_name), loader), (getattr(family_module, tower_name), tower),
                (originals[family, loader_name], loader), (originals[family, tower_name], tower),
                (BrainScoreModel, lambda *a, **k: k), (text_wrapper.TextWrapper, MagicMock(name='TextWrapper')),
                (_shared.build_audio_wrapper, lambda *a, **k: (MagicMock(name='audio'), None)),
                (replies.chat_parts, lambda *a, **k: (None, None))):
            _replace_everywhere(monkeypatch, original, replacement)
        _module(identifier).get_model(identifier)
        return calls
    return run


def _settings(call):
    """A call's arguments without labels (the identifier names the registration in a download check or
    labels the activations; ``backbone_id`` names a stored path) and without the loaded objects."""
    kind, arguments, _ = call
    return kind, {name: value for name, value in arguments.items()
                  if name not in LABELS and not isinstance(value, MagicMock)}


@pytest.mark.parametrize('family, loader_name, tower_name, picture_only, combination', [
    (family, loader, tower, picture_only, combination)
    for family, loader, tower, picture_only, combinations in FAMILIES for combination in combinations])
def test_a_combination_builds_the_picture_tower_of_its_picture_only_registration(
        family, loader_name, tower_name, picture_only, combination, recorded):
    alone = recorded(family, loader_name, tower_name, picture_only)
    combined = recorded(family, loader_name, tower_name, combination)
    for calls in (alone, combined):
        assert [kind for kind, *_ in calls] == ['load', 'tower']
        loaded, given = calls[0][2], [value for value in calls[1][1].values() if isinstance(value, MagicMock)]
        assert given == list(loaded[:len(given)])          # the tower is built from what the loader gave
    assert [_settings(call) for call in combined] == [_settings(call) for call in alone]


@pytest.mark.parametrize('picture_only, combination', [
    ('blip2-opt-2.7b', 'blip2-wav2vec2'), ('clip-vit-b-32', 'clip-wav2vec2'),
    ('qwen2.5-vl-3b', 'qwen2.5-vl-wav2vec2'), ('vjepa1-vitl', 'vjepa1-wav2vec2')])
def test_a_combination_records_its_picture_regions_from_the_same_layers(picture_only, combination):
    alone, combined = _module(picture_only).REGION_LAYER_MAP, _module(combination).REGION_LAYER_MAP
    shared = set(alone) & set(combined) - {'A1', 'language_system'}
    assert shared and {region: combined[region] for region in shared} == {region: alone[region] for region in shared}


def test_the_blip2_load_is_float16_for_every_registration():
    import inspect
    from brainscore.models.blip2_opt_2_7b import model
    source = inspect.getsource(model.load_blip2)
    assert 'torch_dtype=torch.float16' in source
    combination = inspect.getsource(_module('blip2-wav2vec2'))
    assert 'from_pretrained' not in combination and 'torch_dtype' not in combination
