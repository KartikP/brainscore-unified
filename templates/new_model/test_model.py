"""Tests are the spec. A model should: register + load by identifier, declare the
right modalities, and accept a recording target. A full process() smoke needs real
stimuli (do that in an integration test); keep these light + offline. Adapt freely.
"""
import pytest


def test_registered_and_loads():
    import brainscore
    assert 'your-model' in brainscore.model_registry
    model = brainscore.load_model('your-model')
    assert model.identifier == 'your-model'


def test_supported_modalities():
    import brainscore
    model = brainscore.load_model('your-model')
    # supported_modalities is DERIVED from preprocessors.keys() — never hasattr.
    assert 'vision' in model.supported_modalities   # TODO: match your preprocessors


def test_start_recording_accepts_known_region():
    import brainscore
    model = brainscore.load_model('your-model')
    model.start_recording('IT')                      # TODO: a region in your region_layer_map
    # A single unknown STRING is an escape hatch (treated as a raw layer path),
    # so it does NOT raise. The LIST form validates against region_layer_map:
    with pytest.raises(Exception):
        model.start_recording(['NotARegion'])        # unknown region fails fast
