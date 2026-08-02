"""Tests are the spec. A model should: register + load by identifier, declare the right
modalities, accept a recording target, and actually return an assembly from process().

That last one matters. Registration and modality checks pass even when the model cannot
run — they never touch the backbone. The process() smoke below is what catches a
registration that is wired up wrong. Keep it as you swap in your own model.
"""
import os
import tempfile

import numpy as np
import pytest
from PIL import Image


@pytest.fixture(scope='module')
def tiny_stimuli():
    """Four small images on disk, as a StimulusSet. Replace with your own stimuli."""
    from brainscore_core.supported_data_standards.brainio.stimuli import StimulusSet
    directory = tempfile.mkdtemp(prefix='your_model_test_')
    rows, paths = [], {}
    for i in range(4):
        sid = f'stim{i}'
        p = os.path.join(directory, f'{sid}.png')
        rng = np.random.RandomState(i)
        Image.fromarray(rng.randint(0, 255, (64, 64, 3), dtype=np.uint8)).save(p)
        # 'image_file_name' is load-bearing: dispatch picks the modality by looking for
        # a RECOGNIZED column name. stimulus_paths alone is not enough.
        rows.append({'stimulus_id': sid, 'image_file_name': p})
        paths[sid] = p
    stimuli = StimulusSet(rows)
    stimuli.stimulus_paths = paths
    stimuli.identifier = 'your-model-test-stimuli'
    return stimuli


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


def test_process_returns_an_assembly(tiny_stimuli):
    """The one test that proves the registration actually works end to end."""
    import brainscore
    model = brainscore.load_model('your-model')
    model.start_recording('IT')
    assembly = model.process(tiny_stimuli)

    assert 'presentation' in assembly.dims and 'neuroid' in assembly.dims
    assert assembly.sizes['presentation'] == len(tiny_stimuli)
    assert assembly.sizes['neuroid'] > 0, 'recorded zero units — check region_layer_map'
    assert np.isfinite(np.asarray(assembly.values)).all(), 'activations contain NaN/inf'


def test_multi_region_carries_provenance(tiny_stimuli):
    """Recording several regions at once tags every unit with where it came from.

    Only meaningful once region_layer_map has more than one entry; skipped otherwise.
    """
    import brainscore
    model = brainscore.load_model('your-model')
    regions = list(model.region_layer_map)
    if len(regions) < 2:
        pytest.skip('add a second region to region_layer_map to exercise this')
    model.start_recording(regions[:2])
    assembly = model.process(tiny_stimuli)
    tagged = set(np.asarray(assembly['region'].values).astype(str))
    assert set(regions[:2]) <= tagged
