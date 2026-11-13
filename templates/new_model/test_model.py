"""Tests are the spec. A model should: register + load by identifier, declare the right
channels, accept a recording target, and actually return an assembly from a session.

That last one matters. Registration and channel checks pass even when the model cannot
run — they never touch the backbone. The session smoke below (the path every benchmark
takes, through participate) is what catches a registration that is wired up wrong.
Keep it as you swap in your own model.
"""
import os

# Set BEFORE anything imports brainscore. Feature extraction otherwise writes to
# ~/.result_caching, and a template smoke test has no business depending on the state
# of the user's home directory — that path can be missing, read-only, or (as on one
# machine here) a symlink to an unmounted drive, and the test then fails with a
# FileNotFoundError that has nothing to do with the model being registered.
os.environ.setdefault('RESULTCACHING_DISABLE', '1')

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


def test_in_channels():
    import brainscore
    model = brainscore.load_model('your-model')
    # in_channels is derived from the extractor mapping.
    assert 'vision:image' in model.in_channels   # TODO: match your extractor mapping


def test_start_recording_accepts_known_region():
    import brainscore
    model = brainscore.load_model('your-model')
    model.start_recording('IT')                      # TODO: a region in your region_layer_map
    # A name that is neither a region nor a layer of the extractor is refused before any
    # compute, with the same rule for strings and lists.
    for target in ('NotARegion', ['NotARegion']):
        with pytest.raises(ValueError, match='not a region in region_layer_map'):
            model.start_recording(target)


def test_session_returns_an_assembly(tiny_stimuli):
    """The one test that proves the registration actually works end to end."""
    import brainscore
    from brainscore_core.streaming_helpers import neural_response
    model = brainscore.load_model('your-model')
    assembly = neural_response(model, tiny_stimuli, ['neural:IT'])   # TODO: your region

    assert 'presentation' in assembly.dims and 'neuroid' in assembly.dims
    assert assembly.sizes['presentation'] == len(tiny_stimuli)
    assert assembly.sizes['neuroid'] > 0, 'recorded zero units — check region_layer_map'
    assert np.isfinite(np.asarray(assembly.values)).all(), 'activations contain NaN/inf'


def test_multi_region_carries_provenance(tiny_stimuli):
    """Several regions recorded in one session each arrive on their own channel.

    Only meaningful once region_layer_map has more than one entry; skipped otherwise.
    """
    import brainscore
    model = brainscore.load_model('your-model')
    regions = list(model.region_layer_map)
    if len(regions) < 2:
        pytest.skip('add a second region to region_layer_map to exercise this')
    from brainscore_core.streaming_helpers import stimulus_session
    session = stimulus_session(tiny_stimuli, [f'neural:{region}' for region in regions[:2]])
    model.participate(session)
    for region in regions[:2]:   # each region arrives on its own channel, with its layer
        assembly = session.collect(f'neural:{region}')
        assert assembly.sizes['neuroid'] > 0
        assert len(set(np.asarray(assembly['layer'].values).astype(str))) >= 1
