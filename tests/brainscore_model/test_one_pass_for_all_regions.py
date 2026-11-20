"""Sessions that each ask one region of the same stored table share one model pass.

A benchmark family often asks one region per identifier on one stimulus set
(V1, V2, V4, IT). With the store on, the first session's pass records every
region of the map that its extractor records, and the store keeps each layer
apart; the next session reads its region from the store. Each payload equals
the payload of a pass for that region alone, and an entry that a pass for one
region wrote is read as before.
"""
import numpy as np
import pytest
import result_caching
import xarray

from brainscore import BrainScoreModel
from brainscore_core.streaming_helpers import stimulus_session

from tests.brainscore_model import tiny_wrappers

pytestmark = pytest.mark.unit

VARIABLE = 'BRAINSCORE_REGIONS_PER_PASS'


@pytest.fixture
def store(tmp_path, monkeypatch):
    """The store on, in a temporary folder, for the wrapper of the test; and a pass counter on the
    network's class (a hook on the network would enter the store key)."""
    monkeypatch.delenv('RESULTCACHING_DISABLE', raising=False)
    monkeypatch.delenv(VARIABLE, raising=False)

    def install(wrapper, network, passes):
        method = getattr(wrapper, '_from_texts_stored', None) or wrapper._from_paths_stored
        storage = next(cell.cell_contents for cell in method.__func__.__closure__
                       if isinstance(cell.cell_contents, result_caching._Storage))
        monkeypatch.setattr(storage, '_storage_directory', str(tmp_path / 'store'))
        forward = type(network).forward

        def counted(self, *args, **kwargs):
            if self is network:
                passes.append(1)
            return forward(self, *args, **kwargs)
        monkeypatch.setattr(type(network), 'forward', counted)
    return install


def _model(kind, folder, store):
    wrapper, stimuli, regions, network = tiny_wrappers.FACTORIES[kind](folder)
    passes = []
    store(wrapper, network, passes)
    model = BrainScoreModel(f'tiny-{kind}', extractor={tiny_wrappers.MODALITY[kind]: wrapper},
                            region_layer_map=regions)
    return model, stimuli, passes


def assert_same_output(actual, expected):
    """Equal values, dtype, dims, coordinates, index levels and attributes. The order in which the
    coordinates are listed is left out: it already depends on the store (an entry of several layers
    lists ``neuroid`` first)."""
    assert type(actual) is type(expected) and actual.dims == expected.dims and actual.dtype == expected.dtype
    np.testing.assert_array_equal(actual.values, expected.values)
    for dim, index in expected.indexes.items():
        assert list(actual.indexes[dim].names) == list(index.names), dim
    xarray.testing.assert_identical(actual, expected)


def _payload(model, stimuli, region):
    session = stimulus_session(stimuli, [f'neural:{region}'])
    model.participate(session)
    (event,) = [e for e in session.emitted if e.channel == f'neural:{region}']
    return event.payload


@pytest.mark.parametrize('kind', ['text', 'vlm_vision'])
def test_two_sessions_of_one_region_each_take_one_pass(tmp_path, store, kind):
    model, stimuli, passes = _model(kind, tmp_path, store)
    _payload(model, stimuli, 'V1')
    _payload(model, stimuli, 'IT')
    assert len(passes) == len(stimuli)           # batch size 1: one pass per stimulus, for both regions


@pytest.mark.parametrize('kind', ['text', 'vlm_vision', 'video'])
def test_each_region_equals_a_pass_for_that_region_alone(tmp_path, store, monkeypatch, kind):
    model, stimuli, _ = _model(kind, tmp_path, store)
    shared = {region: _payload(model, stimuli, region) for region in ('V1', 'IT')}
    monkeypatch.setenv(VARIABLE, 'one')          # one pass for each region, its layer alone
    for region in ('V1', 'IT'):
        model, stimuli, passes = _model(kind, tmp_path, store)  # the same files
        stimuli.identifier = f'{stimuli.identifier}-alone-{region}'
        alone = _payload(model, stimuli, region)
        assert_same_output(shared[region], alone)
        # A reduction downstream gives the same bits: a row sum, and a mean over time bins (as the
        # naturalistic benchmarks take before the regression).
        assert np.array_equal(shared[region].values.sum(axis=-1), alone.values.sum(axis=-1))
        if 'time_bin' in alone.dims:
            assert np.array_equal(shared[region].mean('time_bin').values, alone.mean('time_bin').values)
        assert len(passes) == len(stimuli)


@pytest.mark.parametrize('kind', ['text', 'vlm_vision'])
def test_an_entry_of_one_region_is_read_and_completed_in_one_pass(tmp_path, store, monkeypatch, kind):
    monkeypatch.setenv(VARIABLE, 'one')          # writes the entry a pass for one region writes
    model, stimuli, passes = _model(kind, tmp_path, store)
    alone = _payload(model, stimuli, 'V1')
    assert len(passes) == len(stimuli)
    monkeypatch.delenv(VARIABLE)
    model, stimuli, passes = _model(kind, tmp_path, store)
    read = _payload(model, stimuli, 'V1')
    assert_same_output(read, alone)
    _payload(model, stimuli, 'V4')
    _payload(model, stimuli, 'IT')
    assert len(passes) == len(stimuli)           # the entry's layer is read; the missing ones in one pass


@pytest.mark.parametrize('kind', ['text', 'vlm_vision'])
def test_a_stored_region_is_read_in_cached_only_mode(tmp_path, store, monkeypatch, kind):
    monkeypatch.setenv(VARIABLE, 'one')          # stores the layer of V1 alone
    model, stimuli, _ = _model(kind, tmp_path, store)
    alone = _payload(model, stimuli, 'V1')
    monkeypatch.delenv(VARIABLE)
    monkeypatch.setenv('RESULTCACHING_CACHEDONLY', '1')
    model, stimuli, passes = _model(kind, tmp_path, store)
    assert_same_output(_payload(model, stimuli, 'V1'), alone)
    assert passes == []
    with pytest.raises(result_caching.NotCachedError):
        _payload(model, stimuli, 'V4')            # a requested layer that is not stored still fails


def test_an_extractor_without_a_layer_store_keeps_its_own_call(tmp_path, monkeypatch):
    """An extractor that states ``independent_layers`` but always runs the network gains nothing from
    a wider pass: it records only the layers the session asks."""
    monkeypatch.delenv('RESULTCACHING_DISABLE', raising=False)
    monkeypatch.delenv(VARIABLE, raising=False)
    wrapper, stimuli, regions, _ = tiny_wrappers.vlm_video(tmp_path)
    calls = []
    call = type(wrapper).__call__
    monkeypatch.setattr(type(wrapper), '__call__',
                        lambda self, stimuli, layers, **kwargs: calls.append(list(layers)) or call(
                            self, stimuli, layers, **kwargs))
    model = BrainScoreModel('tiny-vlm-video', extractor={'video': wrapper}, region_layer_map=regions)
    _payload(model, stimuli, 'V1')
    assert calls == [list(regions['V1'].layer_paths)]


def test_a_layer_added_to_an_entry_reads_as_a_new_pass(tmp_path, store):
    """Write layer A, then ask A and B (the store merges B into the entry), then read B alone:
    the units and values equal a new pass for B alone (each layer has more than ten units, so an
    order by name puts unit 10 before unit 2)."""
    wrapper, stimuli, _, network = tiny_wrappers.vlm_vision(tmp_path)
    store(wrapper, network, [])
    wrapper(stimuli, layers=['0'])
    wrapper(stimuli, layers=['0', '1'])
    read = wrapper(stimuli, layers=['1'])
    stimuli.identifier = f'{stimuli.identifier}-new'
    new = wrapper(stimuli, layers=['1'])
    assert read.sizes['neuroid'] > 10
    assert list(read['neuroid_num'].values) == list(new['neuroid_num'].values)
    assert_same_output(read, new)


def test_three_regions_asked_together_keep_the_layout_of_one_joint_call(tmp_path, store):
    """Release code already shared a session of several regions: each region is the plain selection
    of its neuroids from one joint call (values, strides and the mean over eight time bins)."""
    from brainscore.model_helpers.video_wrapper import VideoWrapper
    wrapper, stimuli, regions, network = tiny_wrappers.video(tmp_path)
    wrapper = VideoWrapper(network, tiny_wrappers._frames_to_tensor, identifier='tiny-video-8',
                           num_frames=8, batch_size=1, device='cpu')
    store(wrapper, network, [])
    model = BrainScoreModel('tiny-video-8', extractor={'video': wrapper}, region_layer_map=regions)
    names = ['V1', 'V4', 'IT']
    session = stimulus_session(stimuli, [f'neural:{name}' for name in names])
    model.participate(session)
    payloads = {e.channel: e.payload for e in session.emitted}
    layers = [regions[name].layer_paths[0] for name in names]
    stimuli.identifier = 'tiny-clips-reference'          # a new store entry for the reference call
    joint = wrapper(stimuli, layers=layers)
    labels = np.asarray(joint['layer'].values)
    for name, layer in zip(names, layers):
        expected = joint.isel(neuroid=np.flatnonzero(labels == layer))
        actual = payloads[f'neural:{name}']
        assert 'time_bin' in actual.dims and actual.sizes['time_bin'] == 8
        np.testing.assert_array_equal(actual.values, expected.values)
        assert actual.values.strides == expected.values.strides
        assert np.array_equal(actual.mean('time_bin').values, expected.mean('time_bin').values)
