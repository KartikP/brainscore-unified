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


@pytest.mark.parametrize('kind', ['text', 'vlm_vision'])
def test_each_region_equals_a_pass_for_that_region_alone(tmp_path, store, monkeypatch, kind):
    model, stimuli, _ = _model(kind, tmp_path, store)
    shared = {region: _payload(model, stimuli, region) for region in ('V1', 'IT')}
    monkeypatch.setenv(VARIABLE, 'one')          # one pass for each region, its layer alone
    for region in ('V1', 'IT'):
        model, stimuli, passes = _model(kind, tmp_path, store)  # the same files
        stimuli.identifier = f'{stimuli.identifier}-alone-{region}'
        assert_same_output(shared[region], _payload(model, stimuli, region))
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
    # The store merges the new layers into the entry and orders its neuroids by name, as it does
    # today when a second region is added to an entry: compare each neuroid.
    xarray.testing.assert_identical(read.sortby('neuroid_id'), alone.sortby('neuroid_id'))
    _payload(model, stimuli, 'V4')
    _payload(model, stimuli, 'IT')
    assert len(passes) == len(stimuli)           # the entry's layer is read; the missing ones in one pass
