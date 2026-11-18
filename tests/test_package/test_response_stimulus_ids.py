"""response_stimulus_ids accepts only a response that answers each presented
stimulus exactly once, and refuses every other shape."""
import numpy as np
import pytest
import xarray as xr

from brainscore_core.supported_data_standards.brainio.assemblies import NeuroidAssembly

from brainscore.presentation import response_stimulus_ids

IDS = ['s0', 's1', 's2']


def _assembly(presentation_coords, n=3):
    return NeuroidAssembly(
        np.zeros((n, 2)), dims=['presentation', 'neuroid'],
        coords={**presentation_coords, 'neuroid_id': ('neuroid', ['a', 'b'])})


def test_a_stimulus_id_coord_is_read():
    response = _assembly({'stimulus_id': ('presentation', IDS),
                          'other': ('presentation', [1, 2, 3])})
    assert list(response_stimulus_ids(response, IDS)) == IDS


def test_any_order_is_accepted():
    response = _assembly({'stimulus_id': ('presentation', IDS[::-1]),
                          'other': ('presentation', [1, 2, 3])})
    assert list(response_stimulus_ids(response, IDS)) == IDS[::-1]


def test_a_lone_stimulus_id_coord_is_read_from_the_plain_index():
    response = _assembly({'stimulus_id': ('presentation', IDS)})
    assert list(response_stimulus_ids(response, IDS)) == IDS


def test_a_lone_unrelated_coord_is_refused():
    response = _assembly({'image_id': ('presentation', ['x', 'y', 'z'])})
    with pytest.raises(ValueError, match='exactly once'):
        response_stimulus_ids(response, IDS)


@pytest.mark.parametrize('ids', [['s0', 's1', 'other'], ['s0', 's1', 's1']])
def test_wrong_or_repeated_ids_are_refused(ids):
    response = _assembly({'stimulus_id': ('presentation', ids),
                          'other': ('presentation', [1, 2, 3])})
    with pytest.raises(ValueError, match='exactly once'):
        response_stimulus_ids(response, IDS)


def test_too_few_rows_are_refused():
    response = _assembly({'stimulus_id': ('presentation', IDS[:2]),
                          'other': ('presentation', [1, 2])}, n=2)
    with pytest.raises(ValueError, match='exactly once'):
        response_stimulus_ids(response, IDS)


def test_a_multiindex_without_stimulus_id_is_refused():
    response = _assembly({'image_id': ('presentation', IDS),
                          'other': ('presentation', [1, 2, 3])})
    with pytest.raises(ValueError, match='no stimulus_id'):
        response_stimulus_ids(response, IDS)


def test_a_scalar_stimulus_id_is_refused():
    response = xr.DataArray(np.zeros((3, 2)), dims=['presentation', 'neuroid'],
                            coords={'stimulus_id': 's0'})
    with pytest.raises(ValueError, match='dims'):
        response_stimulus_ids(response, IDS)


def test_a_stimulus_id_on_another_axis_is_refused():
    response = xr.DataArray(np.zeros((3, 3)), dims=['presentation', 'neuroid'],
                            coords={'stimulus_id': ('neuroid', IDS)})
    with pytest.raises(ValueError, match='dims'):
        response_stimulus_ids(response, IDS)


def test_a_response_without_a_presentation_axis_is_refused():
    with pytest.raises(ValueError, match="no 'presentation' axis"):
        response_stimulus_ids(xr.DataArray(np.zeros(3), dims=['neuroid']), IDS)
