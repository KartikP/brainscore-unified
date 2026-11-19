"""The full heatmap addresses every saved neuron without rerunning a model."""
import json
from pathlib import Path
import threading
from types import SimpleNamespace
from urllib.error import HTTPError
from urllib.request import urlopen

import numpy as np
import pytest
import xarray as xr

from brainscore.run_record import PayloadCodec
from umi_lab.activation_data import activation_page, temporal_page, PAGE_SIZE


def save(directory: Path, assembly: xr.DataArray) -> None:
    directory.mkdir(parents=True)
    encoded = PayloadCodec(directory).encode([assembly])
    (directory / 'result.json').write_text(json.dumps(encoded, allow_nan=False))


def test_every_neuron_is_reachable_across_pages_including_the_last(tmp_path):
    data = np.arange(2 * 9216, dtype=np.float32).reshape(2, 9216)
    assembly = xr.DataArray(data, dims=['presentation', 'neuroid'],
        coords={'stimulus_id': ('presentation', ['first', 'second']),
                'neuroid_id': ('neuroid', [f'u{i}' for i in range(9216)])})
    save(tmp_path / 'experiment', assembly)
    values = []
    for offset in range(0, 9216, PAGE_SIZE):
        page = activation_page(tmp_path / 'experiment', presentation=1, offset=offset,
                               stats=offset == 0)
        assert page['offset'] == offset and len(page['values']) <= PAGE_SIZE
        assert page['total_columns'] == 9216 and page['stimulus'] == 'second'
        if offset == 0:
            assert page['extent'] == float(data[1, -1])
        values.extend(page['values'])
    np.testing.assert_array_equal(values, data[1])
    last = activation_page(tmp_path / 'experiment', offset=9215, limit=1, labels=True)
    assert last['labels'] == ['u9215'] and last['values'] == [9215.0]


def test_full_view_preserves_dimension_order_fixed_slice_and_missing_values(tmp_path):
    data = np.arange(2 * 3 * 5, dtype=float).reshape(2, 3, 5)
    data[0, 1, 4] = np.nan
    save(tmp_path / 'experiment', xr.DataArray(data, dims=['time_bin', 'presentation', 'neuroid']))
    page = activation_page(tmp_path / 'experiment', presentation=1, stats=True)
    assert page['values'] == [5., 6., 7., 8., None]
    assert page['slice'] == {'time_bin': 0}
    assert page['extent'] == 8.
    json.dumps(page, allow_nan=False)


def test_time_bin_selector_reaches_last_bin_without_changing_saved_values(tmp_path):
    data = np.arange(3 * 2 * 5, dtype=float).reshape(3, 2, 5)
    data[2, 1, 4] = np.nan
    directory = tmp_path / 'experiment'
    save(directory, xr.DataArray(data, dims=['time_bin', 'presentation', 'neuroid'],
        coords={'time_bin_start_ms': ('time_bin', [0., 25., 70.]),
                'time_bin_end_ms': ('time_bin', [25., 50., 90.])}))
    page = activation_page(directory, presentation=1, time_bin=2, stats=True)
    assert page['values'] == [25., 26., 27., 28., None]
    assert page['extent'] == 28.
    assert page['slice'] == {'time_bin': 2}
    assert page['time'] == {'count': 3, 'index': 2, 'start_ms': 70., 'end_ms': 90., 'center_ms': None}
    assert activation_page(directory)['values'] == list(data[0, 0])
    for time_bin in (-1, 3, True, 1.5):
        with pytest.raises(ValueError):
            activation_page(directory, time_bin=time_bin)


@pytest.mark.parametrize('modality', ['video', 'audio'])
def test_native_time_metadata_survives_multiindex_recordings(tmp_path, modality):
    from brainscore_core.supported_data_standards.brainio.assemblies import NeuroidAssembly
    coords = {'time_bin_start_ms': ('time_bin', [0., 40.]),
              'time_bin_end_ms': ('time_bin', [40., 80.])}
    if modality == 'video':
        coords.update(time_bin_center_ms=('time_bin', [20., 60.]), time_bin_idx=('time_bin', [0, 1]))
    else:
        coords['time_bin_id'] = ('time_bin', [0, 1])
    directory = tmp_path / 'experiment'
    save(directory, NeuroidAssembly(np.arange(12).reshape(2, 2, 3),
        dims=['presentation', 'time_bin', 'neuroid'], coords=coords))
    page = temporal_page(directory, time_offset=1)
    assert page['time_bins'] == [{'index': 1, 'start_ms': 40., 'end_ms': 80.,
                                 'center_ms': 60. if modality == 'video' else None}]
    assert activation_page(directory, time_bin=1)['time']['start_ms'] == 40.


def test_temporal_pages_preserve_order_gaps_nonfinite_values_and_other_axes(tmp_path):
    data = np.arange(5 * 4 * 2 * 2, dtype=float).reshape(5, 4, 2, 2)
    data[4, 3, 1, 0] = np.inf
    directory = tmp_path / 'experiment'
    save(directory, xr.DataArray(data, dims=['neuroid', 'time_bin', 'presentation', 'feature'],
        coords={'time_bin_start_ms': ('time_bin', [0., 20., 80., 100.]),
                'time_bin_end_ms': ('time_bin', [20., 40., 100., 120.])}))
    page = temporal_page(directory, presentation=1, offset=3, limit=2, time_offset=2, time_limit=2)
    expected = data[3:5, 2:4, 1, 0].T
    np.testing.assert_equal(np.array([[np.nan if value is None else value for value in row]
                                    for row in page['values']]), np.where(np.isfinite(expected), expected, np.nan))
    assert page['neurons'] == [3, 4] and page['total_columns'] == 5
    assert page['time_bins'][0]['start_ms'] == 80.
    assert page['slice'] == {'feature': 0}
    trace = temporal_page(directory, presentation=1, neurons=[4, 0], time_limit=4)
    assert trace['neurons'] == [4, 0]
    assert trace['values'][0] == [float(data[4, 0, 1, 0]), float(data[0, 0, 1, 0])]
    assert trace['values'][-1][0] is None
    json.dumps(trace, allow_nan=False)


def test_temporal_channels_keep_independent_native_clocks(tmp_path):
    directory = tmp_path / 'experiment'
    directory.mkdir()
    def output(count, width):
        return xr.DataArray(np.arange(2 * count * 3).reshape(2, count, 3),
            dims=['presentation', 'time_bin', 'neuroid'],
            coords={'time_bin_start_ms': ('time_bin', np.arange(count) * width),
                    'time_bin_end_ms': ('time_bin', (np.arange(count) + 1) * width)})
    encoded = PayloadCodec(directory).encode([{'neural:IT': output(2, 40), 'neural:A1': output(4, 20)}])
    (directory / 'result.json').write_text(json.dumps(encoded, allow_nan=False))
    video = temporal_page(directory, channel='neural:IT')
    audio = temporal_page(directory, channel='neural:A1')
    assert video['total_time_bins'] == 2 and audio['total_time_bins'] == 4
    assert video['time_bins'][1]['start_ms'] == 40.
    assert audio['time_bins'][1]['start_ms'] == 20.


def test_unknown_time_bounds_remain_unknown_and_large_windows_fail_before_materialization(tmp_path):
    directory = tmp_path / 'experiment'
    save(directory, xr.DataArray(np.zeros((1, 1024, 65)), dims=['presentation', 'time_bin', 'neuroid']))
    page = temporal_page(directory, time_offset=1023, time_limit=1, offset=64, limit=1)
    assert page['values'] == [[0.]]
    assert page['time_bins'] == [{'index': 1023, 'start_ms': None, 'end_ms': None, 'center_ms': None}]
    with pytest.raises(ValueError, match='exceeds'):
        temporal_page(directory, time_limit=1024, limit=65)
    for settings in ({'time_offset': -1}, {'time_offset': 1024}, {'time_limit': 0},
                     {'time_limit': 1025}, {'neurons': [0, 0]}, {'neurons': [65]},
                     {'neurons': list(range(9))}, {'neurons': []}):
        with pytest.raises(ValueError):
            temporal_page(directory, **settings)
    no_time = tmp_path / 'still'
    save(no_time, xr.DataArray(np.zeros((1, 2)), dims=['presentation', 'neuroid']))
    with pytest.raises(ValueError, match='no recorded time'):
        temporal_page(no_time)


@pytest.mark.parametrize('settings', [{'offset': -1}, {'offset': 3}, {'limit': 0},
    {'limit': PAGE_SIZE + 1}, {'presentation': 2}, {'index': 1}])
def test_invalid_view_ranges_are_refused(tmp_path, settings):
    save(tmp_path / 'experiment', xr.DataArray(np.zeros((2, 3)), dims=['presentation', 'neuroid']))
    with pytest.raises(ValueError):
        activation_page(tmp_path / 'experiment', **settings)


def test_artifact_references_cannot_escape_run_directory(tmp_path):
    directory = tmp_path / 'experiment'
    save(directory, xr.DataArray([[1.]], dims=['presentation', 'neuroid']))
    encoded = json.loads((directory / 'result.json').read_text())
    encoded[0]['data'] = {'$array': '../outside'}
    (directory / 'result.json').write_text(json.dumps(encoded))
    with pytest.raises(ValueError, match='array reference'):
        activation_page(directory)


def test_http_full_view_works_without_a_subject_and_bounds_each_response(tmp_path):
    from umi_lab.server import make_server
    root = tmp_path / 'run'
    save(root / 'experiment', xr.DataArray(np.arange(60).reshape(2, 30), dims=['presentation', 'neuroid']))
    controller = SimpleNamespace(runs={'fixture': root}, token='fixture',
                                  snapshot=lambda: {'run': {'id': 'fixture', 'phase': 'complete'}})
    server = make_server(controller, tmp_path, 0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f'http://127.0.0.1:{server.server_port}/api/runs/fixture/activation'
    try:
        with urlopen(base + '?presentation=1&offset=29&limit=1') as response:
            assert json.load(response)['values'] == [59.]
        for query in ('?limit=999999', '?offset=../../outside', '?index=0&index=1', '?unknown=1'):
            with pytest.raises(HTTPError) as error:
                urlopen(base + query)
            assert error.value.code == 400
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_http_temporal_views_need_no_subject_and_reject_invalid_parameters(tmp_path):
    from umi_lab.server import make_server
    root = tmp_path / 'run'
    save(root / 'experiment', xr.DataArray(np.arange(24).reshape(2, 3, 4),
        dims=['presentation', 'time_bin', 'neuroid']))
    controller = SimpleNamespace(runs={'fixture': root}, token='fixture',
                                  snapshot=lambda: {'run': {'id': 'fixture', 'phase': 'complete'}})
    server = make_server(controller, tmp_path, 0)
    thread = threading.Thread(target=server.serve_forever, daemon=True); thread.start()
    base = f'http://127.0.0.1:{server.server_port}/api/runs/fixture/'
    try:
        with urlopen(base + 'activation?time_bin=2&presentation=1&limit=1') as response:
            assert json.load(response)['values'] == [20.]
        with urlopen(base + 'temporal?time_offset=1&time_limit=2&neurons=3,0&presentation=1') as response:
            assert json.load(response)['values'] == [[19., 16.], [23., 20.]]
        for endpoint in ('activation?time_bin=3', 'temporal?neurons=1,1', 'temporal?time_bin=0',
                         'temporal?time_offset=0&time_offset=1', 'temporal?time_limit=100000'):
            with pytest.raises(HTTPError) as error:
                urlopen(base + endpoint)
            assert error.value.code == 400
    finally:
        server.shutdown(); server.server_close(); thread.join(timeout=2)
