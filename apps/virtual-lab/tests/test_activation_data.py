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
from umi_lab.activation_data import activation_page, PAGE_SIZE


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
