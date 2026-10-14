"""Disk snapshots must preserve every activation without whole-array copies."""
import hashlib
from types import SimpleNamespace

import numpy as np
import pytest

from brainscore.validation import benchmark_parity as parity
from brainscore.validation import parity_storage as storage

pytestmark = pytest.mark.unit


@pytest.mark.parametrize('layout', ['C', 'F', 'sliced', 'reversed'])
@pytest.mark.parametrize('dtype', ['float32', 'float64', '>f4'])
def test_snapshot_preserves_layout_independent_hash_and_values(tmp_path, monkeypatch, layout, dtype):
    values = np.arange(120, dtype=dtype).reshape(10, 12)
    if layout == 'F':
        values = np.asfortranarray(values)
    elif layout == 'sliced':
        values = values[:, ::2]
    elif layout == 'reversed':
        values = values[::-1, ::-1]
    expected = np.ascontiguousarray(values)
    # Intercept the writer's digest updates to measure actual block sizes.
    digest = hashlib.sha256()
    sizes = []

    def update(data):
        sizes.append(len(data))
        digest.update(data)

    monkeypatch.setattr(storage.hashlib, 'sha256', lambda: SimpleNamespace(
        update=update, hexdigest=digest.hexdigest,
    ))
    saved = storage.snapshot(values, tmp_path / 'values.bin', chunk_bytes=32)
    assert max(sizes) <= 32
    assert saved.sha256 == hashlib.new('sha256', expected).hexdigest()
    chunks = list(saved.chunks(elements=3))
    assert all(len(chunk) <= 3 for chunk in chunks)
    np.testing.assert_array_equal(np.concatenate(chunks).reshape(values.shape), values)
    assert saved.dtype == values.dtype


def compare(tmp_path, actual, reference, monkeypatch, **kwargs):
    monkeypatch.setattr(parity, 'CHUNK_BYTES', 16)
    return parity._compare_snapshots(
        storage.snapshot(actual, tmp_path / 'actual'),
        storage.snapshot(reference, tmp_path / 'reference'),
        route=kwargs.get('route', 'native'), case=parity.CASES[0],
        native_atol=1e-6, pereira=kwargs.get('pereira', False), native_ulps=16,
    )


def test_final_chunk_mismatch_is_not_missed(tmp_path, monkeypatch):
    reference = np.zeros((4, 5), dtype=np.float32)
    actual = reference.copy()
    actual[-1, -1] = 1
    delta, _, error = compare(tmp_path, actual, reference, monkeypatch)
    assert delta == 1
    assert 'native activations differ' in error


def test_adapter_still_requires_exact_values(tmp_path, monkeypatch):
    reference = np.ones((4, 5), dtype=np.float32)
    actual = reference.copy()
    actual[-1, -1] = np.nextafter(np.float32(1), np.float32(2))
    with pytest.raises(AssertionError, match='adapter activations differ'):
        compare(tmp_path, actual, reference, monkeypatch, route='adapter')


def test_chunked_pereira_ulps_match_full_array_policy(tmp_path, monkeypatch):
    reference = np.linspace(-300, 300, 20, dtype=np.float32).reshape(4, 5)
    actual = reference.astype(np.float64) + parity.fp32_ulp_tolerance(reference)
    full = parity.assert_pereira_activations(actual, reference)
    delta, ulps, error = compare(tmp_path, actual, reference, monkeypatch, pereira=True)
    assert error is None
    assert ulps == full
    assert delta == np.max(np.abs(actual - reference))


def test_same_size_different_shape_fails(tmp_path, monkeypatch):
    with pytest.raises(AssertionError, match='shapes differ'):
        compare(tmp_path, np.ones((2, 4)), np.ones((4, 2)), monkeypatch)


def test_truncated_snapshot_fails(tmp_path):
    saved = storage.snapshot(np.ones((2, 3)), tmp_path / 'values')
    saved.path.write_bytes(b'')
    with pytest.raises(IOError, match='size changed'):
        list(saved.chunks(elements=2))


def test_insufficient_disk_fails_before_creating_file(tmp_path, monkeypatch):
    monkeypatch.setattr(storage.shutil, 'disk_usage', lambda path: SimpleNamespace(free=0))
    with pytest.raises(OSError, match='scratch space'):
        storage.snapshot(np.ones(10), tmp_path / 'values')
    assert not list(tmp_path.iterdir())
