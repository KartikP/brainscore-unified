"""Shared coordinate checks and stable encodings for fitted activity tools."""
import hashlib
import json
from typing import Any

import numpy as np
from brainscore_core.events import Selection


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, allow_nan=False, separators=(',', ':'))


def _checksum(payload: dict) -> str:
    return hashlib.sha256(_canonical(payload).encode()).hexdigest()


def _activity(value: Any, target: Selection, *, label: str = 'Probe') -> tuple[np.ndarray, tuple[int, ...]]:
    import xarray as xr
    if not isinstance(value, xr.DataArray) or 'neuroid' not in value.dims:
        raise ValueError('Supply an assembly with a neuroid dimension and original unit_index coordinates')
    unit_coord = None
    for name in ('unit_index', 'neuron_number_in_layer', 'neuroid_num'):
        try:
            unit_coord = value[name]
            break
        except KeyError:
            continue
    try:
        layer_coord = value['layer']
    except KeyError:
        raise ValueError('Activity must retain original layer coordinates') from None
    if unit_coord is None:
        raise ValueError('Activity must retain original unit_index coordinates or a wrapper equivalent')
    if value.dims[-1] != 'neuroid':
        raise ValueError('Activity must end in the neuroid dimension')
    units = np.asarray(unit_coord.values)
    if (unit_coord.dims != ('neuroid',) or units.dtype.kind not in 'iu'
            or np.any(units < 0) or len(np.unique(units)) != len(units)):
        raise ValueError('unit_index must contain distinct, nonnegative original integer addresses')
    if layer_coord.dims != ('neuroid',) or np.any(layer_coord.values != target.layer):
        raise ValueError(f'Activity must come from target {target.layer!r}')
    units = tuple(int(i) for i in units)
    if target.indices is not None and units != tuple(target.indices):
        raise ValueError('Activity units or their order differ from the fitted target')
    array = np.asarray(value.values, dtype=np.float64)
    if not array.size or not np.isfinite(array).all():
        raise ValueError(f'{label} activity must be nonempty and finite')
    return array, units


def _fitting_data(activity: Any, labels: Any, target: Selection) -> tuple[np.ndarray, tuple[int, ...], np.ndarray]:
    values, units = _activity(activity, target, label='Fitting')
    labels = np.asarray(labels)
    if activity.dims != ('presentation', 'neuroid'):
        raise ValueError('Fit requires (presentation, neuroid); choose any pooling explicitly')
    if (labels.ndim != 1 or len(labels) != len(values) or labels.dtype.kind not in 'US'
            or np.any(labels == '') or len(np.unique(labels)) < 2):
        raise ValueError('Provide one nonempty string label per example and at least two classes')
    return values, units, labels


def _fitting_provenance(values: np.ndarray, units: tuple[int, ...], labels: np.ndarray,
                        supplied: dict | None = None) -> dict:
    return {**(supplied or {}), 'n_fitting_examples': len(values), 'fitting_dtype': 'float64',
            'numpy_version': np.__version__,
            'fitting_sha256': hashlib.sha256(
                values.tobytes() + _canonical([list(units), labels.tolist()]).encode()).hexdigest()}
