"""Model-side unit-coordinate support for the topographic-alignment axis.

The topographic metric (``metrics/topographic.py``) reads per-unit positions from
``tissue_x``/``tissue_y`` (or ``x``/``y``/``z``) neuroid coords. This module is the bridge:
it lets a model or benchmark ATTACH those coords, derive a neutral grid fallback for models
that don't supply their own, and build the shuffle-coordinate NULL the axis is validated
against.

Contract: a topographic model (Topo-Omni, TDANN, TopoLM) attaches ``tissue_x``/``tissue_y``
to its ``process()`` output — the unit's position on the model's sheet. A non-topographic
model has no intrinsic 2-D layout; :func:`grid_positions` gives a neutral fallback so the
metric still runs and scores near the shuffle null — which is the honest answer (no brain-like
topography). The shuffle null leaves every unit's responses (hence predictivity) unchanged
while destroying the spatial layout, so it isolates exactly the spatial signal the axis adds.
"""
import numpy as np

from brainscore_core.supported_data_standards.brainio.assemblies import walk_coords


def grid_positions(n_units: int) -> np.ndarray:
    """Neutral 2-D positions on a near-square grid in ``[0, 1]^2`` (row-major)."""
    side = int(np.ceil(np.sqrt(max(int(n_units), 1))))
    gx, gy = np.meshgrid(np.linspace(0, 1, side), np.linspace(0, 1, side))
    return np.stack([gx.ravel(), gy.ravel()], axis=1)[:n_units]


def _coord_names(assembly) -> set:
    try:
        return {name for name, _, _ in walk_coords(assembly)}
    except Exception:
        return set(assembly.coords)


def has_tissue_coords(assembly) -> bool:
    """True if the assembly already carries per-unit positions the metric can read."""
    names = _coord_names(assembly)
    return ({'tissue_x', 'tissue_y'} <= names) or ({'x', 'y'} <= names)


def _flatten_neuroid(assembly):
    """Turn a MultiIndex on the ``neuroid`` dim into plain coords so we can (re)assign
    a level. NeuroidAssembly builds a MultiIndex from its neuroid coords; reassigning
    ``tissue_x`` directly then raises a level-name conflict (xarray 2022.3)."""
    if 'neuroid' in getattr(assembly, 'indexes', {}):
        return assembly.reset_index('neuroid')
    return assembly


def attach_tissue_coords(assembly, positions: np.ndarray):
    """Return a (presentation, neuroid) assembly with ``tissue_x``/``tissue_y`` neuroid
    coords from ``positions`` of shape ``(n_units, 2)``."""
    positions = np.asarray(positions, dtype=float)
    a = assembly.transpose('presentation', 'neuroid')
    n = a.sizes['neuroid']
    if positions.shape[0] != n:
        raise ValueError(f"positions has {positions.shape[0]} rows but the assembly has "
                         f"{n} neuroids — they must match.")
    a = _flatten_neuroid(a)
    return a.assign_coords(tissue_x=('neuroid', positions[:, 0]),
                           tissue_y=('neuroid', positions[:, 1]))


def ensure_tissue_coords(assembly):
    """Return an assembly guaranteed to carry tissue coords: pass through if present,
    else attach a neutral grid (a non-topographic model then scores near the null)."""
    if has_tissue_coords(assembly):
        return assembly.transpose('presentation', 'neuroid')
    a = assembly.transpose('presentation', 'neuroid')
    return attach_tissue_coords(a, grid_positions(a.sizes['neuroid']))


def shuffle_tissue_coords(assembly, seed: int = 0):
    """The matched topographic NULL: permute unit positions across neuroids. Responses
    (hence predictivity) are untouched; only the spatial layout is destroyed."""
    a = ensure_tissue_coords(assembly)
    rng = np.random.RandomState(seed)
    perm = rng.permutation(a.sizes['neuroid'])
    shuffled = np.stack([a['tissue_x'].values[perm], a['tissue_y'].values[perm]], axis=1)
    return attach_tissue_coords(a, shuffled)   # attach_tissue_coords flattens the index
