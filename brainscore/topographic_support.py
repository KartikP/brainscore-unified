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


# ── NSD fsaverage-surface target staging ───────────────────────────

def fsaverage_xyz(vertex_index, hemisphere, surf='infl', rh_offset=200.0):
    """Per-vertex (x,y,z) from the fsaverage template, keyed by
    ``(hemisphere, vertex_index)``. The right hemisphere is offset along x by
    ``rh_offset`` so the two hemispheres don't overlap (mixing them otherwise
    drops cross-hemisphere homology pairs into the far-distance bins). The
    inflated surface ('infl') is used so Euclidean distance approximates
    geodesic cortical distance better than the folded pial surface would.
    """
    from nilearn import surface, datasets
    fs = datasets.fetch_surf_fsaverage('fsaverage')   # fsaverage7, 163842 verts/hemi
    lh = np.asarray(surface.load_surf_mesh(fs[f'{surf}_left'])[0], dtype=float)
    rh = np.asarray(surface.load_surf_mesh(fs[f'{surf}_right'])[0], dtype=float)
    rh = rh + np.array([rh_offset, 0.0, 0.0])
    vidx = np.asarray(vertex_index, dtype=int)
    hemi = np.asarray(hemisphere)
    out = np.zeros((len(vidx), 3), dtype=float)
    lh_m = hemi == 'lh'
    out[lh_m] = lh[vidx[lh_m]]
    out[~lh_m] = rh[vidx[~lh_m]]
    return out


def stage_nsd_surface_target(assembly, *, subject, region, hemisphere='lh',
                             nc_threshold=10.0, surf='infl', rh_offset=200.0,
                             vertex_xyz_fn=fsaverage_xyz):
    """Build an NSD fsaverage-surface topographic brain target from a loaded
    surface assembly (``Allen2022_fmri_surface_*``).

    Steps: restrict to ONE subject + region (all subjects share fsaverage
    vertex indices, so stacking subjects would create duplicate positions —
    a single subject is required); optionally one hemisphere; filter to
    reliable vertices (``nc_testset`` > ``nc_threshold``); attach per-vertex
    ``tissue_x``/``tissue_y`` from ``vertex_xyz_fn``. Returns the staged
    ``(presentation, neuroid)`` assembly ready for ``TopographicBenchmark``.

    ``vertex_xyz_fn(vertex_index, hemisphere, surf=, rh_offset=) -> (n, 3)``
    defaults to :func:`fsaverage_xyz`; inject a stub to stage without nilearn.
    """
    a = assembly
    if 'time_bin' in a.dims:
        a = a.squeeze('time_bin', drop=True)
    mask = ((np.asarray(a['subject'].values) == subject)
            & (np.asarray(a['region'].values) == region))
    if hemisphere != 'both':
        mask = mask & (np.asarray(a['hemisphere'].values) == hemisphere)
    a = a.isel(neuroid=np.where(mask)[0])
    if a.sizes['neuroid'] == 0:
        raise ValueError(
            f"no vertices for subject={subject!r} region={region!r} "
            f"hemisphere={hemisphere!r} — check the assembly's coords.")

    keep = np.where(np.asarray(a['nc_testset'].values) > nc_threshold)[0]
    a = a.isel(neuroid=keep)
    if a.sizes['neuroid'] == 0:
        raise ValueError(
            f"no vertices survive nc_testset > {nc_threshold}; lower the threshold.")

    xyz = vertex_xyz_fn(a['vertex_index'].values, a['hemisphere'].values,
                        surf=surf, rh_offset=rh_offset)
    return attach_tissue_coords(a, xyz)   # stores tissue_x/tissue_y (first 2 cols)
