"""Map model-predicted per-parcel / per-voxel scores onto cortical
visualizations — the "model response on the brain" figures in the MIRAGE style.

Two rendering paths:

  * :func:`parcel_grid_heatmap` and :func:`network_strip` are dependency-light
    (matplotlib only). They always render, so they are safe for tests, CI, and
    as a fallback when surface assets are unavailable.
  * :func:`cortical_surface_map` renders a real inflated-cortex surface coloured
    by per-parcel value (Schaefer 2018 parcellation on fsaverage), using nilearn
    + nibabel. This is the figure that looks like the MIRAGE website. It is run
    on a machine that has the surface assets (EC2); nilearn/nibabel are imported
    lazily so importing this module never requires them.

All renderers take a 1-D array of per-parcel (or per-voxel) values — typically a
benchmark's per-target Pearson r — and a colormap, and either return a
matplotlib Figure or save a PNG.
"""
from typing import Optional, Sequence, Tuple

import numpy as np


# Schaefer 2018 7-network order; the 1000-parcel atlas is 500/hemi, each parcel
# named e.g. '7Networks_LH_Vis_1'. The seven networks in canonical order:
SCHAEFER_7NETWORKS = ['Vis', 'SomMot', 'DorsAttn', 'SalVentAttn', 'Limbic',
                      'Cont', 'Default']


def normalize_values(values: np.ndarray, vmin: Optional[float] = None,
                     vmax: Optional[float] = None, clip: bool = True) -> np.ndarray:
    """Scale ``values`` to ``[0, 1]`` for colormapping. NaNs are preserved."""
    values = np.asarray(values, dtype=float)
    finite = values[np.isfinite(values)]
    if vmin is None:
        vmin = float(np.nanmin(finite)) if finite.size else 0.0
    if vmax is None:
        vmax = float(np.nanmax(finite)) if finite.size else 1.0
    if vmax <= vmin:
        vmax = vmin + 1e-9
    out = (values - vmin) / (vmax - vmin)
    if clip:
        out = np.clip(out, 0.0, 1.0)
    return out


def _network_of_parcel(parcel_names: Sequence[str]) -> np.ndarray:
    """Map each Schaefer parcel name to its 7-network index (0..6), -1 if unknown."""
    idx = np.full(len(parcel_names), -1, dtype=int)
    for i, name in enumerate(parcel_names):
        for n, net in enumerate(SCHAEFER_7NETWORKS):
            if net.lower() in str(name).lower():
                idx[i] = n
                break
    return idx


def parcel_grid_heatmap(values: np.ndarray, *, ncols: int = 40,
                        cmap: str = 'inferno', title: Optional[str] = None,
                        vmin: Optional[float] = None, vmax: Optional[float] = None,
                        out_png: Optional[str] = None):
    """Render per-parcel values as a compact 2-D heatmap (parcels row-major).

    nilearn-free and always works. Not anatomical, but a faithful, glanceable
    picture of "how well each target is predicted" — useful as a thumbnail and
    as a guaranteed fallback for the surface map.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    values = np.asarray(values, dtype=float)
    n = values.size
    nrows = int(np.ceil(n / ncols))
    grid = np.full(nrows * ncols, np.nan)
    grid[:n] = values
    grid = grid.reshape(nrows, ncols)

    fig, ax = plt.subplots(figsize=(ncols * 0.18, nrows * 0.18 + 0.6))
    im = ax.imshow(grid, cmap=cmap, vmin=vmin, vmax=vmax, aspect='equal')
    ax.set_xticks([]); ax.set_yticks([])
    if title:
        ax.set_title(title, fontsize=11)
    cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cb.set_label('value', fontsize=9)
    fig.tight_layout()
    if out_png:
        fig.savefig(out_png, dpi=150, bbox_inches='tight')
        plt.close(fig)
        return out_png
    return fig


def network_strip(values: np.ndarray, parcel_names: Sequence[str], *,
                  cmap: str = 'inferno', title: Optional[str] = None,
                  out_png: Optional[str] = None):
    """Bar plot of mean value per Schaefer 7-network, with error bars.

    Answers "which functional networks does this model predict best?" — the
    summary that accompanies a surface map. nilearn-free.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    values = np.asarray(values, dtype=float)
    net_idx = _network_of_parcel(parcel_names)
    means, sems, labels = [], [], []
    for n, net in enumerate(SCHAEFER_7NETWORKS):
        v = values[net_idx == n]
        v = v[np.isfinite(v)]
        if v.size:
            means.append(float(v.mean()))
            sems.append(float(v.std() / np.sqrt(v.size)))
            labels.append(net)
    fig, ax = plt.subplots(figsize=(6, 3.2))
    x = np.arange(len(labels))
    colors = plt.get_cmap(cmap)(normalize_values(np.array(means)))
    ax.bar(x, means, yerr=sems, capsize=3, color=colors)
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=30, ha='right', fontsize=9)
    ax.set_ylabel('mean value', fontsize=10)
    if title:
        ax.set_title(title, fontsize=11)
    ax.spines[['top', 'right']].set_visible(False)
    fig.tight_layout()
    if out_png:
        fig.savefig(out_png, dpi=150, bbox_inches='tight')
        plt.close(fig)
        return out_png
    return fig


# ---------------------------------------------------------------------------
# Real cortical-surface rendering (nilearn + nibabel; assets fetched on demand).
# ---------------------------------------------------------------------------

def parcels_to_vertices(labels: np.ndarray, values_h: np.ndarray,
                        per_hemi: int) -> np.ndarray:
    """Scatter per-parcel values onto per-vertex values via a FreeSurfer annot.

    The annot's per-vertex ``labels`` are 1-indexed (0 = medial wall), so parcel
    ``j`` (0-based) corresponds to annot label ``j + 1``. This one-off-index is
    the single most error-prone line in the surface path, so it lives in its own
    pure, tested function. Vertices with no parcel (medial wall) stay ``NaN``.
    """
    labels = np.asarray(labels)
    values_h = np.asarray(values_h, dtype=float)
    vtx = np.full(labels.shape, np.nan)
    for j in range(per_hemi):
        vtx[labels == (j + 1)] = values_h[j]
    return vtx


def _safe_plot_surf(plotting, surf, vtx, *, hemi, view, bg_map, cmap,
                    threshold, vmin, vmax, title):
    """Call ``plot_surf_stat_map`` robustly across nilearn versions.

    nilearn's surface-plotting kwargs drift between releases (``darkness`` and
    ``bg_on_data`` were removed in 0.13). Try the rich call, then progressively
    drop optional kwargs on ``TypeError`` so the figure still renders.
    """
    attempts = [
        dict(hemi=hemi, view=view, bg_map=bg_map, cmap=cmap, threshold=threshold,
             vmin=vmin, vmax=vmax, colorbar=True, title=title, bg_on_data=True),
        dict(hemi=hemi, view=view, bg_map=bg_map, cmap=cmap, threshold=threshold,
             vmin=vmin, vmax=vmax, colorbar=True, title=title),
        dict(hemi=hemi, view=view, bg_map=bg_map, cmap=cmap, colorbar=True,
             title=title),
        dict(hemi=hemi, view=view, cmap=cmap, colorbar=True),
    ]
    last = None
    for kw in attempts:
        try:
            return plotting.plot_surf_stat_map(surf, vtx, **kw)
        except TypeError as e:
            last = e
            continue
    raise last

def fetch_schaefer_fsaverage_annot(n_parcels: int = 1000, networks: int = 7,
                                   resolution: str = 'fsaverage5',
                                   cache_dir: str = '/tmp/schaefer_annot'):
    """Download + load the Schaefer 2018 surface annotation for both hemispheres.

    Returns ``(lh_labels, rh_labels, lh_names, rh_names)`` where ``*_labels`` are
    per-vertex parcel indices (0 = medial wall) from the FreeSurfer ``.annot``.
    Files come from the CBIG/Yeo-lab release. Network/parcel value ``k`` maps to
    every vertex whose label == ``k`` (1-indexed in the annot, 0 = background).
    """
    import os
    import urllib.request
    import nibabel as nib

    os.makedirs(cache_dir, exist_ok=True)
    base = ("https://raw.githubusercontent.com/ThomasYeoLab/CBIG/master/"
            "stable_projects/brain_parcellation/Schaefer2018_LocalGlobal/"
            f"Parcellations/FreeSurfer5.3/{resolution}/label/")
    out = []
    names = []
    for hemi in ('lh', 'rh'):
        fname = f"{hemi}.Schaefer2018_{n_parcels}Parcels_{networks}Networks_order.annot"
        path = os.path.join(cache_dir, fname)
        if not os.path.exists(path):
            urllib.request.urlretrieve(base + fname, path)
        labels, _, label_names = nib.freesurfer.read_annot(path)
        out.append(labels)
        names.append([n.decode() if isinstance(n, bytes) else n for n in label_names])
    return out[0], out[1], names[0], names[1]


def cortical_surface_map(parcel_values: np.ndarray, *, n_parcels: int = 1000,
                         networks: int = 7, resolution: str = 'fsaverage5',
                         hemi: str = 'left', view: str = 'lateral',
                         cmap: str = 'inferno', threshold: Optional[float] = None,
                         vmin: Optional[float] = None, vmax: Optional[float] = None,
                         title: Optional[str] = None, out_png: Optional[str] = None):
    """Render per-parcel values on an inflated fsaverage cortex (MIRAGE style).

    ``parcel_values`` is length ``n_parcels`` (500/hemi in canonical LH-then-RH
    order). nilearn + nibabel required; both imported lazily. Returns the PNG
    path (if ``out_png``) or the matplotlib Figure.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from nilearn import datasets, plotting

    parcel_values = np.asarray(parcel_values, dtype=float)
    lh_labels, rh_labels, lh_names, rh_names = fetch_schaefer_fsaverage_annot(
        n_parcels=n_parcels, networks=networks, resolution=resolution)
    fsavg = datasets.fetch_surf_fsaverage(mesh=resolution)

    per_hemi = n_parcels // 2
    if hemi == 'left':
        labels, values_h, surf, bg = lh_labels, parcel_values[:per_hemi], \
            fsavg['infl_left'], fsavg['sulc_left']
    else:
        labels, values_h, surf, bg = rh_labels, parcel_values[per_hemi:], \
            fsavg['infl_right'], fsavg['sulc_right']

    vtx = parcels_to_vertices(labels, values_h, per_hemi)

    fig = _safe_plot_surf(plotting, surf, vtx, hemi=hemi, view=view, bg_map=bg,
                          cmap=cmap, threshold=threshold, vmin=vmin, vmax=vmax,
                          title=title)
    if out_png:
        fig.savefig(out_png, dpi=150, bbox_inches='tight')
        plt.close(fig)
        return out_png
    return fig


def cortical_surface_movie(parcel_values_t: np.ndarray, *, times: Optional[Sequence[float]] = None,
                           n_parcels: int = 1000, networks: int = 7,
                           resolution: str = 'fsaverage5', hemi: str = 'left',
                           view: str = 'lateral', cmap: str = 'inferno',
                           threshold: Optional[float] = None,
                           vmin: Optional[float] = None, vmax: Optional[float] = None,
                           share_scale: bool = True, out_dir: Optional[str] = None,
                           prefix: str = 'frame', title_fmt: Optional[str] = 't = {t:.1f}s'):
    """Render an EVOLVING cortical surface — one frame per timepoint — for a
    sequence of per-parcel maps (e.g. a model's predicted BOLD across a movie
    clip, or measured BOLD across TRs).

    Reuses :func:`cortical_surface_map` for each frame. ``parcel_values_t`` is
    ``(T, n_parcels)`` in canonical LH-then-RH parcel order. With
    ``share_scale`` (default), a single ``vmin``/``vmax`` (2nd/98th percentile
    over ALL frames) is held fixed across the sequence, so the animation reflects
    real change over time rather than per-frame renormalization. ``times`` labels
    each frame (seconds); ``title_fmt`` formats it. Returns the list of PNG paths
    (when ``out_dir`` is given) or the list of matplotlib Figures.
    """
    import os
    arr = np.asarray(parcel_values_t, dtype=float)
    if arr.ndim != 2:
        raise ValueError(f"parcel_values_t must be 2-D (T, n_parcels); got {arr.shape}")
    if arr.shape[1] != n_parcels:
        raise ValueError(f"expected {n_parcels} parcels per frame; got {arr.shape[1]}")
    T = arr.shape[0]
    if share_scale:
        finite = arr[np.isfinite(arr)]
        if finite.size:
            if vmin is None:
                vmin = float(np.nanpercentile(finite, 2))
            if vmax is None:
                vmax = float(np.nanpercentile(finite, 98))
    times = list(times) if times is not None else list(range(T))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    frames = []
    for i in range(T):
        title = title_fmt.format(t=times[i]) if title_fmt else None
        png = os.path.join(out_dir, f'{prefix}_{i:03d}.png') if out_dir else None
        frames.append(cortical_surface_map(
            arr[i], n_parcels=n_parcels, networks=networks, resolution=resolution,
            hemi=hemi, view=view, cmap=cmap, threshold=threshold,
            vmin=vmin, vmax=vmax, title=title, out_png=png))
    return frames


def voxel_surface_map(voxel_values: np.ndarray, *, resolution: str = 'fsaverage5',
                      hemi: str = 'left', view: str = 'lateral',
                      cmap: str = 'inferno', threshold: Optional[float] = None,
                      vmin: Optional[float] = None, vmax: Optional[float] = None,
                      title: Optional[str] = None, out_png: Optional[str] = None):
    """Render per-vertex values directly on fsaverage5 (for voxelwise benchmarks
    like Lahner2024 whose targets already live on the fsaverage5 surface).

    ``voxel_values`` is per-vertex for one hemisphere (10242 on fsaverage5), or
    the full both-hemi vector (20484) from which the requested hemi is sliced.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from nilearn import datasets, plotting

    voxel_values = np.asarray(voxel_values, dtype=float)
    fsavg = datasets.fetch_surf_fsaverage(mesh=resolution)
    n_per_hemi = 10242 if resolution == 'fsaverage5' else voxel_values.size // 2
    if voxel_values.size == 2 * n_per_hemi:
        vtx = voxel_values[:n_per_hemi] if hemi == 'left' else voxel_values[n_per_hemi:]
    else:
        vtx = voxel_values
    surf = fsavg['infl_left'] if hemi == 'left' else fsavg['infl_right']
    bg = fsavg['sulc_left'] if hemi == 'left' else fsavg['sulc_right']

    fig = _safe_plot_surf(plotting, surf, vtx, hemi=hemi, view=view, bg_map=bg,
                          cmap=cmap, threshold=threshold, vmin=vmin, vmax=vmax,
                          title=title)
    if out_png:
        fig.savefig(out_png, dpi=150, bbox_inches='tight')
        plt.close(fig)
        return out_png
    return fig


# ---------------------------------------------------------------------------
# Optional quickbrain backend (pni-lab/quickbrain) — fast one-call brain outline.
# ---------------------------------------------------------------------------

QUICKBRAIN_INSTALL_HINT = (
    'quickbrain is an optional dependency. Install it with:\n'
    '  pip install "quickbrain @ git+https://github.com/pni-lab/quickbrain.git"\n'
    'Or use cortical_surface_map() (nilearn surface) / parcel_grid_heatmap() '
    '(no extra deps) instead.')


def parcels_to_nifti(parcel_values: np.ndarray, *, n_parcels: int = 1000,
                     networks: int = 7):
    """Project per-parcel Schaefer values onto the volumetric atlas as a NIfTI.

    quickbrain (and most volumetric viewers) want a NIfTI; our scores are
    per-parcel. This maps parcel ``j`` -> every voxel labelled ``j+1`` in the
    nilearn Schaefer 2018 volumetric atlas. Background voxels are ``NaN``.
    Returns a ``nibabel.Nifti1Image``. nibabel + nilearn imported lazily.
    """
    import nibabel as nib
    from nilearn import datasets

    parcel_values = np.asarray(parcel_values, dtype=float)
    atlas = datasets.fetch_atlas_schaefer_2018(n_rois=n_parcels,
                                               yeo_networks=networks)
    maps = atlas['maps']
    atlas_img = nib.load(maps) if isinstance(maps, (str, bytes)) else maps
    labels = np.asarray(atlas_img.get_fdata()).astype(int)   # voxel -> 1..N (0 bg)
    vol = np.full(labels.shape, np.nan, dtype=float)
    for j in range(n_parcels):
        vol[labels == (j + 1)] = parcel_values[j]
    return nib.Nifti1Image(vol, atlas_img.affine, atlas_img.header)


def _safe_plot_brain(quickbrain, img, *, cmap, vmin, vmax, background,
                     extra_kwargs):
    """Call ``quickbrain.plot_brain`` robustly across quickbrain versions.

    quickbrain's signature drifts between releases (``vmin``/``vmax`` and
    ``background`` are not in every version). Try the rich call, then drop the
    optional kwargs progressively on ``TypeError`` so the glass brain still
    renders — mirrors :func:`_safe_plot_surf` for the surface path.
    """
    attempts = [
        dict(cmap=cmap, vmin=vmin, vmax=vmax, background=background, **extra_kwargs),
        dict(cmap=cmap, vmin=vmin, vmax=vmax, **extra_kwargs),
        dict(cmap=cmap, background=background, **extra_kwargs),
        dict(cmap=cmap, **extra_kwargs),
        dict(**extra_kwargs),
    ]
    last = None
    for kw in attempts:
        # Drop any None-valued optional kwargs so they don't shadow defaults.
        kw = {k: v for k, v in kw.items() if v is not None}
        try:
            return quickbrain.plot_brain(img, **kw)
        except TypeError as e:
            last = e
            continue
    raise last


def quickbrain_outline_map(parcel_values: np.ndarray, *, n_parcels: int = 1000,
                           networks: int = 7, title: Optional[str] = None,
                           cmap: str = 'turbo', vmin: Optional[float] = None,
                           vmax: Optional[float] = None,
                           out_png: Optional[str] = None, **quickbrain_kwargs):
    """Render per-parcel values on quickbrain's fast brain outline (OPTIONAL).

    Uses the optional ``quickbrain`` package for a stylized one-call glass-brain
    montage (lateral / posterior / dorsal views) with the activation overlaid.
    Per-parcel Schaefer values are first projected to a volumetric NIfTI
    (:func:`parcels_to_nifti`). ``cmap``/``vmin``/``vmax`` pin the colour scale —
    pass a diverging cmap (``'RdBu_r'``) with symmetric limits for signed data
    like BOLD. Any extra keyword arguments are forwarded to
    ``quickbrain.plot_brain``; unsupported ones are dropped gracefully. Raises a
    clear ``ImportError`` with install instructions when quickbrain is absent —
    it is never required; the nilearn surface and matplotlib fallbacks work.
    """
    try:
        import quickbrain
    except ImportError as e:
        raise ImportError(QUICKBRAIN_INSTALL_HINT) from e
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    img = parcels_to_nifti(parcel_values, n_parcels=n_parcels, networks=networks)
    background = quickbrain_kwargs.pop('background', 'white')
    result = _safe_plot_brain(quickbrain, img, cmap=cmap, vmin=vmin, vmax=vmax,
                              background=background, extra_kwargs=quickbrain_kwargs)
    fig = result if hasattr(result, 'savefig') else plt.gcf()
    if title:
        try:
            fig.suptitle(title)
        except Exception:
            pass
    if out_png:
        fig.savefig(out_png, dpi=150, bbox_inches='tight')
        plt.close(fig)
        return out_png
    return fig


# ---------------------------------------------------------------------------
# Glass-brain rendering (nilearn plot_glass_brain — MIP montage). No optional
# dependency: nilearn is already used by the surface path, and the volumetric
# NIfTI comes from parcels_to_nifti. This is the transparent-brain montage
# (lateral / posterior / dorsal projections with L/R labels + cerebellum).
# ---------------------------------------------------------------------------

def _safe_plot_glass_brain(plotting, img, *, display_mode, cmap, vmin, vmax,
                           plot_abs, threshold, title):
    """Call ``plot_glass_brain`` robustly across nilearn versions.

    ``vmin``/``plot_abs`` were added at different releases; drop the optional
    kwargs progressively on ``TypeError`` so the montage still renders. Mirrors
    :func:`_safe_plot_surf`.
    """
    attempts = [
        dict(display_mode=display_mode, colorbar=True, cmap=cmap, plot_abs=plot_abs,
             vmin=vmin, vmax=vmax, threshold=threshold, title=title),
        dict(display_mode=display_mode, colorbar=True, cmap=cmap, plot_abs=plot_abs,
             vmax=vmax, threshold=threshold, title=title),
        dict(display_mode=display_mode, colorbar=True, cmap=cmap, plot_abs=plot_abs,
             title=title),
        dict(display_mode=display_mode, colorbar=True, cmap=cmap, title=title),
    ]
    last = None
    for kw in attempts:
        kw = {k: v for k, v in kw.items() if v is not None}
        try:
            return plotting.plot_glass_brain(img, **kw)
        except TypeError as e:
            last = e
            continue
    raise last


def glass_brain_map(parcel_values: np.ndarray, *, n_parcels: int = 1000,
                    networks: int = 7, display_mode: str = 'ortho',
                    cmap: str = 'RdBu_r', vmin: Optional[float] = None,
                    vmax: Optional[float] = None, plot_abs: bool = False,
                    threshold: Optional[float] = None, title: Optional[str] = None,
                    out_png: Optional[str] = None):
    """Render per-parcel values as a nilearn glass-brain MIP montage.

    The transparent-brain projection figure: ``display_mode='ortho'`` gives the
    three views (sagittal / coronal / axial ≈ lateral / posterior / dorsal) with
    L/R labels and the cerebellum outline; ``'lyrz'`` adds a second lateral view.
    Per-parcel Schaefer values are projected to a volumetric NIfTI
    (:func:`parcels_to_nifti`). For signed data like BOLD pass a diverging
    ``cmap`` with ``plot_abs=False`` and symmetric ``vmin=-vmax``. nilearn
    imported lazily. Returns the PNG path (if ``out_png``) or the nilearn display.
    """
    import matplotlib
    matplotlib.use('Agg')
    from nilearn import plotting

    img = parcels_to_nifti(parcel_values, n_parcels=n_parcels, networks=networks)
    disp = _safe_plot_glass_brain(plotting, img, display_mode=display_mode,
                                  cmap=cmap, vmin=vmin, vmax=vmax, plot_abs=plot_abs,
                                  threshold=threshold, title=title)
    if out_png:
        disp.savefig(out_png, dpi=150)
        disp.close()
        return out_png
    return disp


def glass_brain_movie(parcel_values_t: np.ndarray, *,
                      times: Optional[Sequence[float]] = None,
                      n_parcels: int = 1000, networks: int = 7,
                      display_mode: str = 'ortho', cmap: str = 'RdBu_r',
                      vmin: Optional[float] = None, vmax: Optional[float] = None,
                      plot_abs: bool = False, symmetric: bool = True,
                      share_scale: bool = True, out_dir: Optional[str] = None,
                      prefix: str = 'frame',
                      title_fmt: Optional[str] = 't = {t:.1f}s'):
    """Render an EVOLVING glass-brain MIP montage — one multi-view frame per
    timepoint — for a sequence of per-parcel maps (e.g. measured or predicted
    BOLD across a movie clip).

    The glass-brain analogue of :func:`cortical_surface_movie`, reusing
    :func:`glass_brain_map` per frame. ``parcel_values_t`` is ``(T, n_parcels)``
    in canonical LH-then-RH parcel order. For signed BOLD the default
    ``cmap='RdBu_r'`` + ``symmetric=True`` centre the diverging scale on zero;
    with ``share_scale`` (default) one fixed scale spans the whole clip
    (``vmax`` = 98th percentile of ``|values|`` over ALL frames, ``vmin=-vmax``)
    so the montage shows real change over time, not per-frame renormalisation.
    ``times`` labels each frame; ``title_fmt`` formats it. Returns the list of
    PNG paths (when ``out_dir`` is given) or the list of nilearn displays.
    """
    import os
    arr = np.asarray(parcel_values_t, dtype=float)
    if arr.ndim != 2:
        raise ValueError(f"parcel_values_t must be 2-D (T, n_parcels); got {arr.shape}")
    if arr.shape[1] != n_parcels:
        raise ValueError(f"expected {n_parcels} parcels per frame; got {arr.shape[1]}")
    T = arr.shape[0]
    if share_scale:
        finite = arr[np.isfinite(arr)]
        if finite.size:
            if symmetric:
                if vmax is None:
                    vmax = float(np.nanpercentile(np.abs(finite), 98))
                if vmin is None:
                    vmin = -vmax
            else:
                if vmin is None:
                    vmin = float(np.nanpercentile(finite, 2))
                if vmax is None:
                    vmax = float(np.nanpercentile(finite, 98))
    times = list(times) if times is not None else list(range(T))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    frames = []
    for i in range(T):
        title = title_fmt.format(t=times[i]) if title_fmt else None
        png = os.path.join(out_dir, f'{prefix}_{i:03d}.png') if out_dir else None
        frames.append(glass_brain_map(
            arr[i], n_parcels=n_parcels, networks=networks, display_mode=display_mode,
            cmap=cmap, vmin=vmin, vmax=vmax, plot_abs=plot_abs, title=title,
            out_png=png))
    return frames


def quickbrain_outline_movie(parcel_values_t: np.ndarray, *,
                             times: Optional[Sequence[float]] = None,
                             n_parcels: int = 1000, networks: int = 7,
                             cmap: str = 'RdBu_r', vmin: Optional[float] = None,
                             vmax: Optional[float] = None, symmetric: bool = True,
                             share_scale: bool = True, out_dir: Optional[str] = None,
                             prefix: str = 'frame',
                             title_fmt: Optional[str] = 't = {t:.1f}s',
                             **quickbrain_kwargs):
    """Render an EVOLVING quickbrain glass-brain montage — one 3-view frame per
    timepoint — for a sequence of per-parcel maps (e.g. measured or predicted
    BOLD across a movie clip).

    The glass-brain analogue of :func:`cortical_surface_movie`, reusing
    :func:`quickbrain_outline_map` per frame. ``parcel_values_t`` is
    ``(T, n_parcels)`` in canonical LH-then-RH parcel order.

    For signed data like BOLD, the default ``cmap='RdBu_r'`` + ``symmetric=True``
    centre the diverging scale on zero so warm = above-mean and cool =
    below-mean response. With ``share_scale`` (default) a single scale is held
    fixed across the whole clip so the montage reflects real change over time
    rather than per-frame renormalisation: ``vmax`` is the 98th percentile of
    ``|values|`` over ALL frames and ``vmin = -vmax`` (symmetric), or the 2nd/98th
    percentile of the raw values (non-symmetric). ``times`` labels each frame
    (seconds); ``title_fmt`` formats it. Returns the list of PNG paths (when
    ``out_dir`` is given) or the list of quickbrain Figures.
    """
    import os
    arr = np.asarray(parcel_values_t, dtype=float)
    if arr.ndim != 2:
        raise ValueError(f"parcel_values_t must be 2-D (T, n_parcels); got {arr.shape}")
    if arr.shape[1] != n_parcels:
        raise ValueError(f"expected {n_parcels} parcels per frame; got {arr.shape[1]}")
    T = arr.shape[0]
    if share_scale:
        finite = arr[np.isfinite(arr)]
        if finite.size:
            if symmetric:
                if vmax is None:
                    vmax = float(np.nanpercentile(np.abs(finite), 98))
                if vmin is None:
                    vmin = -vmax
            else:
                if vmin is None:
                    vmin = float(np.nanpercentile(finite, 2))
                if vmax is None:
                    vmax = float(np.nanpercentile(finite, 98))
    times = list(times) if times is not None else list(range(T))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    frames = []
    for i in range(T):
        title = title_fmt.format(t=times[i]) if title_fmt else None
        png = os.path.join(out_dir, f'{prefix}_{i:03d}.png') if out_dir else None
        frames.append(quickbrain_outline_map(
            arr[i], n_parcels=n_parcels, networks=networks, cmap=cmap,
            vmin=vmin, vmax=vmax, title=title, out_png=png, **quickbrain_kwargs))
    return frames
