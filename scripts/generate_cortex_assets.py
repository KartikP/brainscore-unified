"""Generate REAL per-input cortex figures for the UMI site's input panel.

The panel's brain pictures are currently illustrative -- textbook anatomy, model-
independent. Every one of them can instead show the thing Brain-Score actually
measures: **where a model predicts the brain, and how well**. That number already
exists in ``score.attrs['per_voxel_r']``; nobody had rendered it.

Run on EC2 (needs the benchmark data). Writes ``cortex_real_*.png`` -- deliberately
NOT overwriting the illustrative assets, so the swap in data.js happens only after
someone has looked at the output.

    python unified/scripts/generate_cortex_assets.py --out /tmp/cortex_assets

What each figure means changes with this script, and the site copy must change with it:
    before: "roughly where this kind of input drives activity"  (anatomy, no model)
    after:  "where <model> predicts this benchmark, per voxel"  (a measurement)
"""
import argparse
import json
import os
import traceback

import numpy as np


# (asset stem, benchmark identifier, model, kind) -- kind picks the renderer.
# 'vertex' = fsaverage5 vertex-wise (Lahner); 'parcel' = Schaefer 1000 (Algonauts).
TARGETS = [
    ('video', 'Lahner2024-fMRI-naturalistic-visualROI', 'vjepa1-vitl', 'vertex'),
    ('videoaudio', 'Lahner2024-fMRI-naturalistic-multimodal-visualROI',
     'vjepa1-wav2vec2', 'vertex'),
    # Algonauts is per-PARCEL (Schaefer 1000), so it renders through
    # cortical_surface_map rather than the vertex path. All three inputs below are
    # the same benchmark and the same subject -- what differs is which towers the
    # site row is describing, so the figure is legitimately shared.
    ('videoaudiotext', 'Algonauts2025-friends-sub01', 'clip-vit-b-32', 'parcel'),
]

# Benchmarks that cannot become a human cortical surface, and why. Recorded here so
# the omission is a documented decision rather than a gap someone re-discovers.
CANNOT_RENDER = {
    'image': 'MajajHong2015 is macaque V4/IT array recordings -- there is no human '
             'cortical surface to paint. A macaque schematic or no figure are the '
             'honest options.',
    'live loop': 'the grid game is scored on win rate; there is no brain prediction '
                 'to map at all.',
}


def render_vertex(score, stem, out_dir, model_name, benchmark_id):
    """Lahner-style: per-voxel r on fsaverage5, masked to the scored voxels."""
    from brainscore.visualization.brain_map import vertex_surface_map

    per_voxel = np.asarray(score.attrs['per_voxel_r'], dtype=float)
    n_total = int(score.attrs.get('voxel_mask_n_total', 20484))
    mask = np.zeros(n_total, dtype=bool)
    values = np.full(n_total, np.nan, dtype=float)

    # per_voxel_r is ordered over the SCORED voxels only; place it back on the full
    # surface via the benchmark's own mask so the geometry is right.
    kept = score.attrs.get('voxel_mask_indices')
    if kept is not None:
        idx = np.asarray(kept, dtype=int)
    elif per_voxel.size == n_total:
        idx = np.arange(n_total)
    else:
        raise RuntimeError(
            f"cannot place {per_voxel.size} scored voxels onto {n_total} vertices: "
            "the benchmark did not expose voxel_mask_indices. Add it to score.attrs "
            "rather than guessing an ordering here.")
    values[idx] = per_voxel
    mask[idx] = True

    vmax = float(np.nanpercentile(per_voxel, 99))
    paths = []
    for hemi in ('left', 'right'):
        for view in ('lateral', 'ventral'):
            out = os.path.join(out_dir, f'cortex_real_{stem}_{hemi}_{view}.png')
            vertex_surface_map(
                values, mask=mask, hemi=hemi, view=view, cmap='inferno',
                vmin=0.0, vmax=vmax,
                title=f'{model_name} · per-voxel r · {benchmark_id.split("-")[0]}',
                out_png=out)
            paths.append(out)
    return paths, {'vmax': vmax, 'n_voxels': int(per_voxel.size),
                   'median_r': float(np.nanmedian(per_voxel))}


def render_parcel(score, stem, out_dir, model_name, benchmark_id):
    """Algonauts-style: per-parcel r on Schaefer 1000, rendered on fsaverage."""
    from brainscore.visualization.brain_map import cortical_surface_map

    per_parcel = np.asarray(score.attrs['per_parcel_r'], dtype=float)
    if per_parcel.size != 1000:
        raise RuntimeError(
            f"expected 1000 Schaefer parcels, got {per_parcel.size}; the renderer "
            "assumes canonical LH-then-RH order and would mis-place them otherwise")
    vmax = float(np.nanpercentile(per_parcel, 99))
    paths = []
    for hemi in ('left', 'right'):
        for view in ('lateral', 'ventral'):
            out = os.path.join(out_dir, f'cortex_real_{stem}_{hemi}_{view}.png')
            cortical_surface_map(
                per_parcel, n_parcels=1000, hemi=hemi, view=view, cmap='inferno',
                vmin=0.0, vmax=vmax,
                title=f'{model_name} · per-parcel r · Algonauts2025',
                out_png=out)
            paths.append(out)
    return paths, {'vmax': vmax, 'n_parcels': int(per_parcel.size),
                   'median_r': float(np.nanmedian(per_parcel))}


def probe_pereira():
    """Can Pereira be rendered at all? Report, do not assume.

    Pereira2018 neuroids are subject-space language-network voxels. Rendering them on
    a shared cortical surface needs either surface vertex indices or MNI coordinates
    per neuroid. If neither is present the honest answer is that this input keeps a
    schematic, and that is a finding rather than a gap.
    """
    try:
        import brainscore
        bench = brainscore.load_benchmark('Pereira2018.243sentences-linear-unified')
        data = getattr(bench, 'data', None)
        if data is None:
            data = getattr(bench, '_target_assembly', None)
        if data is None:
            return {'renderable': False, 'reason': 'benchmark exposes no target assembly'}
        coords = sorted(str(c) for c in data.coords)
        spatial = [c for c in coords
                   if any(k in c.lower() for k in ('vertex', 'mni', 'x', 'y', 'z', 'voxel'))]
        return {'renderable': bool(spatial), 'neuroid_coords': coords[:25],
                'spatial_candidates': spatial,
                'reason': ('has spatial coords -> a surface mapping is possible'
                           if spatial else
                           'no vertex/MNI coords on neuroids -> cannot place on a shared '
                           'surface; keep the schematic')}
    except Exception as exc:
        return {'renderable': False, 'reason': f'probe failed: {str(exc)[:200]}'}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='/tmp/cortex_assets')
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    import brainscore

    manifest = {'rendered': [], 'failed': [], 'cannot_render': CANNOT_RENDER}
    for stem, benchmark_id, model_name, kind in TARGETS:
        try:
            print(f'--- {stem}: {model_name} on {benchmark_id}', flush=True)
            model = brainscore.load_model(model_name)
            score = brainscore.load_benchmark(benchmark_id)(model)
            if kind == 'vertex':
                paths, meta = render_vertex(score, stem, args.out, model_name, benchmark_id)
            elif kind == 'parcel':
                paths, meta = render_parcel(score, stem, args.out, model_name, benchmark_id)
            else:
                raise NotImplementedError(f'renderer {kind!r} not wired yet')
            manifest['rendered'].append({
                'stem': stem, 'benchmark': benchmark_id, 'model': model_name,
                'score': float(score), 'paths': [os.path.basename(p) for p in paths],
                **meta})
            print(f'    ok: {len(paths)} views, median r={meta["median_r"]:.4f}',
                  flush=True)
        except Exception as exc:
            manifest['failed'].append({'stem': stem, 'error': str(exc)[:400]})
            print(f'    FAILED: {exc}', flush=True)
            traceback.print_exc()

    print('--- probing whether Pereira (text) can be rendered at all', flush=True)
    manifest['pereira_probe'] = probe_pereira()
    print('    ', manifest['pereira_probe'].get('reason'), flush=True)

    with open(os.path.join(args.out, 'manifest.json'), 'w') as fh:
        json.dump(manifest, fh, indent=2)
    print('MANIFEST', json.dumps({k: v for k, v in manifest.items()
                                  if k != 'cannot_render'})[:500], flush=True)
    print('CORTEX_ASSETS_DONE', flush=True)


if __name__ == '__main__':
    main()
