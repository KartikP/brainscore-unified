"""First real-fMRI run of the topographic-alignment axis.

Stages NSD-surface (allen2022_fmri_surface) as a topographic brain target — per-image
fsaverage-surface betas + per-vertex (x,y,z) from the fsaverage template — then scores a
model on it via the unified TopographicBenchmark (raw r(d)-profile alignment minus the
shuffle-coordinate null).

Pipeline:
  1. Load the NSD fsaverage surface assembly (train: 412 averaged images, 8 subjects).
  2. Restrict to ONE subject + a region. All subjects share fsaverage vertex indices, so
     stacking subjects would create duplicate positions — a single subject is required.
  3. Filter to reliable vertices (nc_testset threshold).
  4. Attach fsaverage (x,y,z) per vertex via nilearn (inflated surface; RH offset so the
     hemispheres don't overlap), keyed by (hemisphere, vertex_index).
  5. SANITY: the brain's correlation-vs-distance profile r(d) should decay with distance —
     the decisive check that the metric + coords work on real fMRI.
  6. Build a TopographicBenchmark + score a model. A standard (non-topographic) model has no
     intrinsic sheet, so it gets a neutral grid and is EXPECTED to sit near the shuffle null
     (the negative control + empirical null floor). A positive result needs a topographic
     model (Topo-Omni / TDANN) — a later step.

Env: brainscore-unified (editable installs of brainscore_vision + unified brainscore).
"""
import argparse
import json
import numpy as np


def fsaverage_xyz(vertex_index, hemisphere, surf='infl', rh_offset=200.0):
    """Per-vertex (x,y,z) from the fsaverage template, keyed by (hemisphere, vertex_index).
    RH is offset along x so the two hemispheres don't overlap (they are different hemispheres
    and should be far apart). 'infl' (inflated) is used so Euclidean distance approximates
    geodesic cortical distance better than the folded pial surface would."""
    from nilearn import surface, datasets
    fs = datasets.fetch_surf_fsaverage('fsaverage')        # 163842 vertices/hemi (fsaverage7)
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--region', default='IT')
    ap.add_argument('--subject', default='subj01')
    ap.add_argument('--hemisphere', default='lh', choices=('lh', 'rh', 'both'),
                    help="single hemisphere gives a clean graded r(d); 'both' mixes in "
                         "cross-hemisphere homology pairs at the far-distance bins")
    ap.add_argument('--nc_threshold', type=float, default=10.0)
    ap.add_argument('--model', default='clip-vit-b-32')
    ap.add_argument('--skip_model', action='store_true')
    ap.add_argument('--surf', default='infl')
    ap.add_argument('--out', default='/tmp/topographic_nsd_result.json')
    args = ap.parse_args()

    import brainscore
    from brainscore_vision import load_dataset, load_stimulus_set
    from brainscore.benchmarks.topographic import TopographicBenchmark
    from brainscore.metrics.topographic import correlation_distance_profile
    from brainscore.topographic_support import attach_tissue_coords

    print('[1] loading NSD fsaverage surface assembly (train)...', flush=True)
    asm = load_dataset('Allen2022_fmri_surface_train')
    if 'time_bin' in asm.dims:
        asm = asm.squeeze('time_bin', drop=True)
    print(f'    full: {dict(asm.sizes)}  coords={list(asm.coords)}', flush=True)

    print(f'[2] selecting {args.subject} / {args.region} ...', flush=True)
    m = (asm['subject'].values == args.subject) & (asm['region'].values == args.region)
    if args.hemisphere != 'both':
        m = m & (asm['hemisphere'].values == args.hemisphere)
    asm = asm.isel(neuroid=np.where(m)[0])
    print(f'    {asm.sizes["neuroid"]} vertices (hemisphere={args.hemisphere})', flush=True)

    print(f'[3] reliability filter nc_testset > {args.nc_threshold}% ...', flush=True)
    keep = np.where(np.asarray(asm['nc_testset'].values) > args.nc_threshold)[0]
    asm = asm.isel(neuroid=keep)
    print(f'    {asm.sizes["neuroid"]} reliable vertices', flush=True)

    print(f'[4] attaching fsaverage {args.surf} (x,y,z) ...', flush=True)
    xyz = fsaverage_xyz(asm['vertex_index'].values, asm['hemisphere'].values, surf=args.surf)
    brain = attach_tissue_coords(asm, xyz)

    print('[5] brain r(d) profile (should DECAY with distance):', flush=True)
    resp = np.asarray(brain.transpose('presentation', 'neuroid').values)
    centers, mean_corr, counts = correlation_distance_profile(resp, xyz, n_bins=12)
    for c, mc, n in zip(centers, mean_corr, counts):
        print(f'    d={c:.3f}  r={mc:.4f}  (n_pairs={n})', flush=True)
    decays = bool(np.nanmean(mean_corr[:3]) > np.nanmean(mean_corr[-3:]))
    print(f'    near > far correlation (topographic): {decays}', flush=True)

    result = {'region': args.region, 'subject': args.subject,
              'n_vertices': int(brain.sizes['neuroid']),
              'brain_rd_centers': [float(x) for x in centers],
              'brain_rd_corr': [float(x) for x in mean_corr],
              'brain_rd_decays': decays}

    if not args.skip_model:
        try:
            print(f'[6] scoring model {args.model} (negative control: expect ~null)...', flush=True)
            stimulus_set = load_stimulus_set('Allen2022_fmri_stim_train')
            bench = TopographicBenchmark(f'topographic-nsd-{args.region}',
                                         brain_assembly=brain, stimulus_set=stimulus_set,
                                         region=args.region)
            model = brainscore.load_model(args.model)
            score = bench(model)
            result.update({'model': args.model, 'raw': score.attrs['raw'],
                           'null': score.attrs['null'], 'signal': float(score)})
            print(f'    raw={score.attrs["raw"]:.4f}  null={score.attrs["null"]:.4f}  '
                  f'signal={float(score):.4f}', flush=True)
        except Exception as e:
            import traceback
            result['model_error'] = f'{type(e).__name__}: {e}'
            print('[6] MODEL SCORING FAILED:', flush=True)
            traceback.print_exc()

    json.dump(result, open(args.out, 'w'), indent=2)
    print('RESULT:', json.dumps(result, indent=2), flush=True)
    print(f'saved -> {args.out}', flush=True)


if __name__ == '__main__':
    main()
