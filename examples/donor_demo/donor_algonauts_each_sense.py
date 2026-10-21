"""Donor-demo experiment B: what does each sense contribute when people
watched a TV show WITH sound? (Algonauts 2025 / Courtois NeuroMod Friends,
4 subjects, Schaefer-1000 parcels, TR = 1.49 s.)

Runs ON EC2 ONLY, CPU-only, from the per-clip feature caches already on the
instance (CLIP ViT-B/32 post_layernorm video features at the TR midpoint frame;
Wav2Vec2-base audio mean-pooled per TR; all-MiniLM-L6-v2 text over a 5-TR
transcript window). Fixed extractors, separate and combined readouts.

Conditions (same subjects, parcels, runs, outer folds for every condition):
  video | audio | text | video+audio (banded) | video+audio+text (banded) | concat (single alpha)
Feature stacking: 5-TR stimulus window, HRF delay 3 TRs (identical for every
modality). Outer CV: 5 folds over whole runs (no TR of a run is split). Alpha
chosen by inner run-held-out validation inside each outer training fold.
Null: circular time shift of the targets within each run (misaligned timing).
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

T0 = time.time()
B = Path('/home/ubuntu/.brainio/algonauts2025')


def log(m):
    print(f'[{time.time() - T0:7.1f}s] {m}', flush=True)


def per_voxel_r(Y, P):
    Yc = Y - Y.mean(0)
    Pc = P - P.mean(0)
    num = (Yc * Pc).sum(0)
    den = np.sqrt((Yc ** 2).sum(0) * (Pc ** 2).sum(0))
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.where(den > 0, num / den, np.nan)


def stack_window(F, run_idx, W=5, D=3):
    """feature(t-D-(W-1-off)) for off in 0..W-1, zero where it crosses a run start."""
    n, nf = F.shape
    out = np.zeros((n, W * nf), np.float32)
    ar = np.arange(n)
    for off in range(W):
        shift = D + (W - 1 - off)
        src = ar - shift
        ok = (src >= 0)
        ok[ok] &= run_idx[src[ok]] == run_idx[ar[ok]]
        out[ok, off * nf:(off + 1) * nf] = F[src[ok]]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='/opt/dlami/nvme/donor_demo/out_algonauts')
    ap.add_argument('--subjects', default='01,02,03,05')
    ap.add_argument('--n_outer', type=int, default=5)
    ap.add_argument('--n_inner', type=int, default=3)
    ap.add_argument('--n_shift', type=int, default=20)
    ap.add_argument('--n_boot', type=int, default=500)
    args = ap.parse_args()
    OUT = Path(args.out)
    OUT.mkdir(parents=True, exist_ok=True)
    W, D = 5, 3
    GRID = (1e2, 1e3, 1e4, 1e5, 1e6)
    GRID_BANDED = (1e3, 1e4, 1e5)
    CONDS = ['video', 'audio', 'text', 'video+audio', 'video+audio+text', 'concat']

    ref = xr.open_dataset(B / 'algonauts2025_friends_sub01.nc')
    r_sid = ref['stimulus_id'].values.astype(str)
    r_t = ref['t_within_run'].values.astype(int)
    Xclip = np.load(B / 'video_features_clip.npz')['X'].astype(np.float32)   # aligned to sub-01 rows
    assert Xclip.shape[0] == len(r_sid)
    video_by_key = {(r_sid[i], int(r_t[i])): Xclip[i] for i in range(len(r_sid))}
    log(f'video rows {len(video_by_key)}')

    def modality_rows(sid, t, root, dim):
        cache = {}
        out = np.zeros((len(sid), dim), np.float32)
        present = np.zeros(len(sid), bool)
        for i in range(len(sid)):
            c = sid[i]
            if c not in cache:
                p = root / f'{c}.npy'
                cache[c] = np.load(p) if p.exists() else None
            arr = cache[c]
            if arr is None:
                continue
            out[i] = arr[min(int(t[i]), arr.shape[0] - 1)]
            present[i] = True
        return out, present

    results = {'subjects': {}, 'protocol': {
        'stimulus_window_TRs': W, 'hrf_delay_TRs': D, 'outer_folds_over_runs': args.n_outer,
        'inner_validation_cells_per_outer_fold': args.n_inner, 'alpha_grid_single': GRID, 'alpha_grid_banded': GRID_BANDED,
        'features': {'video': 'CLIP ViT-B/32 post_layernorm (768) at TR-midpoint frame',
                     'audio': 'Wav2Vec2-base last hidden (768), mean within TR',
                     'text': 'all-MiniLM-L6-v2 (384) on 5-TR right-aligned transcript window'},
        'targets': 'Schaefer-1000 parcel BOLD (Algonauts 2025 release preprocessing), Friends S1-S6 + Movie10 training runs',
        'null': f'circular shift of targets within each run by k TRs (k uniform in [20, n_run-20]), {args.n_shift} shifts, fixed alphas',
        'rows_excluded': 'first stimulus_window+hrf_delay TRs of each run, rows lacking any modality, NaN targets',
    }}

    for s in args.subjects.split(','):
        ds = xr.open_dataset(B / f'algonauts2025_friends_sub{s}.nc')
        Y = (ds['data'] if 'data' in ds else list(ds.data_vars.values())[0]).values.astype(np.float32)
        sid = ds['stimulus_id'].values.astype(str)
        run = ds['run'].values.astype(str)
        t = ds['t_within_run'].values.astype(int)
        n = len(sid)
        keys = {}
        run_idx = np.empty(n, np.int64)
        for i in range(n):
            run_idx[i] = keys.setdefault((sid[i], run[i]), len(keys))
        Fv = np.stack([video_by_key.get((sid[i], int(t[i])), np.zeros(768, np.float32)) for i in range(n)])
        pv = np.array([(sid[i], int(t[i])) in video_by_key for i in range(n)])
        Fa, pa = modality_rows(sid, t, B / 'audio_features_wav2vec2', 768)
        Ft, pt = modality_rows(sid, t, B / 'text_features_minilm', 384)
        blocks = {'video': stack_window(Fv, run_idx, W, D), 'audio': stack_window(Fa, run_idx, W, D),
                  'text': stack_window(Ft, run_idx, W, D)}
        first = np.zeros(n, bool)
        for k in range(len(keys)):
            rows = np.flatnonzero(run_idx == k)
            first[rows[:W + D]] = True
        keep = pv & pa & pt & ~first & ~np.isnan(Y).any(1)
        log(f'sub-{s}: n={n} runs={len(keys)} kept rows={keep.sum()}')
        Yk = Y[keep]
        rk = run_idx[keep]
        Xall = np.concatenate([blocks['video'][keep], blocks['audio'][keep], blocks['text'][keep]], 1)
        del blocks, Fv, Fa, Ft
        sizes = [W * 768, W * 768, W * 384]
        p = Xall.shape[1]
        uruns = np.unique(rk)
        rng = np.random.default_rng(0)
        outer_runs = np.array_split(rng.permutation(uruns), args.n_outer)
        fold_of_run = {int(r): k for k, rs in enumerate(outer_runs) for r in rs}
        # cells (k, j): outer fold k split into n_inner run groups
        cells = {}
        for k, rs in enumerate(outer_runs):
            for j, rr in enumerate(np.array_split(np.random.default_rng(100 + k).permutation(rs), args.n_inner)):
                cells[(k, j)] = np.isin(rk, rr)
        cell_X = {}
        for c, m in cells.items():
            Xc = Xall[m].astype(np.float64)
            cell_X[c] = {'n': int(m.sum()), 'sx': Xc.sum(0), 'XtX': Xc.T @ Xc}
            del Xc
        log(f'  XtX per cell done ({len(cells)} cells, p={p})')

        def cell_Y(Ymat):
            out = {}
            for c, m in cells.items():
                Xc = Xall[m].astype(np.float64)
                Yc = Ymat[m].astype(np.float64)
                out[c] = {'sy': Yc.sum(0), 'XtY': Xc.T @ Yc}
            return out

        cY = cell_Y(Yk)

        def agg(cell_list, cy):
            n_ = sum(cell_X[c]['n'] for c in cell_list)
            sx = sum(cell_X[c]['sx'] for c in cell_list)
            sy = sum(cy[c]['sy'] for c in cell_list)
            XtX = sum(cell_X[c]['XtX'] for c in cell_list)
            XtY = sum(cy[c]['XtY'] for c in cell_list)
            xm, ym = sx / n_, sy / n_
            return XtX - n_ * np.outer(xm, xm), XtY - n_ * np.outer(xm, ym), xm, ym

        col_of = {'video': np.arange(0, sizes[0]), 'audio': np.arange(sizes[0], sizes[0] + sizes[1]),
                  'text': np.arange(sizes[0] + sizes[1], p)}

        def cols_groups(cond):
            if cond == 'concat':
                return [np.concatenate([col_of['video'], col_of['audio'], col_of['text']])]
            return [col_of[m] for m in cond.split('+')]

        def solve(XtX_c, XtY_c, groups, alphas):
            cols = np.concatenate(groups)
            lam = np.concatenate([np.full(len(g), a) for g, a in zip(groups, alphas)])
            A = XtX_c[np.ix_(cols, cols)].copy()
            A[np.diag_indices_from(A)] += lam
            return np.linalg.solve(A, XtY_c[cols]), cols

        def predict(rows_mask, cols, Wm, xm, ym):
            return ((Xall[rows_mask][:, cols].astype(np.float64) - xm[cols]) @ Wm + ym).astype(np.float32)

        subj = {}
        preds_store = {}
        for cond in CONDS:
            groups = cols_groups(cond)
            grid = GRID_BANDED if len(groups) > 1 else GRID
            combos = list(product(grid, repeat=len(groups)))
            held = np.full(Yk.shape, np.nan, np.float32)
            alphas_per_fold = []
            for k in range(args.n_outer):
                train_cells = [c for c in cells if c[0] != k]
                # inner validation: hold out one cell from each of n_inner different training folds
                val_cells = [c for c in train_cells if c[1] == 0][:args.n_inner]
                score = {c: [] for c in combos}
                for vc in val_cells:
                    XtX_c, XtY_c, xm, ym = agg([c for c in train_cells if c != vc], cY)
                    Yv = Yk[cells[vc]]
                    for comb in combos:
                        Wm, cols = solve(XtX_c, XtY_c, groups, comb)
                        score[comb].append(np.nanmean(per_voxel_r(Yv, predict(cells[vc], cols, Wm, xm, ym))))
                best = max(combos, key=lambda c: np.mean(score[c]))
                alphas_per_fold.append(list(map(float, best)))
                XtX_c, XtY_c, xm, ym = agg(train_cells, cY)
                Wm, cols = solve(XtX_c, XtY_c, groups, best)
                te = np.isin(rk, outer_runs[k])
                held[te] = predict(te, cols, Wm, xm, ym)
            r = per_voxel_r(Yk, held)
            subj[cond] = {'median_r': float(np.nanmedian(r)), 'mean_r': float(np.nanmean(r)),
                          'alphas_per_fold': alphas_per_fold, 'n_obs': int(len(Yk)), 'n_runs': int(len(uruns))}
            np.save(OUT / f'per_parcel_r__sub{s}__{cond}.npy', r.astype(np.float32))
            preds_store[cond] = held
            log(f'  sub-{s} {cond}: median r={subj[cond]["median_r"]:.4f} alphas={alphas_per_fold}')
        np.savez_compressed(OUT / f'heldout_pred__sub{s}.npz', Y=Yk.astype(np.float16), run_idx=rk,
                            fold_of_run=np.array([fold_of_run[int(r)] for r in uruns]), uruns=uruns,
                            stimulus_id=sid[keep], t_within_run=t[keep],
                            **{f'pred__{c}': v.astype(np.float16) for c, v in preds_store.items()})
        # paired bootstrap over held-out runs
        boot = {c: [] for c in CONDS}
        run_rows = {int(r): np.flatnonzero(rk == r) for r in uruns}
        for b in range(args.n_boot):
            idx = np.concatenate([run_rows[int(r)] for r in rng.choice(uruns, len(uruns))])
            for c in CONDS:
                boot[c].append(np.nanmedian(per_voxel_r(Yk[idx], preds_store[c][idx])))
        boot = {c: np.array(v) for c, v in boot.items()}
        subj['run_bootstrap'] = {c: {'median_r_ci95': [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]} for c, v in boot.items()}
        for a, bb in (('video+audio', 'video'), ('video+audio+text', 'video+audio'), ('video+audio+text', 'video'),
                      ('audio', 'video'), ('text', 'video'), ('concat', 'video+audio+text')):
            d = boot[a] - boot[bb]
            subj['run_bootstrap'][f'{a}_minus_{bb}'] = {'mean': float(d.mean()), 'ci95': [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))],
                                                        'frac_positive': float((d > 0).mean())}
        # timing-shift null (fixed alphas; XtX reused, only XtY recomputed per shift)
        null = {c: [] for c in ('video', 'audio', 'video+audio+text')}
        for sh in range(args.n_shift):
            Ysh = np.empty_like(Yk)
            g = np.random.default_rng(777 + sh)
            for r in uruns:
                rows = run_rows[int(r)]
                Ysh[rows] = np.roll(Yk[rows], int(g.integers(20, max(21, len(rows) - 20))), axis=0)
            cYs = cell_Y(Ysh)
            for cond in null:
                groups = cols_groups(cond)
                held = np.full(Yk.shape, np.nan, np.float32)
                for k in range(args.n_outer):
                    XtX_c, XtY_c, xm, ym = agg([c for c in cells if c[0] != k], cYs)
                    Wm, cols = solve(XtX_c, XtY_c, groups, subj[cond]['alphas_per_fold'][k])
                    te = np.isin(rk, outer_runs[k])
                    held[te] = predict(te, cols, Wm, xm, ym)
                null[cond].append(float(np.nanmedian(per_voxel_r(Ysh, held))))
            log(f'  sub-{s} shift {sh}: ' + ', '.join(f'{c}={null[c][-1]:.4f}' for c in null))
        subj['timing_shift_null'] = {c: {'null_median_r': v, 'null_mean': float(np.mean(v)), 'observed': subj[c]['median_r'],
                                         'p_value': float((np.sum(np.array(v) >= subj[c]['median_r']) + 1) / (len(v) + 1))}
                                     for c, v in null.items()}
        results['subjects'][s] = subj
        json.dump(results, open(OUT / 'results_algonauts.json', 'w'), indent=1)
        del cell_X, cY, Xall, preds_store

    subs = list(results['subjects'])
    tab = {c: np.array([results['subjects'][s][c]['median_r'] for s in subs]) for c in CONDS}
    paired = {'per_subject_median_r': {c: v.tolist() for c, v in tab.items()}, 'subjects': subs}
    from scipy import stats
    for a, bb in (('video+audio', 'video'), ('video+audio+text', 'video+audio'), ('video+audio+text', 'video'), ('audio', 'video'), ('text', 'video')):
        d = tab[a] - tab[bb]
        try:
            w = float(stats.wilcoxon(d).pvalue)
        except ValueError:
            w = float('nan')
        paired[f'{a}_minus_{bb}'] = {'mean_diff': float(d.mean()), 'min': float(d.min()), 'max': float(d.max()),
                                     'n_positive': int((d > 0).sum()), 'n': int(len(d)), 'wilcoxon_p': w}
    results['paired_across_subjects'] = paired
    results['manifest'] = {
        'script_sha1': hashlib.sha1(open(__file__, 'rb').read()).hexdigest(),
        'finished_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        'feature_cache_files': {'video': str(B / 'video_features_clip.npz'), 'audio': str(B / 'audio_features_wav2vec2'), 'text': str(B / 'text_features_minilm')},
        'feature_cache_sha1_video_npz': hashlib.sha1(open(B / 'video_features_clip.npz', 'rb').read()).hexdigest(),
        'unified_commit': subprocess.run(['git', '-C', os.path.expanduser('~/brain-score-unified/unified'), 'rev-parse', 'HEAD'], capture_output=True, text=True).stdout.strip(),
        'python': sys.version, 'numpy': np.__version__,
    }
    json.dump(results, open(OUT / 'results_algonauts.json', 'w'), indent=1)
    log('DONE')


if __name__ == '__main__':
    main()
