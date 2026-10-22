#!/usr/bin/env python
"""Per-TR BOLDMoments replay data (donor poster, moving-brain option 2).

Reuses the pooled V-JEPA v1 blocks.16 features extracted in Experiment A (one 1024-d
vector per clip) and fits a continuous-time encoding model on the single-trial
BOLD time series (OpenNeuro ds005165, TR 1.75 s, fsaverage5):
  features placed at clip onsets -> double-gamma HRF -> ridge per vertex
  BOLD z-scored per run (benchmark convention), leave-one-participant-out,
  ridge penalty chosen by inner leave-one-participant-out on the visual-ROI median r.
Saves held-out per-TR predicted and measured BOLD for display runs containing the
poster's demo clips, per-TR pattern correlations for every run, and summary metrics.
"""
import argparse, json, time, os
from pathlib import Path
import numpy as np

T0 = time.time()
def log(m): print(f'[{time.time()-T0:7.1f}s] {m}', flush=True)

def pearson_cols(Y, P):
    Yc = Y - Y.mean(0); Pc = P - P.mean(0)
    den = np.sqrt((Yc * Yc).sum(0) * (Pc * Pc).sum(0))
    with np.errstate(invalid='ignore', divide='ignore'):
        return np.where(den > 0, (Yc * Pc).sum(0) / den, np.nan)

def pearson_rows(Y, P):
    Yc = Y - Y.mean(1, keepdims=True); Pc = P - P.mean(1, keepdims=True)
    den = np.sqrt((Yc * Yc).sum(1) * (Pc * Pc).sum(1))
    with np.errstate(invalid='ignore', divide='ignore'):
        return np.where(den > 0, (Yc * Pc).sum(1) / den, np.nan)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--features', default='/opt/dlami/nvme/donor2/pooled_features.npz')
    ap.add_argument('--out', default='/opt/dlami/nvme/donor2/out_tr')
    ap.add_argument('--demo_ids', default='stimulus_1083,stimulus_1033,stimulus_1101,stimulus_1017,stimulus_0411,stimulus_1063,stimulus_1006,stimulus_0056,stimulus_0762')
    ap.add_argument('--grid', default='1e1,1e2,1e3,1e4,1e5,1e6')
    args = ap.parse_args()
    OUT = Path(args.out); (OUT / 'display_runs').mkdir(parents=True, exist_ok=True)
    grid = [float(g) for g in args.grid.split(',')]
    demo_ids = args.demo_ids.split(',')

    import brainscore  # noqa: F401  (registers plugins)
    from brainscore.benchmarks.lahner2024.benchmark_timeresolved import (
        Lahner2024BOLDMoments_timeresolved_visualROI, TR_SEC)
    from brainscore_core.temporal import double_gamma_hrf, hrf_convolve

    bench = Lahner2024BOLDMoments_timeresolved_visualROI()
    f = np.load(args.features, allow_pickle=True)
    clip_ids = [str(c) for c in f['clip_ids']]; F = f['video'].astype(np.float32)
    stim_idx = {c: i for i, c in enumerate(clip_ids)}
    log(f'features {F.shape} for {len(clip_ids)} clips')

    asm = bench.assembly
    log(f'assembly dims={asm.dims} sizes={dict(asm.sizes)}')
    n_runs = asm.sizes['presentation']; n_TR_max = asm.sizes['time_bin']
    subjects = np.asarray(asm['subject'].values).astype(str)
    sessions = np.asarray(asm['session'].values).astype(str)
    runs = np.asarray(asm['run'].values).astype(str)
    tasks = np.asarray(asm['task'].values).astype(str)
    n_valid = np.asarray(asm['n_valid_TR'].values).astype(int)

    # ---- design: features at onsets, HRF ----
    feature_ts = np.zeros((n_runs, n_TR_max, F.shape[1]), np.float32)
    run_events = []
    for r in range(n_runs):
        ev = bench._events_for_run(subjects[r], sessions[r], runs[r])
        evs = []
        for _, e in ev.iterrows():
            sid = str(e['stimulus_id']); onset = float(e['onset_sec'])
            evs.append({'stimulus_id': sid, 'onset_sec': onset, 'in_model': sid in stim_idx})
            if sid in stim_idx:
                tr = int(np.floor(onset / TR_SEC))
                if 0 <= tr < n_TR_max:
                    feature_ts[r, tr] += F[stim_idx[sid]]
        run_events.append(evs)
    hrf = double_gamma_hrf(duration_sec=32.0, sampling_rate_hz=1.0 / TR_SEC)
    for r in range(n_runs):
        feature_ts[r] = hrf_convolve(feature_ts[r], sampling_rate_hz=1.0 / TR_SEC, hrf=hrf)
    log('design built + HRF-convolved')

    X, Y, run_of_obs, subj_of_obs = bench._concatenate_with_mask(feature_ts, asm)
    Y = np.nan_to_num(Y).astype(np.float32); X = X.astype(np.float32)
    roi = bench.voxel_mask; roi_idx = np.flatnonzero(roi)
    log(f'X {X.shape} Y {Y.shape} ROI {roi.sum()} vertices')
    usubj = sorted(set(subj_of_obs)); S = len(usubj)
    subj_obs = {s: np.flatnonzero(subj_of_obs == s) for s in usubj}

    # ---- per-participant sufficient statistics (closed-form ridge with centering) ----
    stats = {}
    for s in usubj:
        o = subj_obs[s]; Xs = X[o]; Ys = Y[o]
        stats[s] = dict(n=len(o), sx=Xs.sum(0), sy=Ys.sum(0), xtx=Xs.T @ Xs, xty=Xs.T @ Ys)
    log('sufficient statistics done')

    def fit(train):
        n = sum(stats[s]['n'] for s in train)
        sx = sum(stats[s]['sx'] for s in train); sy = sum(stats[s]['sy'] for s in train)
        mx = sx / n; my = sy / n
        xtx = sum(stats[s]['xtx'] for s in train) - n * np.outer(mx, mx)
        xty = sum(stats[s]['xty'] for s in train) - n * np.outer(mx, my)
        return mx, my, xtx, xty

    def solve(xtx, xty, lam, cols=None):
        A = xtx + lam * np.eye(xtx.shape[0], dtype=xtx.dtype)
        return np.linalg.solve(A, xty if cols is None else xty[:, cols])

    pred = np.zeros_like(Y, dtype=np.float16)   # held-out per-TR predictions, every run
    chosen = {}; per_subj = {}
    for s in usubj:
        train = [t for t in usubj if t != s]
        # inner: leave one training participant out, evaluate on ROI vertices only
        score = {lam: [] for lam in grid}
        for v in train:
            inner = [t for t in train if t != v]
            mx, my, xtx, xty = fit(inner)
            o = subj_obs[v]; Xv = X[o] - mx
            for lam in grid:
                W = solve(xtx, xty, lam, roi_idx)
                P = Xv @ W + my[roi_idx]
                score[lam].append(float(np.nanmedian(pearson_cols(Y[o][:, roi_idx], P))))
        best = max(grid, key=lambda l: np.mean(score[l]))
        chosen[s] = {'alpha': best, 'inner_median_roi_r_by_alpha': {str(l): float(np.mean(score[l])) for l in grid}}
        mx, my, xtx, xty = fit(train)
        W = solve(xtx, xty, best)
        o = subj_obs[s]; P = (X[o] - mx) @ W + my
        pred[o] = P.astype(np.float16)
        r = pearson_cols(Y[o], P)
        per_subj[s] = {'median_r_roi': float(np.nanmedian(r[roi])), 'median_r_whole': float(np.nanmedian(r)),
                       'n_obs': int(len(o))}
        log(f'participant {s}: alpha={best:g} ROI median r={per_subj[s]["median_r_roi"]:.4f} whole={per_subj[s]["median_r_whole"]:.4f}')

    Pall = pred.astype(np.float32)
    r_pooled = pearson_cols(Y, Pall)
    np.save(OUT / 'per_vertex_r_pooled.npy', r_pooled.astype(np.float32))
    spatial_r_roi = pearson_rows(Y[:, roi_idx], Pall[:, roi_idx]).astype(np.float32)
    np.savez(OUT / 'per_tr_spatial_r.npz', spatial_r_roi=spatial_r_roi, run_of_obs=run_of_obs.astype(np.int32),
             subject_of_obs=np.array(subj_of_obs, dtype=str))
    summary = {
        'protocol': {'features': 'V-JEPA v1 ViT-L/16 backbone.blocks.16, mean over 8 tubelets (Experiment A pooled_features.npz)',
                     'target': 'single-trial BOLD, fsaverage5 20484 vertices, z-scored per run (benchmark convention), TR 1.75 s',
                     'design': 'features at clip onset TR, double-gamma HRF (32 s kernel)',
                     'cv': 'leave-one-participant-out (10 folds); penalty by inner leave-one-participant-out on visual-ROI median r',
                     'alpha_grid': grid, 'n_obs': int(len(Y)), 'n_runs': int(n_runs), 'roi_n': int(roi.sum())},
        'pooled': {'median_r_roi': float(np.nanmedian(r_pooled[roi])), 'median_r_whole': float(np.nanmedian(r_pooled)),
                   'median_per_tr_spatial_r_roi': float(np.nanmedian(spatial_r_roi))},
        'per_participant': per_subj, 'alpha_per_fold': chosen,
    }
    log(f'POOLED ROI median r={summary["pooled"]["median_r_roi"]:.4f} whole={summary["pooled"]["median_r_whole"]:.4f} per-TR spatial r={summary["pooled"]["median_per_tr_spatial_r_roi"]:.4f}')

    # ---- display runs: best participant overall, their runs containing each demo clip ----
    best_subj = max(per_subj, key=lambda s: per_subj[s]['median_r_roi'])
    display = []
    seen = set()
    for d in demo_ids:
        cands = [r for r in range(n_runs) if subjects[r] == best_subj and any(e['stimulus_id'] == d for e in run_events[r])]
        if not cands:
            log(f'  demo clip {d}: no run for participant {best_subj}'); continue
        r = cands[0]
        if r in seen: display.append({'demo_clip': d, 'run_index': int(r), 'shared_with_previous': True}); continue
        seen.add(r)
        o = np.flatnonzero(run_of_obs == r)
        name = f'run{r:03d}_sub{subjects[r]}_ses{sessions[r]}_{tasks[r]}_run{runs[r]}'
        np.savez(OUT / 'display_runs' / f'{name}.npz',
                 measured_z=Y[o].astype(np.float16), predicted=pred[o],
                 spatial_r_roi=spatial_r_roi[o],
                 events=json.dumps(run_events[r]), tr_sec=TR_SEC, n_valid_TR=int(n_valid[r]))
        display.append({'demo_clip': d, 'run_index': int(r), 'file': f'display_runs/{name}.npz', 'subject': subjects[r],
                        'session': sessions[r], 'task': tasks[r], 'run': runs[r], 'n_TR': int(len(o)),
                        'run_median_spatial_r_roi': float(np.nanmedian(spatial_r_roi[o]))})
    summary['display'] = {'rule': 'participant with the highest held-out visual-ROI median r; for each demo clip, that participant\'s first run containing it',
                          'participant': best_subj, 'runs': display}
    json.dump(summary, open(OUT / 'results_timeresolved.json', 'w'), indent=1)
    log('DONE')

if __name__ == '__main__':
    main()
