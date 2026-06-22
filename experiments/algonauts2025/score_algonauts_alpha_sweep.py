"""α sweep for CLIP video-only on Algonauts2025-friends-sub01.

Reuses the cached features from the prior 0.1172 run. Modifies the
benchmark's Ridge α and re-runs only the regression step.

Output: /tmp/algonauts_alpha_sweep.json
"""
import json
import sys
import time
import resource

sys.path.insert(0, '/home/ubuntu/brain-score-unified/unified')

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold

t0 = time.time()
def log(msg):
    rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    print(f'[{time.time()-t0:7.1f}s rss={rss_mb:6.0f}MB] {msg}', flush=True)


import brainscore

log('loading model + benchmark...')
model = brainscore.load_model('clip-vit-b-32')
model._region_layer_map_dict['IT'] = 'post_layernorm'
b = brainscore.load_benchmark('Algonauts2025-friends-sub01')

log('expanding stim_set + extracting features (cached)...')
frame_stim_set = b._expand_to_per_TR_frames()
features, frame_ids = b._extract_per_TR_features(model, frame_stim_set)
log(f'features: {features.shape}')

log('aligning to assembly...')
X = b._align_features_to_assembly(features, frame_ids, frame_stim_set)

a_stim = list(b.assembly['stimulus_id'].values)
a_run = list(b.assembly['run'].values)
seen, run_idx_per_obs = {}, np.empty(len(a_stim), dtype=np.int64)
for i, (s, r) in enumerate(zip(a_stim, a_run)):
    key = (str(s), str(r))
    if key not in seen:
        seen[key] = len(seen)
    run_idx_per_obs[i] = seen[key]
n_runs = len(seen)
log(f'{n_runs} unique (stim, run) blocks')

X_stacked = b._apply_stimulus_window_and_hrf(X, run_idx_per_obs)

# Drop excluded samples per run
keep = np.ones(len(X_stacked), dtype=bool)
for ri in range(n_runs):
    run_mask = run_idx_per_obs == ri
    run_idxs = np.where(run_mask)[0]
    if len(run_idxs) == 0:
        continue
    for ki in range(b._excluded_samples_start):
        if ki < len(run_idxs):
            keep[run_idxs[ki]] = False
    for ki in range(b._excluded_samples_end):
        if ki < len(run_idxs):
            keep[run_idxs[-1 - ki]] = False
X_stacked = X_stacked[keep]
Y = b.assembly.values[keep]
run_idx_kept = run_idx_per_obs[keep]
log(f'after exclusions: X={X_stacked.shape} Y={Y.shape}')

# α sweep
ALPHAS = [1.0, 10.0, 100.0, 1000.0, 10000.0, 100000.0]
results = {}
unique_runs = np.unique(run_idx_kept)

for alpha in ALPHAS:
    log(f'α = {alpha:g}')
    kf = KFold(n_splits=5, shuffle=True, random_state=0)
    held_out_preds = np.full_like(Y, np.nan, dtype=np.float32)
    for fold_i, (tr_run_pos, te_run_pos) in enumerate(kf.split(unique_runs)):
        tr_runs = unique_runs[tr_run_pos]
        te_runs = unique_runs[te_run_pos]
        tr = np.isin(run_idx_kept, tr_runs)
        te = np.isin(run_idx_kept, te_runs)
        reg = Ridge(alpha=alpha).fit(X_stacked[tr], Y[tr])
        held_out_preds[te] = reg.predict(X_stacked[te]).astype(np.float32)
    valid = ~np.isnan(held_out_preds[:, 0])
    Yt = Y[valid]
    Yp = held_out_preds[valid]
    Yt_c = Yt - Yt.mean(axis=0, keepdims=True)
    Yp_c = Yp - Yp.mean(axis=0, keepdims=True)
    num = (Yt_c * Yp_c).sum(axis=0)
    den = np.sqrt((Yt_c ** 2).sum(axis=0) * (Yp_c ** 2).sum(axis=0))
    with np.errstate(divide='ignore', invalid='ignore'):
        per_voxel_r = np.where(den > 0, num / den, np.nan)
    pvr_finite = per_voxel_r[~np.isnan(per_voxel_r)]
    median_r = float(np.median(pvr_finite))
    mean_r = float(np.mean(pvr_finite))
    log(f'  α={alpha:g}: median={median_r:.4f}, mean={mean_r:.4f}')
    results[str(alpha)] = {'median_r': median_r, 'mean_r': mean_r}

with open('/tmp/algonauts_alpha_sweep.json', 'w') as f:
    json.dump(results, f, indent=2)
log('wrote /tmp/algonauts_alpha_sweep.json')
log('DONE')
