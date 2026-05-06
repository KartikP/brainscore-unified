"""Score CLIP+Wav2Vec2+MiniLM (video+audio+text) on Algonauts2025-friends-sub01.

Loads cached per-TR features for each modality, builds the design matrix,
runs banded ridge with α grid CV, reports per-parcel Pearson median.

Output: /tmp/algonauts_multimodal.json
"""
import argparse
import json
import sys
import time
import resource
from pathlib import Path

sys.path.insert(0, '/home/ubuntu/brain-score-unified/unified')

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold

t0 = time.time()
def log(msg):
    rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    print(f'[{time.time()-t0:7.1f}s rss={rss_mb:6.0f}MB] {msg}', flush=True)


def load_per_TR_features(features_root: Path, clip_ids: list,
                         feature_dim: int) -> np.ndarray:
    """Read each clip's (n_TRs_clip, feature_dim) .npy and concat in
    the order assembly's TRs appear. clip_ids is the per-TR clip
    identifier from the assembly."""
    # Cache per-clip (loaded once)
    cache = {}
    rows = []
    for cid in clip_ids:
        if cid not in cache:
            p = features_root / f'{cid}.npy'
            if not p.exists():
                cache[cid] = None
            else:
                cache[cid] = np.load(p)
        cache[cid]  # pre-loaded
    # Walk per-TR; index into each clip's array based on t_within_run
    return cache


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--assembly_path',
                        default='/home/ubuntu/.brainio/algonauts2025/algonauts2025_friends_sub01.nc')
    parser.add_argument('--audio_features',
                        default='/home/ubuntu/.brainio/algonauts2025/audio_features_wav2vec2')
    parser.add_argument('--text_features',
                        default='/home/ubuntu/.brainio/algonauts2025/text_features_minilm')
    parser.add_argument('--video_features_cache',
                        default='/home/ubuntu/.brainio/algonauts2025/video_features_clip.npz',
                        help='Cached (X_video, frame_ids) from earlier run.')
    parser.add_argument('--out_json', default='/tmp/algonauts_multimodal.json')
    parser.add_argument('--mode', default='banded',
                        choices=['video_only', 'audio_only', 'text_only',
                                 'concat', 'banded'])
    parser.add_argument('--stimulus_window', type=int, default=5)
    parser.add_argument('--hrf_delay', type=int, default=3)
    parser.add_argument('--alpha_grid', type=str,
                        default='1,10,100,1000,10000,100000,1000000')
    parser.add_argument('--banded_alpha_grid', type=str,
                        default='100,1000,10000,100000',
                        help='Smaller grid to keep banded tractable (n^3 fits per fold).')
    args = parser.parse_args()

    log(f'mode={args.mode}')

    log('imports...')
    import xarray as xr
    import brainscore

    log('loading benchmark + assembly...')
    b = brainscore.load_benchmark('Algonauts2025-friends-sub01')
    a_stim = list(b.assembly['stimulus_id'].values)
    a_run = list(b.assembly['run'].values)
    a_t = list(b.assembly['t_within_run'].values)
    n_TRs = len(a_stim)
    log(f'  n_TRs in assembly: {n_TRs}')

    # Build per-modality X
    X_parts = []
    band_widths = []  # for banded ridge

    if args.mode in ('video_only', 'concat', 'banded'):
        log('loading video X (CLIP)...')
        cache_path = Path(args.video_features_cache)
        if cache_path.exists():
            X_v = np.load(cache_path)['X'].astype(np.float32)
            log(f'  loaded from cache: {X_v.shape}')
        else:
            log('  no cache — re-extracting...')
            model = brainscore.load_model('clip-vit-b-32')
            model._region_layer_map_dict['IT'] = 'post_layernorm'
            frame_stim_set = b._expand_to_per_TR_frames()
            feats_v, frame_ids = b._extract_per_TR_features(
                model, frame_stim_set)
            X_v = b._align_features_to_assembly(
                feats_v, frame_ids, frame_stim_set)
            np.savez_compressed(cache_path, X=X_v.astype(np.float32))
            log(f'  saved cache: {cache_path}')
        log(f'  video X: {X_v.shape}')
        X_parts.append(('video', X_v))

    def per_tr_load(features_root: Path, name: str):
        cache = {}
        feats = None
        first_dim = None
        for i in range(n_TRs):
            cid = str(a_stim[i])
            if cid not in cache:
                p = features_root / f'{cid}.npy'
                cache[cid] = np.load(p) if p.exists() else None
            arr = cache[cid]
            if arr is None:
                if feats is None:
                    raise RuntimeError(f'first clip {cid} missing for {name}')
                feats[i] = 0.0
                continue
            if feats is None:
                first_dim = arr.shape[1]
                feats = np.zeros((n_TRs, first_dim), dtype=np.float32)
            t = int(a_t[i])
            if t < arr.shape[0]:
                feats[i] = arr[t]
            else:
                feats[i] = arr[-1]  # fall back to last TR
        return feats

    if args.mode in ('audio_only', 'concat', 'banded'):
        log('loading audio X (Wav2Vec2)...')
        X_a = per_tr_load(Path(args.audio_features), 'audio')
        log(f'  audio X: {X_a.shape}')
        X_parts.append(('audio', X_a))

    if args.mode in ('text_only', 'concat', 'banded'):
        log('loading text X (MiniLM)...')
        X_t = per_tr_load(Path(args.text_features), 'text')
        log(f'  text X: {X_t.shape}')
        X_parts.append(('text', X_t))

    # Combine
    X = np.concatenate([X for _, X in X_parts], axis=1).astype(np.float32)
    band_widths = [X.shape[1] for _, X in X_parts]
    band_names = [name for name, _ in X_parts]
    log(f'X combined: {X.shape}, bands: {dict(zip(band_names, band_widths))}')

    # Build run_idx_per_obs
    seen, run_idx_per_obs = {}, np.empty(n_TRs, dtype=np.int64)
    for i, (s, r) in enumerate(zip(a_stim, a_run)):
        key = (str(s), str(r))
        if key not in seen:
            seen[key] = len(seen)
        run_idx_per_obs[i] = seen[key]
    n_runs = len(seen)
    log(f'n_runs: {n_runs}')

    # Stimulus window stacking + HRF delay (per-modality)
    log(f'stacking sw={args.stimulus_window}, hrf_delay={args.hrf_delay}...')
    W = args.stimulus_window
    D = args.hrf_delay
    X_stacked_parts = []
    for name, Xm in X_parts:
        n_feat = Xm.shape[1]
        X_st = np.zeros((n_TRs, W * n_feat), dtype=np.float32)
        for offset in range(W):
            shift = D + (W - 1 - offset)
            for i in range(n_TRs):
                src = i - shift
                if (src >= 0 and run_idx_per_obs[src]
                        == run_idx_per_obs[i]):
                    X_st[i, offset*n_feat:(offset+1)*n_feat] = Xm[src]
        X_stacked_parts.append(X_st)
        log(f'  {name} stacked: {X_st.shape}')
    X_stacked = np.concatenate(X_stacked_parts, axis=1).astype(np.float32)
    log(f'X_stacked: {X_stacked.shape}')

    # Drop excluded edges per run
    keep = np.ones(n_TRs, dtype=bool)
    for ri in range(n_runs):
        run_mask = run_idx_per_obs == ri
        idxs = np.where(run_mask)[0]
        if len(idxs) == 0:
            continue
        for ki in range(b._excluded_samples_start):
            if ki < len(idxs):
                keep[idxs[ki]] = False
        for ki in range(b._excluded_samples_end):
            if ki < len(idxs):
                keep[idxs[-1 - ki]] = False
    X_stacked = X_stacked[keep]
    Y = b.assembly.values[keep]
    run_idx_kept = run_idx_per_obs[keep]
    log(f'after exclusions: X={X_stacked.shape} Y={Y.shape}')

    alphas = [float(a) for a in args.alpha_grid.split(',')]
    unique_runs = np.unique(run_idx_kept)

    if args.mode != 'banded':
        # Single α grid sweep over the combined X
        results = {}
        for alpha in alphas:
            kf = KFold(n_splits=5, shuffle=True, random_state=0)
            held_out_preds = np.full_like(Y, np.nan, dtype=np.float32)
            for fold_i, (tr_pos, te_pos) in enumerate(kf.split(unique_runs)):
                tr_runs = unique_runs[tr_pos]
                te_runs = unique_runs[te_pos]
                tr = np.isin(run_idx_kept, tr_runs)
                te = np.isin(run_idx_kept, te_runs)
                reg = Ridge(alpha=alpha).fit(X_stacked[tr], Y[tr])
                held_out_preds[te] = reg.predict(X_stacked[te]).astype(
                    np.float32)
            valid = ~np.isnan(held_out_preds[:, 0])
            Yt = Y[valid]
            Yp = held_out_preds[valid]
            num = ((Yt - Yt.mean(0)) * (Yp - Yp.mean(0))).sum(axis=0)
            den = np.sqrt(((Yt - Yt.mean(0)) ** 2).sum(0)
                          * ((Yp - Yp.mean(0)) ** 2).sum(0))
            with np.errstate(divide='ignore', invalid='ignore'):
                pvr = np.where(den > 0, num / den, np.nan)
            pvr_f = pvr[~np.isnan(pvr)]
            log(f'  α={alpha:g}: median={float(np.median(pvr_f)):.4f}, '
                f'mean={float(np.mean(pvr_f)):.4f}')
            results[str(alpha)] = {
                'median_r': float(np.median(pvr_f)),
                'mean_r': float(np.mean(pvr_f)),
            }
    else:
        # Banded ridge: separate α per modality. Sweep α-tuples on a
        # held-out validation fold within the train set, pick the best
        # by mean Pearson, then refit on full train and score test.
        log('banded ridge — sweeping α-grid per band...')
        from itertools import product
        # Compute per-band col index ranges (post-stacking)
        offsets = []
        cur = 0
        for name, Xm in X_parts:
            width = Xm.shape[1] * W
            offsets.append((cur, cur + width))
            cur += width
        log(f'  band offsets in stacked X: {dict(zip(band_names, offsets))}')

        # Generate α tuples — use small grid to keep banded tractable
        banded_alphas = [float(a) for a in args.banded_alpha_grid.split(',')]
        n_bands = len(X_parts)
        tuples = list(product(banded_alphas, repeat=n_bands))
        log(f'  banded α-grid: {banded_alphas}, '
            f'{len(tuples)} tuples × 5 folds = {len(tuples)*5} fits')

        # 5-fold CV outer: pick best α-tuple based on inner train→val
        kf = KFold(n_splits=5, shuffle=True, random_state=0)
        held_out_preds = np.full_like(Y, np.nan, dtype=np.float32)
        chosen_alphas = []
        for fold_i, (tr_pos, te_pos) in enumerate(kf.split(unique_runs)):
            tr_runs = unique_runs[tr_pos]
            te_runs = unique_runs[te_pos]
            tr = np.isin(run_idx_kept, tr_runs)
            te = np.isin(run_idx_kept, te_runs)
            # Inner split: 80/20 of train
            inner_kf = KFold(n_splits=5, shuffle=True, random_state=fold_i)
            inner_tr_pos, inner_va_pos = next(iter(inner_kf.split(tr_runs)))
            inner_tr_runs = tr_runs[inner_tr_pos]
            inner_va_runs = tr_runs[inner_va_pos]
            i_tr = np.isin(run_idx_kept, inner_tr_runs)
            i_va = np.isin(run_idx_kept, inner_va_runs)

            X_itr, Y_itr = X_stacked[i_tr], Y[i_tr]
            X_iva, Y_iva = X_stacked[i_va], Y[i_va]

            # Hoist X^T X and X^T Y out of the α-tuple loop — only
            # diag(α) changes per tuple. ~Nx speedup on cartesian grid.
            XtX_base = X_itr.T @ X_itr  # (P, P)
            XtY = X_itr.T @ Y_itr  # (P, n_voxels)
            log(f'    fold {fold_i+1}: XtX_base={XtX_base.shape}, '
                f'sweeping {len(tuples)} α-tuples...')

            best_score = -np.inf
            best_tuple = None
            Y_iva_c = Y_iva - Y_iva.mean(0)
            Y_iva_var = (Y_iva_c ** 2).sum(0)
            for atup in tuples:
                # Build diag(α) — n_features-long, per-band
                diag_alpha = np.zeros(X_itr.shape[1], dtype=np.float32)
                for (lo, hi), a_val in zip(offsets, atup):
                    diag_alpha[lo:hi] = a_val
                XtX = XtX_base.copy()
                XtX[np.diag_indices_from(XtX)] += diag_alpha
                try:
                    W_sol = np.linalg.solve(XtX, XtY)
                except np.linalg.LinAlgError:
                    continue
                pred = X_iva @ W_sol
                pred_c = pred - pred.mean(0)
                num = (Y_iva_c * pred_c).sum(axis=0)
                den = np.sqrt(Y_iva_var * (pred_c ** 2).sum(0))
                with np.errstate(divide='ignore', invalid='ignore'):
                    val_r = np.where(den > 0, num / den, 0.0)
                mean_r = float(np.mean(val_r))
                if mean_r > best_score:
                    best_score = mean_r
                    best_tuple = atup
            log(f'    fold {fold_i+1}/5: best α-tuple = '
                f'{dict(zip(band_names, best_tuple))} '
                f'(val_mean_r={best_score:.4f})')
            chosen_alphas.append(best_tuple)

            # Refit on full train + score test
            diag_alpha = np.zeros(X_stacked.shape[1], dtype=np.float32)
            for (lo, hi), a_val in zip(offsets, best_tuple):
                diag_alpha[lo:hi] = a_val
            XtX = X_stacked[tr].T @ X_stacked[tr]
            XtY = X_stacked[tr].T @ Y[tr]
            XtX[np.diag_indices_from(XtX)] += diag_alpha
            W_sol = np.linalg.solve(XtX, XtY)
            held_out_preds[te] = (X_stacked[te] @ W_sol).astype(np.float32)

        valid = ~np.isnan(held_out_preds[:, 0])
        Yt = Y[valid]
        Yp = held_out_preds[valid]
        num = ((Yt - Yt.mean(0)) * (Yp - Yp.mean(0))).sum(axis=0)
        den = np.sqrt(((Yt - Yt.mean(0)) ** 2).sum(0)
                      * ((Yp - Yp.mean(0)) ** 2).sum(0))
        with np.errstate(divide='ignore', invalid='ignore'):
            pvr = np.where(den > 0, num / den, np.nan)
        pvr_f = pvr[~np.isnan(pvr)]
        log(f'BANDED: median={float(np.median(pvr_f)):.4f}, '
            f'mean={float(np.mean(pvr_f)):.4f}')
        results = {
            'mode': 'banded',
            'bands': band_names,
            'chosen_alphas_per_fold': [
                dict(zip(band_names, [float(x) for x in t]))
                for t in chosen_alphas],
            'median_r': float(np.median(pvr_f)),
            'mean_r': float(np.mean(pvr_f)),
            'n_parcels_scored': int(len(pvr_f)),
            'n_TRs': int(valid.sum()),
        }

    out = {
        'mode': args.mode,
        'bands': band_names,
        'band_widths': band_widths,
        'stimulus_window': args.stimulus_window,
        'hrf_delay': args.hrf_delay,
        'results': results,
    }
    with open(args.out_json, 'w') as f:
        json.dump(out, f, indent=2, default=str)
    log(f'wrote {args.out_json}')


if __name__ == '__main__':
    main()
