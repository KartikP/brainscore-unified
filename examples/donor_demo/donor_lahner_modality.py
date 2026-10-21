"""Donor-demo experiment: which model inputs predict BOLDMoments responses?

Runs ON EC2 ONLY. Stimuli, assembly, checkpoints and audio sidecars are the
registered Brain-Score assets already present on the instance.

Design (all conditions share clips, targets, voxels and outer folds):
  video      V-JEPA v1 ViT-L  backbone.blocks.16, time-mean over 8 tubelets (1024-d)
  audio      Wav2Vec2-base    encoder.layers.6,   time-mean over ~150 frames (768-d)
  both       banded ridge, one alpha per modality (joint fit)
  concat     single-alpha ridge on [video|audio]
  *_pca256   matched feature budget: PCA(256) per modality, fit on the training fold
Hyper-parameters (alpha, PCA) are chosen inside each outer training fold.
Null: clip-label permutation of the training targets (fixed alphas).

BOLDMoments clips were presented SILENTLY (Lahner et al. 2024, Methods), so the
audio tower sees the clip's soundtrack which the participants never heard. The
comparison is about information available to an encoding model, not senses.

Outputs -> OUT/ (see MANIFEST written at the end).
"""
import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from itertools import product
from pathlib import Path

import numpy as np

T0 = time.time()


def log(msg):
    print(f'[{time.time() - T0:7.1f}s] {msg}', flush=True)


def sha1(path, block=1 << 20):
    h = hashlib.sha1()
    with open(path, 'rb') as f:
        while True:
            b = f.read(block)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


# ------------------------------------------------------------------ ridge --
def center(X, Y):
    xm = X.mean(0, keepdims=True)
    ym = Y.mean(0, keepdims=True)
    return X - xm, Y - ym, xm, ym


def ridge_solve(XtX, XtY, lam_vec):
    A = XtX.copy()
    A[np.diag_indices_from(A)] += lam_vec
    return np.linalg.solve(A, XtY)


def per_voxel_r(Y, P):
    Yc = Y - Y.mean(0)
    Pc = P - P.mean(0)
    num = (Yc * Pc).sum(0)
    den = np.sqrt((Yc ** 2).sum(0) * (Pc ** 2).sum(0))
    with np.errstate(divide='ignore', invalid='ignore'):
        r = np.where(den > 0, num / den, np.nan)
    return r


def fit_predict(groups_tr, groups_te, Y_tr, alphas):
    """Banded/single ridge with intercept via centering. alphas: one per group."""
    X_tr = np.concatenate(groups_tr, 1).astype(np.float64)
    X_te = np.concatenate(groups_te, 1).astype(np.float64)
    Xc, Yc, xm, ym = center(X_tr, Y_tr.astype(np.float64))
    lam = np.concatenate([np.full(g.shape[1], a) for g, a in zip(groups_tr, alphas)])
    W = ridge_solve(Xc.T @ Xc, Xc.T @ Yc, lam)
    return (X_te - xm) @ W + ym, W


def select_alphas(groups_tr, Y_tr, roi_sets, grid, n_inner, seed, sel_cols):
    """Inner-CV alpha selection. Returns {roi_name: alpha_tuple} and the score table.

    sel_cols: voxel columns used for selection scoring (ROI voxels + a random
    whole-cortex subsample), so the solve has few right-hand sides.
    """
    n = Y_tr.shape[0]
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    folds = np.array_split(perm, n_inner)
    combos = list(product(grid, repeat=len(groups_tr)))
    Ysel = Y_tr[:, sel_cols].astype(np.float64)
    # accumulate inner-val predictions per combo
    preds = {c: np.zeros_like(Ysel) for c in combos}
    for k in range(n_inner):
        va = folds[k]
        tr = np.concatenate([folds[j] for j in range(n_inner) if j != k])
        X_tr = np.concatenate([g[tr] for g in groups_tr], 1).astype(np.float64)
        X_va = np.concatenate([g[va] for g in groups_tr], 1).astype(np.float64)
        Xc, Yc, xm, ym = center(X_tr, Ysel[tr])
        XtX, XtY = Xc.T @ Xc, Xc.T @ Yc
        sizes = [g.shape[1] for g in groups_tr]
        for c in combos:
            lam = np.concatenate([np.full(s, a) for s, a in zip(sizes, c)])
            W = ridge_solve(XtX, XtY, lam)
            preds[c][va] = (X_va - xm) @ W + ym
    table = {}
    for c in combos:
        r = per_voxel_r(Ysel, preds[c])
        table[c] = r
    chosen = {}
    for roi, cols in roi_sets.items():
        # cols are positions within sel_cols
        best, best_c = -np.inf, None
        for c in combos:
            s = np.nanmean(table[c][cols])
            if s > best:
                best, best_c = s, c
        chosen[roi] = best_c
    return chosen, {str(c): {roi: float(np.nanmean(table[c][cols])) for roi, cols in roi_sets.items()}
                    for c in combos}


# ------------------------------------------------------------------- main --
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='/opt/dlami/nvme/donor_demo/out')
    ap.add_argument('--n_outer', type=int, default=5)
    ap.add_argument('--n_inner', type=int, default=4)
    ap.add_argument('--n_perm', type=int, default=100)
    ap.add_argument('--n_boot', type=int, default=1000)
    ap.add_argument('--n_demo', type=int, default=12)
    ap.add_argument('--skip_extract', action='store_true')
    ap.add_argument('--stage', default='all', choices=['all', 'extract', 'score'])
    args = ap.parse_args()
    OUT = Path(args.out)
    (OUT / 'features').mkdir(parents=True, exist_ok=True)
    (OUT / 'neural').mkdir(exist_ok=True)
    (OUT / 'predictions').mkdir(exist_ok=True)
    (OUT / 'metrics').mkdir(exist_ok=True)
    (OUT / 'demo').mkdir(exist_ok=True)
    GRID = (1e0, 1e1, 1e2, 1e3, 1e4, 1e5)

    import brainscore
    from brainscore.benchmarks.lahner2024.benchmark_multimodal import (
        Lahner2024BOLDMoments_multimodal)
    from brainscore.benchmarks.lahner2024.auditory_roi import build_auditory_mask

    bench = Lahner2024BOLDMoments_multimodal(reliability_threshold=0.3)
    video_stim = bench._video_stim_set()
    audio_stim = bench._audio_stim_set()
    clip_ids = list(video_stim['stimulus_id'])
    n_clips = len(clip_ids)
    log(f'{n_clips} clips')

    # ---------------------------------------------------------- extraction
    feat_npz = OUT / 'features' / 'pooled_features.npz'
    v_nc = OUT / 'features' / 'video_vjepa1_blocks16_time_resolved.nc'
    a_nc = OUT / 'features' / 'audio_wav2vec2_layer6_time_resolved.nc'
    if not args.skip_extract and args.stage in ('all', 'extract') and not feat_npz.exists():
        import xarray as xr
        if v_nc.exists() and a_nc.exists():
            log('reusing saved time-resolved assemblies from this run directory')
            va = xr.open_dataarray(v_nc).load()
            aa = xr.open_dataarray(a_nc).load()
            model = None
        else:
            model = brainscore.load_model('vjepa1-wav2vec2')
            log('V-JEPA v1 video extraction via BrainScoreModel.process ...')
            model.start_recording('IT', time_bins=[(0, 3000)])
            va = model.process(video_stim)
            log(f'  video assembly dims={va.dims} shape={va.shape}')
            log('Wav2Vec2 audio extraction via BrainScoreModel.process ...')
            model.start_recording('A1', time_bins=[(0, 3000)])
            aa = model.process(audio_stim)
            log(f'  audio assembly dims={aa.dims} shape={aa.shape}')

        def ids_of(asm):
            return [str(x) for x in np.asarray(asm['stimulus_id'].values)]

        def save_asm(asm, name):
            a = asm.copy()
            for d in ('presentation', 'neuroid', 'time_bin'):
                if d in a.indexes and hasattr(a.indexes[d], 'levels'):
                    a = a.reset_index(d)
            a.to_netcdf(OUT / 'features' / name)

        if model is not None:
            save_asm(va, 'video_vjepa1_blocks16_time_resolved.nc')
            save_asm(aa, 'audio_wav2vec2_layer6_time_resolved.nc')
        v_ids, a_ids = ids_of(va), ids_of(aa)
        vX = va.mean('time_bin').values if 'time_bin' in va.dims else va.values
        aX = aa.mean('time_bin').values if 'time_bin' in aa.dims else aa.values
        vX = vX[[v_ids.index(c) for c in clip_ids]]
        aX = aX[[a_ids.index(c) for c in clip_ids]]
        np.savez(feat_npz, clip_ids=np.array(clip_ids), video=vX.astype(np.float32),
                 audio=aX.astype(np.float32),
                 video_time_bin_start_ms=np.asarray(va['time_bin_start_ms'].values if 'time_bin_start_ms' in va.coords else []),
                 audio_time_bin_start_ms=np.asarray(aa['time_bin_start_ms'].values if 'time_bin_start_ms' in aa.coords else []))
        log(f'pooled features saved video={vX.shape} audio={aX.shape}')
        del model
        if args.stage == 'extract':
            return
    f = np.load(feat_npz, allow_pickle=True)
    assert list(f['clip_ids']) == clip_ids
    Xv, Xa = f['video'], f['audio']

    # -------------------------------------------------------------- neural
    asm = bench.assembly.squeeze('time_bin', drop=True)  # (neuroid, presentation)
    sid = np.asarray(asm['stimulus_id'].values).astype(str)
    rep = np.asarray(asm['repetition'].values).astype(int)
    units = sorted(set(rep.tolist()))
    data = asm.values  # (n_vox, n_pres)
    n_vox = data.shape[0]
    Y_unit = np.zeros((len(units), n_clips, n_vox), np.float32)
    pos = {c: i for i, c in enumerate(clip_ids)}
    for j in range(data.shape[1]):
        Y_unit[units.index(rep[j]), pos[sid[j]]] = data[:, j]
    Y_mean = Y_unit.mean(0)
    log(f'neural: units={len(units)} Y_mean={Y_mean.shape}')
    rel = np.asarray(bench._split_half_reliability(n_splits=20, random_state=0))
    ceiling = np.sqrt(np.clip(rel, 0, 1))
    vis_mask = rel >= 0.3
    aud_mask = build_auditory_mask()
    hemi = np.asarray(asm['hemi'].values).astype(str)
    np.savez_compressed(OUT / 'neural' / 'targets.npz', clip_ids=np.array(clip_ids),
                        Y_mean=Y_mean, reliability=rel.astype(np.float32),
                        ceiling_sqrt_reliability=ceiling.astype(np.float32),
                        visual_roi_mask=vis_mask, auditory_roi_mask=aud_mask,
                        hemi=hemi, neuroid_id=np.asarray(asm['neuroid_id'].values),
                        unit_ids=np.array(units))
    roi_cols = np.flatnonzero(vis_mask | aud_mask)
    np.savez_compressed(OUT / 'neural' / 'targets_per_unit_roi.npz', clip_ids=np.array(clip_ids),
                        roi_cols=roi_cols, Y_unit_roi=Y_unit[:, :, roi_cols].astype(np.float16),
                        unit_ids=np.array(units))
    silent = {}
    man = json.load(open(os.path.expanduser('~/.brainio/lahner2024_audio_16k/manifest.json')))
    for e in man['entries']:
        silent[e['stimulus_id']] = (e['action'] == 'silent_placeholder')
    silent_flag = np.array([silent.get(c, False) for c in clip_ids])
    log(f'ROI sizes: visual={vis_mask.sum()} auditory={aud_mask.sum()} silent_clips={silent_flag.sum()}')

    # ------------------------------------------------------------- folds
    rng = np.random.default_rng(0)
    perm = rng.permutation(n_clips)
    outer = np.array_split(perm, args.n_outer)
    fold_of = np.zeros(n_clips, int)
    for k, te in enumerate(outer):
        fold_of[te] = k
    np.save(OUT / 'metrics' / 'outer_fold_of_clip.npy', fold_of)

    # selection columns: all ROI voxels + 3000 random whole-cortex voxels
    sel_extra = rng.choice(np.flatnonzero(~(vis_mask | aud_mask)), 3000, replace=False)
    sel_cols = np.concatenate([roi_cols, sel_extra])
    sel_pos = {c: i for i, c in enumerate(sel_cols)}
    roi_sets = {
        'visual': np.array([sel_pos[c] for c in np.flatnonzero(vis_mask)]),
        'auditory': np.array([sel_pos[c] for c in np.flatnonzero(aud_mask)]),
        'whole': np.arange(len(sel_cols)),
    }
    rois = {'visual': np.flatnonzero(vis_mask), 'auditory': np.flatnonzero(aud_mask),
            'whole': np.arange(n_vox)}

    from sklearn.decomposition import PCA

    def make_groups(cond, tr, te):
        """Return (groups_tr, groups_te, extra_info)."""
        if cond == 'video':
            return [Xv[tr]], [Xv[te]], {}
        if cond == 'audio':
            return [Xa[tr]], [Xa[te]], {}
        if cond in ('both', 'concat'):
            return [Xv[tr], Xa[tr]], [Xv[te], Xa[te]], {}
        if cond.endswith('_pca256'):
            base = cond.replace('_pca256', '')
            outs_tr, outs_te, info = [], [], {}
            for name, X in (('video', Xv), ('audio', Xa)):
                if base in (name, 'both'):
                    p = PCA(256, random_state=0).fit(X[tr])
                    outs_tr.append(p.transform(X[tr]))
                    outs_te.append(p.transform(X[te]))
                    info[f'{name}_pca_var_explained'] = float(p.explained_variance_ratio_.sum())
            return outs_tr, outs_te, info
        raise ValueError(cond)

    CONDS = ['video', 'audio', 'both', 'concat', 'video_pca256', 'audio_pca256', 'both_pca256']

    def run_target(Y, tag, conds, keep_full_pred, n_inner):
        """Full nested-CV run for one target matrix. Returns dict of results."""
        res = {}
        for cond in conds:
            pred = {roi: np.full((n_clips, len(cols)), np.nan, np.float32)
                    for roi, cols in rois.items()} if cond != 'concat' else None
            alpha_log = []
            info_log = []
            for k, te in enumerate(outer):
                tr = np.concatenate([outer[j] for j in range(args.n_outer) if j != k])
                g_tr, g_te, info = make_groups(cond, tr, te)
                if cond == 'concat':
                    g_tr, g_te = [np.concatenate(g_tr, 1)], [np.concatenate(g_te, 1)]
                chosen, table = select_alphas(g_tr, Y[tr], roi_sets, GRID, n_inner,
                                              seed=1000 + k, sel_cols=sel_cols)
                alpha_log.append({roi: list(map(float, a)) for roi, a in chosen.items()})
                info_log.append(info)
                if cond == 'concat':
                    pred = pred or {roi: np.full((n_clips, len(cols)), np.nan, np.float32)
                                    for roi, cols in rois.items()}
                # fit once per distinct alpha tuple, slice ROI predictions
                done = {}
                for roi, a in chosen.items():
                    if a not in done:
                        P, _ = fit_predict(g_tr, g_te, Y[tr], a)
                        done[a] = P
                    pred[roi][te] = done[a][:, rois[roi]]
            out = {'alphas_per_fold': alpha_log, 'fold_info': info_log, 'roi': {}}
            for roi, cols in rois.items():
                r = per_voxel_r(Y[:, cols], pred[roi])
                nr = r / ceiling[cols]
                out['roi'][roi] = {
                    'median_r': float(np.nanmedian(r)), 'mean_r': float(np.nanmean(r)),
                    'median_r_ceiling_normalized': float(np.nanmedian(nr)),
                    'n_voxels': int(len(cols)),
                    'median_r_excluding_silent_clips': float(np.nanmedian(
                        per_voxel_r(Y[~silent_flag][:, cols], pred[roi][~silent_flag]))),
                }
                np.save(OUT / 'metrics' / f'per_voxel_r__{tag}__{cond}__{roi}.npy', r.astype(np.float32))
                # per-clip spatial pattern correlation (predicted vs measured across ROI voxels)
                if roi != 'whole':
                    Pc = pred[roi] - pred[roi].mean(1, keepdims=True)
                    Yc = Y[:, cols] - Y[:, cols].mean(1, keepdims=True)
                    sp = (Pc * Yc).sum(1) / np.sqrt((Pc ** 2).sum(1) * (Yc ** 2).sum(1))
                    np.save(OUT / 'metrics' / f'per_clip_spatial_r__{tag}__{cond}__{roi}.npy', sp.astype(np.float32))
            if keep_full_pred:
                np.save(OUT / 'predictions' / f'heldout_pred__{tag}__{cond}__whole.npy', pred['whole'].astype(np.float16))
            else:
                for roi in ('visual', 'auditory'):
                    np.save(OUT / 'predictions' / f'heldout_pred__{tag}__{cond}__{roi}.npy', pred[roi].astype(np.float16))
            res[cond] = out
            log(f'  [{tag}] {cond}: ' + ', '.join(f"{roi} r={o['median_r']:.4f}" for roi, o in out['roi'].items()))
        return res

    results = {'target_subject_mean': run_target(Y_mean, 'mean', CONDS, True, args.n_inner)}
    json.dump(results, open(OUT / 'metrics' / 'results_main.json', 'w'), indent=1)

    # ---------------------------------------------------- per-unit targets
    per_unit = {}
    for ui, u in enumerate(units):
        per_unit[str(u)] = run_target(Y_unit[ui], f'unit{u}', ['video', 'audio', 'both'], False, 3)
    results['target_per_unit'] = per_unit
    json.dump(results, open(OUT / 'metrics' / 'results_main.json', 'w'), indent=1)

    # ------------------------------------------------ paired stats (units)
    from scipy import stats
    paired = {}
    for roi in rois:
        tab = {c: np.array([per_unit[str(u)][c]['roi'][roi]['median_r'] for u in units])
               for c in ('video', 'audio', 'both')}
        paired[roi] = {'per_unit_median_r': {c: v.tolist() for c, v in tab.items()}}
        for a, b in (('both', 'video'), ('video', 'audio'), ('both', 'audio')):
            d = tab[a] - tab[b]
            boots = [np.mean(rng.choice(d, len(d))) for _ in range(5000)]
            try:
                w = stats.wilcoxon(d).pvalue
            except ValueError:
                w = float('nan')
            paired[roi][f'{a}_minus_{b}'] = {
                'mean_diff': float(d.mean()), 'ci95': [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
                'wilcoxon_p': float(w), 'n_units': len(d), 'n_units_positive': int((d > 0).sum())}
    results['paired_across_units'] = paired

    # ------------------------------------------- clip bootstrap (mean target)
    boot = {}
    preds_mean = {c: np.load(OUT / 'predictions' / f'heldout_pred__mean__{c}__whole.npy').astype(np.float32)
                  for c in ('video', 'audio', 'both', 'concat')}
    for roi in ('visual', 'auditory'):
        cols = rois[roi]
        Yr = Y_mean[:, cols]
        med = {c: [] for c in preds_mean}
        for b in range(args.n_boot):
            idx = rng.choice(n_clips, n_clips)
            for c, P in preds_mean.items():
                med[c].append(np.nanmedian(per_voxel_r(Yr[idx], P[idx][:, cols])))
        med = {c: np.array(v) for c, v in med.items()}
        boot[roi] = {c: {'median_r_ci95': [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]} for c, v in med.items()}
        for a, bb in (('both', 'video'), ('video', 'audio'), ('concat', 'video')):
            d = med[a] - med[bb]
            boot[roi][f'{a}_minus_{bb}'] = {'mean': float(d.mean()), 'ci95': [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))],
                                            'frac_positive': float((d > 0).mean())}
    results['clip_bootstrap_subject_mean'] = boot

    # ------------------------------------------------------- permutation null
    null = {}
    main_alphas = {c: results['target_subject_mean'][c]['alphas_per_fold'] for c in ('video', 'audio', 'both')}
    for roi in ('visual', 'auditory'):
        cols = rois[roi]
        null[roi] = {}
        for cond in ('video', 'audio', 'both'):
            meds = []
            for p in range(args.n_perm):
                prm = np.random.default_rng(5000 + p).permutation(n_clips)
                P = np.zeros((n_clips, len(cols)), np.float32)
                for k, te in enumerate(outer):
                    tr = np.concatenate([outer[j] for j in range(args.n_outer) if j != k])
                    g_tr, g_te, _ = make_groups(cond, tr, te)
                    a = tuple(main_alphas[cond][k][roi])
                    Ytr = Y_mean[prm[tr]][:, cols]  # shuffled clip labels (train only)
                    P[te], _ = fit_predict(g_tr, g_te, Ytr, a)
                meds.append(float(np.nanmedian(per_voxel_r(Y_mean[:, cols], P))))
            obs = results['target_subject_mean'][cond]['roi'][roi]['median_r']
            null[roi][cond] = {'null_median_r': meds, 'null_mean': float(np.mean(meds)), 'null_p975': float(np.percentile(meds, 97.5)),
                               'observed': obs, 'p_value': float((np.sum(np.array(meds) >= obs) + 1) / (len(meds) + 1))}
            log(f'  null [{roi}] {cond}: null mean={np.mean(meds):.4f} obs={obs:.4f}')
    results['permutation_null'] = null
    json.dump(results, open(OUT / 'metrics' / 'results_main.json', 'w'), indent=1)

    # ------------------------------------------------------- demo clips
    sp = np.load(OUT / 'metrics' / 'per_clip_spatial_r__mean__video__visual.npy')
    order = np.argsort(-sp)
    cand = [i for i in order if not silent_flag[i]]
    top = cand[:args.n_demo]
    median_i = cand[len(cand) // 2]
    worst = cand[-3:]
    demo = {'rule': 'top-N held-out clips by per-clip spatial correlation (V-JEPA video condition, visual ROI, subject-mean target), excluding clips with silent-placeholder audio; plus the median clip and the 3 worst for honesty',
            'top': [clip_ids[i] for i in top], 'median': clip_ids[median_i], 'worst': [clip_ids[i] for i in worst],
            'spatial_r': {clip_ids[i]: float(sp[i]) for i in list(top) + [median_i] + list(worst)}}
    json.dump(demo, open(OUT / 'demo' / 'selection.json', 'w'), indent=1)
    log(f'demo clips: {demo["top"][:5]} ...')

    # token-level activations for demo clips (direct hooks on the same backbone)
    import torch
    from brainscore.models.vjepa_v1.model import (
        _default_cache_dir, _download_checkpoint, _load_vjepa_v1_vitl, _make_preprocessing,
        CHECKPOINT_URL, NUM_INPUT_FRAMES)
    from brainscore.model_helpers.video_wrapper import default_uniform_frame_sampler
    import cv2
    ckpt = _download_checkpoint(CHECKPOINT_URL, _default_cache_dir() / 'vitl16.pth.tar')
    backbone = _load_vjepa_v1_vitl(ckpt).eval().cuda()
    pre = _make_preprocessing()
    layers = [4, 9, 15, 16, 23]
    store = {}
    def _hook(l):
        def fn(m, i, o):
            o = o[0] if isinstance(o, (tuple, list)) else o
            store[l] = o.detach().float().cpu().numpy()
        return fn
    hooks = [backbone.blocks[l].register_forward_hook(_hook(l)) for l in layers]
    demo_ids = demo['top'] + [demo['median']] + demo['worst']
    frame_meta = {}
    for cid in demo_ids:
        path = video_stim.get_stimulus(cid)
        cap = cv2.VideoCapture(str(path)); total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); fps = cap.get(cv2.CAP_PROP_FPS) or 30.0; cap.release()
        frames = default_uniform_frame_sampler(path, num_frames=NUM_INPUT_FRAMES)
        idxs = list(range(total)) if total <= NUM_INPUT_FRAMES else [int(round(i * (total - 1) / (NUM_INPUT_FRAMES - 1))) for i in range(NUM_INPUT_FRAMES)]
        d = OUT / 'demo' / cid
        d.mkdir(exist_ok=True)
        for i, fr in enumerate(frames):
            cv2.imwrite(str(d / f'frame_{i:02d}.jpg'), cv2.cvtColor(fr, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 92])
        x = pre(frames).unsqueeze(0).permute(0, 2, 1, 3, 4).cuda()  # (1,C,T,H,W)
        store.clear()
        with torch.no_grad():
            backbone(x)
        for l in layers:
            arr = store[l][0]  # (T*S, H)
            np.save(d / f'tokens_blocks{l}.npy', arr.reshape(8, 196, -1).astype(np.float16))
        frame_meta[cid] = {'video_path': str(path), 'sha1': sha1(path), 'fps': fps, 'n_frames_total': total,
                           'duration_s': total / fps, 'sampled_frame_indices': idxs[:len(frames)],
                           'sampled_frame_times_s': [i / fps for i in idxs[:len(frames)]],
                           'tubelet_time_windows_s': [[idxs[2 * t] / fps, idxs[min(2 * t + 1, len(idxs) - 1)] / fps] for t in range(8)],
                           'audio_silent_placeholder': bool(silent.get(cid, False)),
                           'token_layout': '(8 tubelets, 196 spatial patches 14x14 row-major, 1024 hidden)'}
    for h in hooks:
        h.remove()
    json.dump(frame_meta, open(OUT / 'demo' / 'frame_meta.json', 'w'), indent=1)

    # ------------------------------------------------------------ manifest
    def git(repo):
        d = os.path.expanduser(f'~/brain-score-unified/{repo}')
        return {'commit': subprocess.run(['git', '-C', d, 'rev-parse', 'HEAD'], capture_output=True, text=True).stdout.strip(),
                'branch': subprocess.run(['git', '-C', d, 'rev-parse', '--abbrev-ref', 'HEAD'], capture_output=True, text=True).stdout.strip()}
    import torch, transformers, sklearn, xarray, scipy
    manifest = {
        'script': Path(__file__).name, 'script_sha1': sha1(__file__), 'finished_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        'host': platform.node(), 'instance': 'i-0bdbdf83c4db9bdae (g6.4xlarge, NVIDIA L4)',
        'python': sys.version, 'torch': torch.__version__, 'cuda': torch.version.cuda, 'transformers': transformers.__version__,
        'sklearn': sklearn.__version__, 'numpy': np.__version__, 'xarray': xarray.__version__, 'scipy': scipy.__version__,
        'repos': {r: git(r) for r in ('core', 'vision', 'language', 'unified')},
        'model': {'identifier': 'vjepa1-wav2vec2', 'video': 'V-JEPA v1 ViT-L/16 vitl16.pth.tar (sha1 below), backbone.blocks.16, 16 frames uniformly sampled, 224x224, ImageNet norm, fp32',
                  'video_ckpt_sha1': sha1(_default_cache_dir() / 'vitl16.pth.tar'),
                  'audio': 'facebook/wav2vec2-base encoder.layers.6, 16 kHz mono WAV demuxed from the clip MP4, fp32'},
        'data': {'assembly': 'assy_Lahner2024-fMRI.nc sha1=2c7f1d2e5724b8cc3c5cf47986e956c4f13001e4 (presentation=1026 clips x 10 units; neuroid=20484 fsaverage5 vertices L||R)',
                 'stimuli': 'BOLDMoments stimulus set zip sha1=dc9c3bf631632cd433d02f2f1847fd33c01ae0b3 (1026 mp4)',
                 'audio_sidecar_manifest': man['counts']},
        'protocol': {'outer_folds': args.n_outer, 'inner_folds': args.n_inner, 'alpha_grid': GRID, 'n_perm': args.n_perm, 'n_boot': args.n_boot,
                     'time_pooling': 'mean over model time bins within the 3 s clip (one BOLD estimate per clip; no HRF step applies to GLM betas)',
                     'targets': 'subject_mean = mean over the 10 units of the presentation axis; per_unit = each unit separately'},
        'result_caching_home': os.environ.get('RESULTCACHING_HOME'),
    }
    json.dump(manifest, open(OUT / 'MANIFEST.json', 'w'), indent=1)
    log('DONE')


if __name__ == '__main__':
    main()
