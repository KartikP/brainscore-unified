"""Generate Codabench held-out predictions (per subject, per episode) for one
split (friends_s7 | ood) using the validated CLIP+Wav2Vec2+MiniLM banded-ridge
encoder. Writes per-subject episode dicts sub-0X_<split>.npy (keys = the
target_sample_number episode keys Codabench expects).

Per subject:
  1. fit banded ridge on that subject's Friends-train (cached features + BOLD)
  2. predict each held-out episode at its own target_sample_number TR count
Banded alphas from the validated optimum (video=1e5, audio=1e4, text=1e3).
"""
import argparse, glob, sys, time
from pathlib import Path
import numpy as np
sys.path.insert(0, '/home/ubuntu/brain-score-unified/unified')

t0 = time.time()
def log(m): print(f'[{time.time()-t0:7.1f}s] {m}', flush=True)

BR = '/home/ubuntu/.brainio/algonauts2025'
ALPHAS = {'video': 1e5, 'audio': 1e4, 'text': 1e3}
W, D = 5, 3                       # stimulus window, HRF delay
VID_DIM, AUD_DIM, TXT_DIM = 768, 768, 384


def per_tr_load(root, sids, tws, n, dim):
    """Per-TR feature matrix (n, dim): for each TR take clip[t_within_run]."""
    cache, feats = {}, np.zeros((n, dim), np.float32)
    for i in range(n):
        cid = sids[i]
        if cid not in cache:
            p = Path(root) / f'{cid}.npy'
            cache[cid] = np.load(p) if p.exists() else None
        arr = cache[cid]
        if arr is None:
            continue                       # leave zeros (e.g. chaplin text)
        t = int(tws[i])
        feats[i] = arr[t] if t < arr.shape[0] else arr[-1]
    return feats


def stack(Xm, ridx, n):
    """window+HRF stack within run blocks (ridx = run index per obs)."""
    nf = Xm.shape[1]
    Xst = np.zeros((n, W * nf), np.float32)
    for off in range(W):
        shift = D + (W - 1 - off)
        src = np.arange(n) - shift
        ok = src >= 0
        ok[ok] &= (ridx[src[ok]] == ridx[np.arange(n)[ok]])
        Xst[ok, off*nf:(off+1)*nf] = Xm[src[ok]]
    return Xst


def fit_subject(subject, brainscore, clip_model):
    """Fit banded ridge on subject's Friends-train; return W_sol + band offsets."""
    b = brainscore.load_benchmark(f'Algonauts2025-friends-sub{subject:02d}')
    asm = b.assembly
    sids = np.array([str(x) for x in asm['stimulus_id'].values])
    run = np.array([str(x) for x in asm['run'].values])
    tw = np.array(asm['t_within_run'].values, dtype=int)
    n = len(sids)
    log(f'  sub-{subject:02d} train: n_TR={n}')

    # video: re-align per subject (cached frame features → fast)
    fss = b._expand_to_per_TR_frames()
    feats_v, fids = b._extract_per_TR_features(clip_model, fss)
    Xv = np.asarray(b._align_features_to_assembly(feats_v, fids, fss)).astype(np.float32)
    Xa = per_tr_load(f'{BR}/audio_features_wav2vec2', sids, tw, n, AUD_DIM)
    Xt = per_tr_load(f'{BR}/text_features_minilm', sids, tw, n, TXT_DIM)
    parts = [('video', Xv), ('audio', Xa), ('text', Xt)]

    seen, ridx = {}, np.empty(n, np.int64)
    for i, (s, r) in enumerate(zip(sids, run)):
        k = (s, r); ridx[i] = seen.setdefault(k, len(seen))

    Xst_parts, offsets, cur = [], [], 0
    for name, Xm in parts:
        Xst = stack(Xm, ridx, n)
        Xst_parts.append(Xst)
        offsets.append((cur, cur + Xst.shape[1], name)); cur += Xst.shape[1]
    Xs = np.concatenate(Xst_parts, axis=1)
    Y = asm.values.astype(np.float32)

    # drop excluded run edges
    keep = np.ones(n, bool)
    for ri in range(len(seen)):
        idxs = np.where(ridx == ri)[0]
        for ki in range(b._excluded_samples_start):
            if ki < len(idxs): keep[idxs[ki]] = False
        for ki in range(b._excluded_samples_end):
            if ki < len(idxs): keep[idxs[-1-ki]] = False
    Xs, Y = Xs[keep], Y[keep]
    log(f'  sub-{subject:02d} fit: X={Xs.shape}')

    diag = np.zeros(Xs.shape[1], np.float32)
    for lo, hi, name in offsets:
        diag[lo:hi] = ALPHAS[name]
    XtX = Xs.T @ Xs
    XtX[np.diag_indices_from(XtX)] += diag
    Wsol = np.linalg.solve(XtX, Xs.T @ Y).astype(np.float32)
    return Wsol, offsets


def load_heldout_video_per_clip(split):
    """Split the stub-aligned held-out video npz into per-clip arrays keyed by
    the stim_csv stimulus_id (e.g. friends_s07e01a / chaplin1)."""
    import xarray as xr
    X = np.load(f'{BR}/video_features_clip_{split}.npz')['X'].astype(np.float32)
    stub = xr.open_dataarray(f'{BR}/algonauts2025_{split}_sub01.nc')
    sids = np.array([str(x) for x in stub['stimulus_id'].values])
    tw = np.array(stub['t_within_run'].values, dtype=int)
    per_clip = {}
    for cid in np.unique(sids):
        m = sids == cid
        order = np.argsort(tw[m])
        per_clip[cid] = X[np.where(m)[0][order]]
    return per_clip


def predict_subject(subject, split, Wsol, offsets, vid_per_clip):
    """Predict each held-out episode at this subject's target_sample_number TR
    count. Returns {target_key: (n_TRs, 1000) float32}."""
    tsn = glob.glob(f'/home/ubuntu/algonauts_2025/fmri/sub-{subject:02d}/'
                    f'target_sample*number/sub-{subject:02d}_'
                    f'{"friends-s7" if split=="friends_s7" else "ood"}_fmri_samples.npy')
    counts = np.load(tsn[0], allow_pickle=True).item()
    out = {}
    for tkey, L in counts.items():
        clip = f'friends_{tkey}' if split == 'friends_s7' else tkey
        Xv = vid_per_clip.get(clip)
        if Xv is None:
            raise RuntimeError(f'no video for {clip}')
        # build per-TR feature seq length L (pad with last row if short)
        def take(arr, dim):
            if arr is None: return np.zeros((L, dim), np.float32)
            if arr.shape[0] >= L: return arr[:L]
            pad = np.repeat(arr[-1:], L - arr.shape[0], axis=0)
            return np.concatenate([arr, pad], axis=0)
        a = Path(f'{BR}/audio_features_wav2vec2_{split}/{clip}.npy')
        t = Path(f'{BR}/text_features_minilm_{split}/{clip}.npy')
        Xa = np.load(a) if a.exists() else None
        Xt = np.load(t) if t.exists() else None
        parts = [('video', take(Xv, VID_DIM)), ('audio', take(Xa, AUD_DIM)),
                 ('text', take(Xt, TXT_DIM))]
        ridx = np.zeros(L, np.int64)        # single episode block
        Xs = np.concatenate([stack(Xm, ridx, L) for _, Xm in parts], axis=1)
        out[tkey] = (Xs @ Wsol).astype(np.float32)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--split', choices=['friends_s7', 'ood'], required=True)
    ap.add_argument('--subjects', type=int, nargs='+', default=[1, 2, 3, 5])
    ap.add_argument('--out_dir', default='/home/ubuntu/algo_preds')
    args = ap.parse_args()
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    import brainscore
    log('load CLIP...')
    clip = brainscore.load_model('clip-vit-b-32')
    clip._region_layer_map_dict['IT'] = 'post_layernorm'
    log(f'load held-out video per-clip ({args.split})...')
    vid_per_clip = load_heldout_video_per_clip(args.split)
    log(f'  {len(vid_per_clip)} clips')

    for s in args.subjects:
        log(f'=== subject {s} ===')
        Wsol, offsets = fit_subject(s, brainscore, clip)
        preds = predict_subject(s, args.split, Wsol, offsets, vid_per_clip)
        shapes = {k: v.shape for k, v in list(preds.items())[:2]}
        np.save(f'{args.out_dir}/sub-{s:02d}_{args.split}.npy', preds,
                allow_pickle=True)
        log(f'  sub-{s:02d}: {len(preds)} episodes, e.g. {shapes} -> saved')
    log('DONE')


if __name__ == '__main__':
    main()
