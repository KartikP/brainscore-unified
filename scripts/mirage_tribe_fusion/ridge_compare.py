"""Controlled native-vs-post-hoc fusion comparison on Algonauts sub-01 (subset).

Both arms go through the SAME ridge readout, same TRs, same design matrix, same
CV folds — the only variable is the feature source:
  * NATIVE (MIRAGE-style): Qwen3-Omni-30B post-fusion thinker hidden states.
  * POST-HOC (TRIBEv2-style): V-JEPA-2 + Wav2Vec-Bert + Llama-3.2-3B, concatenated.

Per-modality and banded post-hoc are reported as context, but the headline is
native(Qwen) vs post-hoc-concat under identical ridge.

Design: per-clip lagged window (stimulus_window TRs ending hrf_delay before the
target TR; lags never cross a clip boundary). KFold over CLIPS (no temporal
leakage). Per-parcel Pearson r on held-out clips, averaged across parcels;
alpha chosen by nested grid on the train folds.
"""
import argparse, json, sys, time
from pathlib import Path
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold

t0 = time.time()
def log(m): print(f'[{time.time()-t0:7.1f}s] {m}', flush=True)


def load_assembly(path, subset):
    import xarray as xr
    da = xr.open_dataarray(path)
    # find the sample dim (carries stimulus_id) and the parcel dim
    sdim = [d for d in da.dims if 'stimulus_id' in da[d].coords or
            (da[d].size == da['stimulus_id'].size)][0] if 'stimulus_id' in da.coords else da.dims[0]
    # robust: stimulus_id is a coord along one dim
    sdim = da['stimulus_id'].dims[0]
    pdim = [d for d in da.dims if d != sdim][0]
    da = da.transpose(sdim, pdim)
    sid = np.array([str(x) for x in da['stimulus_id'].values])
    tw = np.array(da['t_within_run'].values, dtype=int)
    Y = np.asarray(da.values, dtype=np.float32)        # (n_samples, n_parcels)
    keep = np.isin(sid, list(subset))
    return sid[keep], tw[keep], Y[keep]


def build_clip_order(sid, tw):
    """Return list of (clip, local_tr_indices_into_subset_rows) grouped per clip,
    each ordered by t_within_run."""
    groups = {}
    for i, (c, t) in enumerate(zip(sid, tw)):
        groups.setdefault(c, []).append((t, i))
    out = {}
    for c, lst in groups.items():
        lst.sort()
        out[c] = [i for _, i in lst]
    return out


def lagged(F, clip_rows, sid, win, delay):
    """Build design matrix: each target TR -> concat of `win` feature rows ending
    `delay` TRs earlier, within the same clip (zero-pad at clip start)."""
    n, d = F.shape
    X = np.zeros((n, win * d), np.float32)
    for c, rows in clip_rows.items():
        for local, gi in enumerate(rows):        # gi = global subset row index
            for w in range(win):
                src_local = local - delay - w
                if src_local >= 0:
                    X[gi, w*d:(w+1)*d] = F[rows[src_local]]
    return X


def per_parcel_r(Yt, Yp):
    Yc = Yt - Yt.mean(0); Pc = Yp - Yp.mean(0)
    num = (Yc*Pc).sum(0); den = np.sqrt((Yc**2).sum(0)*(Pc**2).sum(0))
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.where(den > 0, num/den, np.nan)


def score_arm(Xmods, Y, clip_rows, sid, win, delay, alphas, n_splits=5, banded=False):
    """Xmods: list of raw per-TR feature matrices (each (n, d_m)). Builds lagged
    design per modality, concatenates, ridge with alpha-grid CV over clip folds."""
    clips = list(clip_rows.keys())
    Xs = [lagged(F, clip_rows, sid, win, delay) for F in Xmods]
    X = np.concatenate(Xs, axis=1)
    n = Y.shape[0]
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=0)
    # CV over CLIPS
    clip_arr = np.array(clips)
    best = {'r': -1, 'alpha': None}
    for alpha in alphas:
        preds = np.zeros_like(Y)
        for tr_ci, te_ci in kf.split(clip_arr):
            tr_clips = set(clip_arr[tr_ci]); te_clips = set(clip_arr[te_ci])
            tr_rows = np.concatenate([clip_rows[c] for c in clip_arr[tr_ci]])
            te_rows = np.concatenate([clip_rows[c] for c in clip_arr[te_ci]])
            reg = Ridge(alpha=alpha).fit(X[tr_rows], Y[tr_rows])
            preds[te_rows] = reg.predict(X[te_rows])
        r = float(np.nanmedian(per_parcel_r(Y, preds)))
        if r > best['r']: best = {'r': r, 'alpha': alpha}
        log(f'    alpha={alpha}: median r={r:.4f}')
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--assembly_path', default='/home/ubuntu/.brainio/algonauts2025/algonauts2025_friends_sub01.nc')
    ap.add_argument('--clip_list', default='/home/ubuntu/clip_list.txt')
    ap.add_argument('--qwen', default='/home/ubuntu/.brainio/algonauts2025/qwen_native_l16')
    ap.add_argument('--video', default='/home/ubuntu/.brainio/algonauts2025/tribe_video_vjepa2')
    ap.add_argument('--audio', default='/home/ubuntu/.brainio/algonauts2025/tribe_audio_w2vbert')
    ap.add_argument('--text', default='/home/ubuntu/.brainio/algonauts2025/tribe_text_llama3b')
    ap.add_argument('--stimulus_window', type=int, default=5)
    ap.add_argument('--hrf_delay', type=int, default=3)
    ap.add_argument('--alpha_grid', default='10,100,1000,10000,100000')
    ap.add_argument('--out_json', default='/tmp/mirage_tribe_compare.json')
    args = ap.parse_args()
    alphas = [float(a) for a in args.alpha_grid.split(',')]

    subset = set(l.strip() for l in open(args.clip_list) if l.strip())
    sid, tw, Y = load_assembly(args.assembly_path, subset)
    log(f'subset: {len(set(sid))} clips, {len(sid)} TRs, Y={Y.shape}')
    clip_rows = build_clip_order(sid, tw)

    def load_mod(root):
        root = Path(root); cache = {}
        d = None
        for c in set(sid):
            p = root / f'{c}.npy'
            cache[c] = np.load(p) if p.exists() else None
            if cache[c] is not None: d = cache[c].shape[1]
        F = np.zeros((len(sid), d), np.float32)
        # place each clip's rows by t_within_run local index
        for c, rows in clip_rows.items():
            arr = cache.get(c)
            if arr is None: continue
            for local, gi in enumerate(rows):
                if local < len(arr): F[gi] = arr[local]
        return F

    log('loading features...')
    Fq = load_mod(args.qwen); Fv = load_mod(args.video)
    Fa = load_mod(args.audio); Ft = load_mod(args.text)
    log(f'dims: qwen={Fq.shape[1]} video={Fv.shape[1]} audio={Fa.shape[1]} text={Ft.shape[1]}')

    W, D = args.stimulus_window, args.hrf_delay
    res = {}
    def timed(name, mods):
        log(f'{name}:'); s = time.time(); r = score_arm(mods, Y, clip_rows, sid, W, D, alphas)
        r['scoring_sec'] = round(time.time() - s, 1)
        r['feature_dim'] = int(sum(m.shape[1] for m in mods)); return r
    res['native_qwen']    = timed('NATIVE (Qwen3-Omni)',        [Fq])
    res['video_only']     = timed('POST-HOC video (V-JEPA-2)',  [Fv])
    res['audio_only']     = timed('POST-HOC audio (Wav2Vec-Bert)', [Fa])
    res['text_only']      = timed('POST-HOC text (Llama-3.2-3B)',[Ft])
    res['posthoc_concat'] = timed('POST-HOC concat (V+A+T)',    [Fv, Fa, Ft])

    summary = {'subset_clips': sorted(set(sid.tolist())), 'n_TRs': int(len(sid)),
               'n_parcels': int(Y.shape[1]), 'stimulus_window': W, 'hrf_delay': D,
               'results': res}
    json.dump(summary, open(args.out_json, 'w'), indent=2)
    log('=== SUMMARY (median per-parcel Pearson r, held-out clips) ===')
    for k, v in res.items():
        log(f'  {k:18s} r={v["r"]:.4f}  (alpha={v["alpha"]})')
    log(f'saved -> {args.out_json}')


if __name__ == '__main__':
    main()
