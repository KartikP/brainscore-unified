"""Within-Qwen fusion curve: score each of Qwen3-Omni's 49 thinker hidden-state
layers through the SAME ridge readout (same dim, same pooling, same CV) on
Algonauts sub-01. Layer 0 = modality tower streams (fusion OFF); later layers =
progressively cross-modally fused (fusion ON). The rise from layer 0 to the peak
isolates the contribution of native fusion within ONE backbone — no
backbone-identity confound.

Reuses the lagged-design ridge from ridge_compare. Input: qwen_alllayers/*.npy
of shape (n_TRs, 49, 2048). Output: /tmp/qwen_fusion_curve.json
"""
import argparse, json, time
from pathlib import Path
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold

t0 = time.time()
def log(m): print(f'[{time.time()-t0:7.1f}s] {m}', flush=True)


def load_assembly(path, subset):
    import xarray as xr
    da = xr.open_dataarray(path)
    sdim = da['stimulus_id'].dims[0]; pdim = [d for d in da.dims if d != sdim][0]
    da = da.transpose(sdim, pdim)
    sid = np.array([str(x) for x in da['stimulus_id'].values])
    tw = np.array(da['t_within_run'].values, dtype=int)
    Y = np.asarray(da.values, dtype=np.float32)
    keep = np.isin(sid, list(subset))
    return sid[keep], tw[keep], Y[keep]


def build_clip_order(sid, tw):
    groups = {}
    for i, (c, t) in enumerate(zip(sid, tw)): groups.setdefault(c, []).append((t, i))
    out = {}
    for c, lst in groups.items(): lst.sort(); out[c] = [i for _, i in lst]
    return out


def lagged(F, clip_rows, win, delay):
    n, d = F.shape; X = np.zeros((n, win*d), np.float32)
    for c, rows in clip_rows.items():
        for local, gi in enumerate(rows):
            for w in range(win):
                s = local - delay - w
                if s >= 0: X[gi, w*d:(w+1)*d] = F[rows[s]]
    return X


def per_parcel_r(Yt, Yp):
    Yc = Yt - Yt.mean(0); Pc = Yp - Yp.mean(0)
    num = (Yc*Pc).sum(0); den = np.sqrt((Yc**2).sum(0)*(Pc**2).sum(0))
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.where(den > 0, num/den, np.nan)


def score(F, Y, clip_rows, win, delay, alphas, n_splits=5):
    X = lagged(F, clip_rows, win, delay)
    clips = np.array(list(clip_rows.keys()))
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=0)
    best = -1
    for a in alphas:
        preds = np.zeros_like(Y)
        for tr_ci, te_ci in kf.split(clips):
            tr = np.concatenate([clip_rows[c] for c in clips[tr_ci]])
            te = np.concatenate([clip_rows[c] for c in clips[te_ci]])
            preds[te] = Ridge(alpha=a).fit(X[tr], Y[tr]).predict(X[te])
        r = float(np.nanmedian(per_parcel_r(Y, preds)))
        best = max(best, r)
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--assembly_path', default='/home/ubuntu/.brainio/algonauts2025/algonauts2025_friends_sub01.nc')
    ap.add_argument('--clip_list', default='/home/ubuntu/clip_list.txt')
    ap.add_argument('--alllayers', default='/home/ubuntu/.brainio/algonauts2025/qwen_alllayers')
    ap.add_argument('--layers', default='', help='comma list; default every 2nd layer + 0,24,48')
    ap.add_argument('--stimulus_window', type=int, default=5)
    ap.add_argument('--hrf_delay', type=int, default=3)
    ap.add_argument('--alpha_grid', default='100,1000,10000')
    ap.add_argument('--out_json', default='/tmp/qwen_fusion_curve.json')
    args = ap.parse_args()
    alphas = [float(a) for a in args.alpha_grid.split(',')]

    subset = set(l.strip() for l in open(args.clip_list) if l.strip())
    sid, tw, Y = load_assembly(args.assembly_path, subset)
    clip_rows = build_clip_order(sid, tw)
    log(f'subset: {len(set(sid))} clips, {len(sid)} TRs, Y={Y.shape}')

    # load all-layers features (n_TRs, 49, H) per clip -> aligned (n_sid, 49, H)
    root = Path(args.alllayers); cache = {}
    nL = H = None
    for c in set(sid):
        p = root / f'{c}.npy'
        if p.exists():
            cache[c] = np.load(p); nL, H = cache[c].shape[1], cache[c].shape[2]
        else: cache[c] = None
    F_all = np.zeros((len(sid), nL, H), np.float32)
    for c, rows in clip_rows.items():
        arr = cache.get(c)
        if arr is None: continue
        for local, gi in enumerate(rows):
            if local < len(arr): F_all[gi] = arr[local].astype(np.float32)
    log(f'features (n,nL,H)={F_all.shape}')

    if args.layers:
        layer_idx = [int(x) for x in args.layers.split(',')]
    else:
        layer_idx = sorted(set(list(range(0, nL, 2)) + [0, 24, nL-1]))
    W, D = args.stimulus_window, args.hrf_delay
    curve = {}
    for L in layer_idx:
        r = score(F_all[:, L, :], Y, clip_rows, W, D, alphas)
        curve[L] = round(r, 4)
        log(f'  layer {L:2d}: r={r:.4f}')
    peak_L = max(curve, key=curve.get)
    out = {'n_TRs': int(len(sid)), 'n_parcels': int(Y.shape[1]),
           'stimulus_window': W, 'hrf_delay': D,
           'fusion_off_layer0_r': curve.get(0), 'peak_layer': int(peak_L),
           'peak_r': curve[peak_L], 'curve': {str(k): v for k, v in curve.items()}}
    json.dump(out, open(args.out_json, 'w'), indent=2)
    log(f'=== fusion OFF (layer 0) r={curve.get(0):.4f}  |  fusion ON (peak layer {peak_L}) r={curve[peak_L]:.4f}  '
        f'|  fusion gain +{curve[peak_L]-curve.get(0,0):.4f} ===')
    log(f'saved -> {args.out_json}')


if __name__ == '__main__':
    main()
