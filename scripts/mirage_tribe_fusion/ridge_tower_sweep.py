"""Per-tower layer sweep + fair best-vs-best comparison on Algonauts sub-01.

For each TRIBEv2 tower (video/audio/text, all-layers .npy of shape (n_TRs, nL, d)):
sweep every layer through the same ridge -> best layer + per-layer curve. Then
concat the three best-layer features -> the fair post-hoc-concat score. Reports
alongside native@peak (Qwen layer 42) so all arms are best-layer.

Reuses the lagged-design ridge from ridge_fusion_curve. Output: /tmp/tower_sweep.json
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
    g = {}
    for i, (c, t) in enumerate(zip(sid, tw)): g.setdefault(c, []).append((t, i))
    out = {}
    for c, lst in g.items(): lst.sort(); out[c] = [i for _, i in lst]
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
    X = lagged(F, clip_rows, win, delay); clips = np.array(list(clip_rows.keys()))
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=0); best = -1
    for a in alphas:
        preds = np.zeros_like(Y)
        for tr, te in kf.split(clips):
            r = np.concatenate([clip_rows[c] for c in clips[tr]]); e = np.concatenate([clip_rows[c] for c in clips[te]])
            preds[e] = Ridge(alpha=a).fit(X[r], Y[r]).predict(X[e])
        best = max(best, float(np.nanmedian(per_parcel_r(Y, preds))))
    return best


def load_tower(root, sid, clip_rows):
    """Load all-layers (n_TRs, nL, d) per clip -> aligned (n_sid, nL, d)."""
    root = Path(root); cache = {}; nL = d = None
    for c in set(sid):
        p = root / f'{c}.npy'
        if p.exists(): cache[c] = np.load(p); nL, d = cache[c].shape[1], cache[c].shape[2]
        else: cache[c] = None
    F = np.zeros((len(sid), nL, d), np.float32)
    for c, rows in clip_rows.items():
        arr = cache.get(c)
        if arr is None: continue
        for local, gi in enumerate(rows):
            if local < len(arr): F[gi] = arr[local].astype(np.float32)
    return F, nL


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--assembly_path', default='/home/ubuntu/.brainio/algonauts2025/algonauts2025_friends_sub01.nc')
    ap.add_argument('--clip_list', default='/home/ubuntu/clip_list.txt')
    ap.add_argument('--video', default='/home/ubuntu/.brainio/algonauts2025/tribe_video_alllayers')
    ap.add_argument('--audio', default='/home/ubuntu/.brainio/algonauts2025/tribe_audio_alllayers')
    ap.add_argument('--text', default='/home/ubuntu/.brainio/algonauts2025/tribe_text_alllayers')
    ap.add_argument('--stimulus_window', type=int, default=5)
    ap.add_argument('--hrf_delay', type=int, default=3)
    ap.add_argument('--alpha_grid', default='100,1000,10000')
    ap.add_argument('--step', type=int, default=2, help='sweep every Nth layer')
    ap.add_argument('--out_json', default='/tmp/tower_sweep.json')
    args = ap.parse_args()
    alphas = [float(a) for a in args.alpha_grid.split(',')]
    subset = set(l.strip() for l in open(args.clip_list) if l.strip())
    sid, tw, Y = load_assembly(args.assembly_path, subset)
    clip_rows = build_clip_order(sid, tw)
    W, D = args.stimulus_window, args.hrf_delay
    log(f'subset {len(set(sid))} clips, {len(sid)} TRs, Y={Y.shape}')

    res = {}
    best_feats = {}
    for name, root in [('video', args.video), ('audio', args.audio), ('text', args.text)]:
        F, nL = load_tower(root, sid, clip_rows)
        log(f'{name}: (n,nL,d)={F.shape}')
        layer_idx = sorted(set(list(range(0, nL, args.step)) + [nL-1]))
        curve = {}
        for L in layer_idx:
            curve[L] = round(score(F[:, L, :], Y, clip_rows, W, D, alphas), 4)
            log(f'  {name} layer {L}: r={curve[L]}')
        bL = max(curve, key=curve.get)
        res[name] = {'best_layer': int(bL), 'best_r': curve[bL], 'curve': {str(k): v for k, v in curve.items()}}
        best_feats[name] = F[:, bL, :]

    # fair post-hoc concat: best layer of each tower
    concat = np.concatenate([best_feats['video'], best_feats['audio'], best_feats['text']], axis=1)
    res['posthoc_concat_bestlayers'] = {'r': round(score(concat, Y, clip_rows, W, D, alphas), 4),
                                        'dim': int(concat.shape[1])}
    json.dump(res, open(args.out_json, 'w'), indent=2)
    log('=== TOWER SWEEP SUMMARY ===')
    for t in ['video', 'audio', 'text']:
        log(f"  {t:6s} best layer {res[t]['best_layer']:3d}  r={res[t]['best_r']}")
    log(f"  post-hoc concat (best layers): r={res['posthoc_concat_bestlayers']['r']}")
    log(f'saved -> {args.out_json}')


if __name__ == '__main__':
    main()
