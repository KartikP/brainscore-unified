"""Movie->brain demo, DUAL STACK: predict the held-out Friends s01e02 window with
BOTH the best-layer TRIBEv2 multimodal stack (V-JEPA-2 L14 + Wav2Vec-Bert L12 +
Llama L14) AND native Qwen3-Omni (L42), render recorded-vs-predicted glass brains
for each, sharing one SD color scale across human + both models so the website
tabs are directly comparable. Features come from the all-layer per-clip caches
(no GPU/extraction). CPU-only.

Outputs (/tmp/clipdual): human/bold_*.png (shared), model_tribe/bold_*.png,
model_qwen/bold_*.png, meta.json (per_tr_r + mean_r for each stack).
"""
import argparse, json, os, time
from pathlib import Path
import numpy as np
t0 = time.time()
def log(m): print(f'[{time.time()-t0:6.1f}s] {m}', flush=True)
BR = '/home/ubuntu/.brainio/algonauts2025'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--segment', default='friends_s01e02a')
    ap.add_argument('--win_start', type=int, default=20)
    ap.add_argument('--win_n', type=int, default=7)
    ap.add_argument('--tr_sec', type=float, default=1.49)
    ap.add_argument('--window', type=int, default=5)
    ap.add_argument('--hrf', type=int, default=3)
    ap.add_argument('--out', default='/tmp/clipdual')
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    W, Dd = args.window, args.hrf

    import brainscore
    log('load assembly...')
    b = brainscore.load_benchmark('Algonauts2025-friends-sub01')
    asm = b.assembly
    sid = np.array([str(x) for x in asm['stimulus_id'].values])
    run = np.array([str(x) for x in asm['run'].values])
    tw = np.array(asm['t_within_run'].values, dtype=int)
    n = len(sid); Y = asm.values.astype(np.float32)
    log(f'  n_TR={n}')

    # best-layer feature loaders from all-layer per-clip caches (n_TRs, nL, dim)
    def load_layer(root, L):
        cache, feats = {}, None
        for i in range(n):
            cid = sid[i]
            if cid not in cache:
                p = Path(root)/f'{cid}.npy'
                cache[cid] = np.load(p) if p.exists() else None
            arr = cache[cid]
            if arr is None:
                if feats is None: raise RuntimeError(f'missing {cid}')
                continue
            if feats is None: feats = np.zeros((n, arr.shape[2]), np.float32)
            t = int(tw[i]); feats[i] = arr[min(t, arr.shape[0]-1), L].astype(np.float32)
        # which rows had features (clip present in this cache)
        present = np.array([sid[i] in cache and cache[sid[i]] is not None for i in range(n)])
        return feats, present

    log('load best-layer features (video L14, audio L12, text L14, qwen L42)...')
    Xv, pv = load_layer(f'{BR}/tribe_video_alllayers', 14)
    Xa, pa = load_layer(f'{BR}/tribe_audio_alllayers', 12)
    Xt, pt = load_layer(f'{BR}/tribe_text_alllayers', 14)
    Xq, pq = load_layer(f'{BR}/qwen_alllayers', 42)
    present = pv & pa & pt & pq          # rows usable for the demo (subset clips)
    log(f'  usable rows (subset): {int(present.sum())}/{n}')

    seen, ridx = {}, np.empty(n, np.int64)
    for i, (s, r) in enumerate(zip(sid, run)):
        k = (s, r); seen.setdefault(k, len(seen)); ridx[i] = seen[k]

    def stack(parts):
        Xs_parts, offsets, cur = [], [], 0
        for name, Xm in parts:
            nf = Xm.shape[1]; Xst = np.zeros((n, W*nf), np.float32)
            for off in range(W):
                shift = Dd + (W-1-off)
                if shift == 0: Xst[:, off*nf:(off+1)*nf] = Xm; continue
                src = np.arange(n)-shift; ok = src >= 0
                ok[ok] &= (ridx[src[ok]] == ridx[np.arange(n)[ok]])
                Xst[ok, off*nf:(off+1)*nf] = Xm[src[ok]]
            Xs_parts.append(Xst); offsets.append((cur, cur+W*nf, name)); cur += W*nf
        return np.concatenate(Xs_parts, 1).astype(np.float32), offsets

    def banded_fit_predict(Xs, offsets, alphas, tr_idx, sel):
        diag = np.zeros(Xs.shape[1], np.float32)
        for lo, hi, name in offsets: diag[lo:hi] = alphas[name]
        Xtr = Xs[tr_idx]; XtX = Xtr.T @ Xtr
        XtX[np.diag_indices_from(XtX)] += diag
        Wsol = np.linalg.solve(XtX, Xtr.T @ Y[tr_idx])
        return (Xs[sel] @ Wsol).astype(np.float32)

    seg_mask = (sid == args.segment)
    gidx = np.where(seg_mask)[0]
    sel = gidx[(tw[gidx] >= args.win_start) & (tw[gidx] < args.win_start+args.win_n)]
    sel = sel[np.argsort(tw[sel])]
    log(f'window TRs: {list(tw[sel])}')
    tr_idx = np.where(~seg_mask & present)[0]
    log(f'train rows: {len(tr_idx)}')

    Xs_t, off_t = stack([('video', Xv), ('audio', Xa), ('text', Xt)])
    Xs_q, off_q = stack([('qwen', Xq)])
    pred_tribe = banded_fit_predict(Xs_t, off_t, {'video':1e5,'audio':1e4,'text':1e3}, tr_idx, sel)
    pred_qwen = banded_fit_predict(Xs_q, off_q, {'qwen':1e4}, tr_idx, sel)
    gt = Y[sel].astype(np.float32)

    def per_tr_r(pred):
        out = []
        for t in range(pred.shape[0]):
            a, c = pred[t], gt[t]; ok = np.isfinite(a) & np.isfinite(c)
            if ok.sum() > 2 and np.std(a[ok]) > 0 and np.std(c[ok]) > 0:
                out.append(round(float(np.corrcoef(a[ok], c[ok])[0,1]), 4))
            else: out.append(None)
        return out
    r_tribe, r_qwen = per_tr_r(pred_tribe), per_tr_r(pred_qwen)
    mean_t = round(float(np.nanmean([x for x in r_tribe if x is not None])), 4)
    mean_q = round(float(np.nanmean([x for x in r_qwen if x is not None])), 4)
    log(f'TRIBEv2 per-TR r {r_tribe} mean={mean_t}')
    log(f'Qwen   per-TR r {r_qwen} mean={mean_q}')

    # common SD scale across human + both models
    def _z(a):
        f = a[np.isfinite(a)]; m = float(np.nanmean(f)); s = float(np.nanstd(f))
        return (a-m)/(s if s > 1e-8 else 1.0)
    gtz, ptz, pqz = _z(gt), _z(pred_tribe), _z(pred_qwen)
    allv = np.concatenate([gtz, ptz, pqz], 0); fin = allv[np.isfinite(allv)]
    M = float(np.nanpercentile(np.abs(fin), 98)); shared = dict(vmin=-M, vmax=M)
    log(f'shared SD scale ±{M:.3f}')

    from brainscore.visualization import glass_brain_movie as render
    times = [round(float(t*args.tr_sec), 1) for t in tw[sel]]
    kw = dict(cmap='RdBu_r', display_mode='ortho', symmetric=True,
              share_scale=True, n_parcels=1000, **shared)
    render(gtz,  times=times, out_dir=f'{args.out}/human',       prefix='bold', title_fmt='human · t={t:.1f}s', **kw)
    render(ptz,  times=times, out_dir=f'{args.out}/model_tribe', prefix='bold', title_fmt='TRIBEv2 · t={t:.1f}s', **kw)
    render(pqz,  times=times, out_dir=f'{args.out}/model_qwen',  prefix='bold', title_fmt='Qwen3-Omni · t={t:.1f}s', **kw)

    transcript = None
    try: transcript = [str(x) for x in asm['text_per_tr'].values[sel]]
    except Exception: pass
    json.dump({'segment': args.segment, 'tr_sec': args.tr_sec, 'win_n': int(len(sel)),
               'times': times, 'transcript': transcript,
               'tribe': {'per_tr_r': r_tribe, 'mean_r': mean_t},
               'qwen': {'per_tr_r': r_qwen, 'mean_r': mean_q}},
              open(f'{args.out}/meta.json', 'w'), indent=2)
    log(f'done -> {args.out}')


if __name__ == '__main__':
    main()
