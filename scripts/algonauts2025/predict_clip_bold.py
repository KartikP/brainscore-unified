"""Produce a real model->brain clip demo for the website (human vs model).

Reuses the Phase-3 cached features and the banded-ridge encoding model to
PREDICT subject-1's per-TR Schaefer-1000 BOLD for one Friends segment, fetches
the subject's ACTUALLY RECORDED BOLD for the same TRs, and renders BOTH on the
brain so the held-out prediction can be compared to ground truth.

Default video tower is V-JEPA-2 (``video_features_vjepa2.npz``, rung-2 upgrade;
the CLIP npz still works via ``--video_features``). Audio (Wav2Vec2) + text
(MiniLM) caches are reused as-is. When the video npz carries an ``extracted``
mask (V-JEPA was only run on a subset of TRs), training is restricted to filled
rows.

Time alignment note: the model's design matrix builds in the hemodynamic lag
(``shift = hrf + (window-1-off)``), and the recorded BOLD carries that same lag
physiologically — so model and human are on the SAME scan clock and are directly
comparable at each TR. Both lag the MOVIE by ~4.5 s; the website panel explains
this.

Rendering uses ``brainscore.visualization.quickbrain_outline_movie`` (the
glass-brain 3-view montage), one shared SYMMETRIC diverging scale per stream
(human gets its own, model gets its own — ridge predictions are regularized so
smaller in magnitude; the comparison is spatial).

Outputs (default /tmp/clipdemo):
  model/bold_###.png   — predicted BOLD per TR (3-view glass brain)
  human/bold_###.png   — recorded BOLD per TR (3-view glass brain)
  pred_parcels.npy, gt_parcels.npy, meta.json
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, '/home/ubuntu/brain-score-unified/unified')
import numpy as np

t0 = time.time()
def log(m): print(f'[{time.time()-t0:6.1f}s] {m}', flush=True)

BR = '/home/ubuntu/.brainio/algonauts2025'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--segment', default='friends_s01e02a')
    ap.add_argument('--win_start', type=int, default=20)   # first TR of the window
    ap.add_argument('--win_n', type=int, default=7)        # ~10.4s at TR 1.49
    ap.add_argument('--tr_sec', type=float, default=1.49)
    ap.add_argument('--train_cap', type=int, default=12000)
    ap.add_argument('--window', type=int, default=5)       # stimulus window (W)
    ap.add_argument('--hrf', type=int, default=3)          # HRF delay (D)
    ap.add_argument('--video_features', default=f'{BR}/video_features_vjepa2.npz')
    ap.add_argument('--renderer', default='glass',
                    choices=['glass', 'quickbrain', 'surface'])
    ap.add_argument('--per_stream_scale', action='store_true',
                    help='give human and model each its own color scale '
                         '(default: one shared scale across both)')
    ap.add_argument('--no_standardize', action='store_true',
                    help='shared scale in RAW BOLD units (model renders faint). '
                         'Default: z-score each stream to its own variance so the '
                         'shared scale is in SD units and both patterns stay vivid.')
    ap.add_argument('--out', default='/tmp/clipdemo')
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    W, Dd = args.window, args.hrf
    ALPHAS = {'video': 1e5, 'audio': 1e4, 'text': 1e3}   # CLAUDE.md banded optimum

    import brainscore
    log('load benchmark assembly...')
    b = brainscore.load_benchmark('Algonauts2025-friends-sub01')
    asm = b.assembly
    sid = np.array([str(x) for x in asm['stimulus_id'].values])
    run = np.array([str(x) for x in asm['run'].values])
    tw = np.array(asm['t_within_run'].values, dtype=int)
    n = len(sid)
    log(f'  n_TR={n}')

    log(f'load video features: {args.video_features}')
    vz = np.load(args.video_features)
    X_v = vz['X'].astype(np.float32)
    vid_extracted = vz['extracted'].astype(bool) if 'extracted' in vz.files \
        else np.ones(n, bool)
    log(f'  video {X_v.shape}  extracted={int(vid_extracted.sum())}/{n}')

    def per_tr_load(root):
        cache, feats = {}, None
        for i in range(n):
            cid = sid[i]
            if cid not in cache:
                p = Path(root) / f'{cid}.npy'
                cache[cid] = np.load(p) if p.exists() else None
            arr = cache[cid]
            if arr is None:
                if feats is None:
                    raise RuntimeError(f'missing first clip {cid}')
                feats[i] = 0.0
                continue
            if feats is None:
                feats = np.zeros((n, arr.shape[1]), np.float32)
            t = int(tw[i])
            feats[i] = arr[t] if t < arr.shape[0] else arr[-1]
        return feats

    log('load cached audio/text features...')
    X_a = per_tr_load(f'{BR}/audio_features_wav2vec2')
    X_t = per_tr_load(f'{BR}/text_features_minilm')
    parts = [('video', X_v), ('audio', X_a), ('text', X_t)]
    log(f'  video{X_v.shape} audio{X_a.shape} text{X_t.shape}')

    # run index = unique (segment, run)
    seen, ridx = {}, np.empty(n, np.int64)
    for i, (s, r) in enumerate(zip(sid, run)):
        k = (s, r)
        if k not in seen:
            seen[k] = len(seen)
        ridx[i] = seen[k]

    log(f'stack window={W} hrf={Dd}...')
    Xs_parts, offsets, cur = [], [], 0
    for name, Xm in parts:
        nf = Xm.shape[1]
        Xst = np.zeros((n, W * nf), np.float32)
        for off in range(W):
            shift = Dd + (W - 1 - off)
            if shift == 0:
                Xst[:, off*nf:(off+1)*nf] = Xm
                continue
            src = np.arange(n) - shift
            ok = (src >= 0)
            ok[ok] &= (ridx[src[ok]] == ridx[np.arange(n)[ok]])
            Xst[ok, off*nf:(off+1)*nf] = Xm[src[ok]]
        Xs_parts.append(Xst)
        offsets.append((cur, cur + W * nf, name))
        cur += W * nf
    Xs = np.concatenate(Xs_parts, axis=1).astype(np.float32)
    Y = asm.values.astype(np.float32)
    log(f'  Xs={Xs.shape} Y={Y.shape}')

    # target window: contiguous TRs in the chosen segment
    seg_mask = (sid == args.segment)
    if seg_mask.sum() == 0:
        raise SystemExit(f'segment {args.segment} not in assembly')
    gidx = np.where(seg_mask)[0]
    sel = gidx[(tw[gidx] >= args.win_start) & (tw[gidx] < args.win_start + args.win_n)]
    sel = sel[np.argsort(tw[sel])]
    if not vid_extracted[sel].all():
        raise SystemExit(f'target window TRs not all video-extracted: '
                         f'{list(tw[sel][~vid_extracted[sel]])}')
    log(f'target window TRs (t_within_run): {list(tw[sel])}')

    # train banded ridge on NON-target TRs that have video features extracted
    tr_idx = np.where(~seg_mask & vid_extracted)[0]
    rng = np.random.RandomState(0)
    if len(tr_idx) > args.train_cap:
        tr_idx = np.sort(rng.choice(tr_idx, args.train_cap, replace=False))
    log(f'fit banded ridge on {len(tr_idx)} TRs (alphas {ALPHAS})...')
    diag = np.zeros(Xs.shape[1], np.float32)
    for lo, hi, name in offsets:
        diag[lo:hi] = ALPHAS[name]
    Xtr = Xs[tr_idx]
    XtX = Xtr.T @ Xtr
    XtX[np.diag_indices_from(XtX)] += diag
    XtY = Xtr.T @ Y[tr_idx]
    Wsol = np.linalg.solve(XtX, XtY)

    pred = (Xs[sel] @ Wsol).astype(np.float32)            # (win_n, 1000) model
    gt = Y[sel].astype(np.float32)                        # (win_n, 1000) human (recorded)
    np.save(f'{args.out}/pred_parcels.npy', pred)
    np.save(f'{args.out}/gt_parcels.npy', gt)
    # held-out quality check: per-parcel correlation across the window
    if pred.shape[0] >= 3:
        pc = np.array([np.corrcoef(pred[:, j], gt[:, j])[0, 1] for j in range(pred.shape[1])])
        log(f'window pred-vs-recorded median per-parcel r = {np.nanmedian(pc):.3f}')

    # per-TR voxel-wise Pearson r ACROSS all parcels (human vs model) — animates
    # with the clip. One number per scan: how well the model's whole-brain pattern
    # matches the recorded whole-brain pattern at that moment.
    per_tr_r = []
    for t in range(pred.shape[0]):
        a, b = pred[t], gt[t]
        ok = np.isfinite(a) & np.isfinite(b)
        if ok.sum() > 2 and np.std(a[ok]) > 0 and np.std(b[ok]) > 0:
            per_tr_r.append(round(float(np.corrcoef(a[ok], b[ok])[0, 1]), 4))
        else:
            per_tr_r.append(None)
    log(f'per-TR voxel-wise r: {per_tr_r}')

    times = [round(float(t * args.tr_sec), 1) for t in tw[sel]]
    pred_r, gt_r = pred, gt
    if args.per_stream_scale:
        shared, unit = {}, 'per-stream'
    else:
        # Default: standardize each stream to its OWN variance, then a single
        # shared symmetric scale in SD units. This is a genuine common scale
        # (identical colorbar for both) that keeps BOTH spatial patterns vivid —
        # the model's ridge predictions are ~5x smaller in raw units, so a shared
        # RAW scale washes the model out. The magnitude gap stays honest via the
        # reported r and the note; the maps compare PATTERN. --no_standardize
        # falls back to a shared RAW scale.
        if not args.no_standardize:
            def _z(a):
                f = a[np.isfinite(a)]
                m = float(np.nanmean(f)) if f.size else 0.0
                s = float(np.nanstd(f)) if f.size else 1.0
                return (a - m) / (s if s > 1e-8 else 1.0)
            pred_r, gt_r = _z(pred), _z(gt)
            unit = 'SD'
        else:
            unit = 'raw'
        both = np.concatenate([pred_r, gt_r], axis=0)
        finite = both[np.isfinite(both)]
        M = float(np.nanpercentile(np.abs(finite), 98)) if finite.size else 1.0
        shared = dict(vmin=-M, vmax=M)
        log(f'  shared {unit} scale: ±{M:.3f}')
    log(f'render human + model ({args.renderer}, scale={unit})...')
    if args.renderer == 'glass':
        from brainscore.visualization import glass_brain_movie as render
        kw = dict(cmap='RdBu_r', display_mode='ortho', symmetric=True,
                  share_scale=True, n_parcels=1000, **shared)
    elif args.renderer == 'quickbrain':
        from brainscore.visualization import quickbrain_outline_movie as render
        kw = dict(cmap='RdBu_r', symmetric=True, share_scale=True, n_parcels=1000, **shared)
    else:
        from brainscore.visualization import cortical_surface_movie as render
        kw = dict(cmap='turbo', n_parcels=1000, hemi='left', view='lateral', **shared)
    render(pred_r, times=times, out_dir=f'{args.out}/model', prefix='bold',
           title_fmt='model  ·  t = {t:.1f}s', **kw)
    render(gt_r, times=times, out_dir=f'{args.out}/human', prefix='bold',
           title_fmt='human  ·  t = {t:.1f}s', **kw)

    transcript = None
    try:
        transcript = [str(x) for x in asm['text_per_tr'].values[sel]]
    except Exception:
        pass
    json.dump({'segment': args.segment, 'tr_sec': args.tr_sec,
               'win_start': int(args.win_start), 'win_n': int(len(sel)),
               'tw': [int(x) for x in tw[sel]], 'times': times,
               'renderer': args.renderer,
               'video_features': os.path.basename(args.video_features),
               'per_tr_voxelwise_r': per_tr_r,
               'transcript': transcript,
               'clip_start_sec': round(float(tw[sel][0] * args.tr_sec), 2),
               'clip_dur_sec': round(float(len(sel) * args.tr_sec), 2)},
              open(f'{args.out}/meta.json', 'w'), indent=2)
    log(f'done -> {args.out}')


if __name__ == '__main__':
    main()
