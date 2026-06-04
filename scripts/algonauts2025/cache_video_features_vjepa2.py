"""Extract per-TR V-JEPA-2 ViT-L video features for the movie->brain demo.

Rung-2 upgrade of the Algonauts encoding model: replace the CLIP frame-aggregate
video tower with V-JEPA-2 (the Algonauts-2025-winning video backbone family).
Audio (Wav2Vec2) and text (MiniLM) caches already cover every clip, so ONLY the
video tower is re-extracted here.

Per TR we take the frames in that TR's 1.49 s window, run them through the
registered ``vjepa2-vitl`` preprocessing + encoder (layer 16, skip_predictor),
spatial-mean-pool (the registration's post-hook) and then mean over the 32
temporal tubelets -> one 1024-d vector per TR. The exact preprocessing and
reshape are imported from ``brainscore.models.vjepa.model`` so this stays
bit-consistent with how the model scores on Lahner2024.

Extraction is bounded to the videos we actually need: the held-out TARGET video
plus a same-season training pool (capped at ``--max_train_tr`` TRs), because a
full V-JEPA pass over all 162k Friends TRs would take ~a day. Output is a single
.npz with ``X`` (n_TR_total, 1024, zeros where not extracted) and an
``extracted`` boolean mask so the predictor trains only on filled rows.

Output (default): /home/ubuntu/.brainio/algonauts2025/video_features_vjepa2.npz
EC2 g5.4xlarge (A10G-24GB). ~half a day including the render that follows.
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, '/home/ubuntu/brain-score-unified/unified')
import numpy as np

t0 = time.time()
def log(m): print(f'[{time.time()-t0:7.1f}s] {m}', flush=True)

TR_SEC = 1.49
LAYER_IDX = 16            # encoder.layer.16 — V-JEPA brain-optimal layer (Lahner sweep)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--target', default='friends_s01e02a')
    ap.add_argument('--prefer_prefix', default='friends_s01',
                    help='draw the training pool from clips with this id prefix first')
    ap.add_argument('--max_train_tr', type=int, default=6000)
    ap.add_argument('--batch_size', type=int, default=4)
    ap.add_argument('--fp16', action='store_true', default=True)
    ap.add_argument('--out', default='/home/ubuntu/.brainio/algonauts2025/video_features_vjepa2.npz')
    args = ap.parse_args()

    import torch
    from transformers import VJEPA2Model, VJEPA2VideoProcessor
    from brainscore.models.vjepa.model import (
        _make_preprocessing, _vjepa_post_hook, NUM_INPUT_FRAMES)
    import brainscore

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dtype = torch.float16 if (args.fp16 and device == 'cuda') else torch.float32

    log('load assembly + stimulus set...')
    b = brainscore.load_benchmark('Algonauts2025-friends-sub01')
    asm = b.assembly
    sid = np.array([str(x) for x in asm['stimulus_id'].values])
    tw = np.array(asm['t_within_run'].values, dtype=int)
    n = len(sid)
    ss = b._load_stimulus_set()
    sid2path = {str(s): str(p) for s, p in zip(ss['stimulus_id'], ss['video_path'])}

    # choose videos: target first, then same-prefix pool until max_train_tr TRs
    order = []
    if args.target in sid2path:
        order.append(args.target)
    pool = [s for s in dict.fromkeys(sid) if s != args.target and s in sid2path]
    pool.sort(key=lambda s: (not s.startswith(args.prefer_prefix), s))
    budget = args.max_train_tr
    for s in pool:
        if budget <= 0:
            break
        order.append(s)
        budget -= int((sid == s).sum())
    extract_set = set(order)
    log(f'  videos to extract: {len(order)} (target + pool); '
        f'~{int(sum((sid==s).sum() for s in order))} TRs')

    log(f'load vjepa2-vitl ({dtype})...')
    checkpoint = 'facebook/vjepa2-vitl-fpc64-256'
    vjepa = VJEPA2Model.from_pretrained(checkpoint, torch_dtype=dtype).to(device).eval()
    processor = VJEPA2VideoProcessor.from_pretrained(checkpoint)
    preprocess = _make_preprocessing(processor)

    captured = {}
    def hook(_m, _i, out):
        captured['x'] = (out[0] if isinstance(out, tuple) else out).detach()
    h = vjepa.encoder.layer[LAYER_IDX].register_forward_hook(hook)

    @torch.no_grad()
    def forward_batch(tensors):
        # tensors: list of (T,C,H,W) -> (B,T,C,H,W); hook captures (B, 8192, 1024)
        batch = torch.stack(tensors).to(device=device, dtype=dtype)
        vjepa(pixel_values_videos=batch, skip_predictor=True)
        arr = captured['x'].float().cpu().numpy()           # (B, 8192, 1024)
        bt = _vjepa_post_hook(arr)                           # (B, 32, 1024)
        return bt.mean(axis=1).astype(np.float32)            # (B, 1024) — mean over tubelets

    import cv2
    X = None
    extracted = np.zeros(n, bool)

    def ensure_X(feat_dim):
        nonlocal X
        if X is None:
            X = np.zeros((n, feat_dim), np.float32)

    for vi, stim in enumerate(order):
        path = sid2path[stim]
        rows = np.where(sid == stim)[0]                      # assembly rows for this video
        tw_rows = tw[rows]
        row_for_tr = {int(t): int(r) for t, r in zip(tw_rows, rows)}
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            log(f'  [skip] cannot open {path}')
            continue
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        max_tr = int(max(tw_rows))
        # stream frames, bucket into TR windows, batched forward as TRs complete
        buckets = {}          # tr -> list[RGB frame]
        pend_tensors, pend_rows = [], []
        done = 0

        def flush():
            nonlocal done
            if not pend_tensors:
                return
            feats = forward_batch(pend_tensors)
            ensure_X(feats.shape[1])
            for r, f in zip(pend_rows, feats):
                X[r] = f
                extracted[r] = True
            done += len(pend_rows)
            pend_tensors.clear(); pend_rows.clear()

        def finalize_tr(tr):
            frames = buckets.pop(tr, None)
            if not frames or tr not in row_for_tr:
                return
            pend_tensors.append(preprocess(frames))
            pend_rows.append(row_for_tr[tr])
            if len(pend_tensors) >= args.batch_size:
                flush()

        fidx = 0
        cur_tr = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            tr = int((fidx / fps) / TR_SEC)
            if tr > cur_tr:
                for t in range(cur_tr, tr):       # finalize any completed TR(s)
                    finalize_tr(t)
                cur_tr = tr
            if tr <= max_tr and tr in row_for_tr:
                buckets.setdefault(tr, []).append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            fidx += 1
        for t in list(buckets.keys()):            # finalize trailing TRs
            finalize_tr(t)
        flush()
        cap.release()
        # TRs with no decoded frames (video shorter than fMRI run): leave zero,
        # extracted stays False -> predictor drops them from training.
        log(f'  [{vi+1}/{len(order)}] {stim}: filled {done}/{len(rows)} TRs')

    h.remove()
    if X is None:
        raise SystemExit('no features extracted')
    np.savez_compressed(args.out, X=X, extracted=extracted)
    log(f'saved {args.out}  (extracted {int(extracted.sum())}/{n} TRs, dim={X.shape[1]})')


if __name__ == '__main__':
    main()
