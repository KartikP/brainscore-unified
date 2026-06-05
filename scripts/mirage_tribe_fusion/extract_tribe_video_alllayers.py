"""TRIBEv2 video tower, ALL layers: per-TR V-JEPA-2 ViT-L features at every
encoder layer (for the per-tower depth sweep). Saves (n_TRs, nL, 1024) float16.
Adapted from extract_tribe_video.py (output_hidden_states instead of one hook)."""
import argparse, sys, time
from pathlib import Path
from collections import Counter
sys.path.insert(0, '/home/ubuntu/brain-score-unified/unified')
import numpy as np
TR_SEC = 1.49


def log(m, t0): print(f'[{time.time()-t0:7.1f}s] {m}', flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stim_csv', default='/home/ubuntu/.brainio/algonauts2025/algonauts2025_stim_friends.csv')
    ap.add_argument('--features_root', default='/home/ubuntu/.brainio/algonauts2025/tribe_video_alllayers')
    ap.add_argument('--assembly_path', default='/home/ubuntu/.brainio/algonauts2025/algonauts2025_friends_sub01.nc')
    ap.add_argument('--clip_list', default='/home/ubuntu/clip_list.txt')
    ap.add_argument('--batch_size', type=int, default=4)
    args = ap.parse_args()
    t0 = time.time()

    import torch, cv2, pandas as pd, xarray as xr
    from transformers import VJEPA2Model, VJEPA2VideoProcessor
    from brainscore.models.vjepa.model import (_make_preprocessing,
        NUM_TEMPORAL_STEPS, NUM_SPATIAL_PATCHES)

    da = xr.open_dataarray(args.assembly_path)
    sid_all = np.array([str(x) for x in da['stimulus_id'].values])
    tw_all = np.array(da['t_within_run'].values, dtype=int)
    stim_to_n = dict(Counter(sid_all.tolist()))
    subset = set(l.strip() for l in open(args.clip_list) if l.strip()) if args.clip_list else None
    df = pd.read_csv(args.stim_csv)
    sid2path = {str(s): str(p) for s, p in zip(df['stimulus_id'], df['video_path'])}

    device = 'cuda'; dtype = torch.float16
    ckpt = 'facebook/vjepa2-vitl-fpc64-256'
    vjepa = VJEPA2Model.from_pretrained(ckpt, torch_dtype=dtype).to(device).eval()
    preprocess = _make_preprocessing(VJEPA2VideoProcessor.from_pretrained(ckpt))
    nL = vjepa.config.num_hidden_layers + 1
    log(f'vjepa loaded, nL={nL}', t0)

    T, S = NUM_TEMPORAL_STEPS, NUM_SPATIAL_PATCHES

    @torch.no_grad()
    def forward_batch(tensors):
        batch = torch.stack(tensors).to(device=device, dtype=dtype)
        out = vjepa(pixel_values_videos=batch, skip_predictor=True, output_hidden_states=True)
        # GPU-side pooling (avoid 25x huge CPU transfers): each (B, T*S, H)
        # -> reshape (B,T,S,H) -> mean spatial -> mean time -> (B,H). Transfer
        # only the tiny (B,H) per layer.
        per_layer = []
        for hs in out.hidden_states:
            B = hs.shape[0]
            v = hs.reshape(B, T, S, hs.shape[-1]).mean(dim=2).mean(dim=1)  # (B,H) on GPU
            per_layer.append(v.float().cpu().numpy())
        return np.stack(per_layer, axis=1).astype(np.float16)  # (B, nL, H)

    feat_root = Path(args.features_root); feat_root.mkdir(parents=True, exist_ok=True)
    n_done = n_skip = 0
    clips = [s for s in dict.fromkeys(sid_all.tolist()) if (subset is None or s in subset)]
    for stim in clips:
        if stim not in sid2path: continue
        fp = feat_root / f'{stim}.npy'
        if fp.exists(): n_skip += 1; continue
        n_TRs = stim_to_n[stim]
        per = np.zeros((n_TRs, nL, 1024), np.float16)
        rows = np.where(sid_all == stim)[0]
        tw_sorted = sorted(int(tw_all[r]) for r in rows)
        tr_local = {t: i for i, t in enumerate(tw_sorted)}
        cap = cv2.VideoCapture(sid2path[stim])
        if not cap.isOpened(): log(f'cannot open {stim}', t0); continue
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        buckets = {}; pend_t, pend_i = [], []
        def flush():
            if not pend_t: return
            feats = forward_batch(pend_t)        # (B, nL, 1024)
            for li, f in zip(pend_i, feats): per[li] = f
            pend_t.clear(); pend_i.clear()
        def finalize(tr):
            frames = buckets.pop(tr, None)
            if not frames or tr not in tr_local: return
            pend_t.append(preprocess(frames)); pend_i.append(tr_local[tr])
            if len(pend_t) >= args.batch_size: flush()
        fidx = 0; cur = 0; maxtr = max(tw_sorted)
        while True:
            ok, fr = cap.read()
            if not ok: break
            tr = int((fidx/fps)/TR_SEC)
            if tr > cur:
                for t in range(cur, tr): finalize(t)
                cur = tr
            if tr <= maxtr and tr in tr_local:
                buckets.setdefault(tr, []).append(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB))
            fidx += 1
        for t in list(buckets): finalize(t)
        flush(); cap.release()
        np.save(fp, per); n_done += 1
        log(f'done={n_done} skip={n_skip} ({stim} -> {per.shape})', t0)
    log(f'FINAL done={n_done} skip={n_skip}', t0)


if __name__ == '__main__':
    main()
