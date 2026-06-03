"""Layer-contribution-per-modality on Algonauts — the MIRAGE Fig 4 analogue.

For each modality tower (video CLIP, audio Wav2Vec2, text MiniLM), extract
*per-layer* per-TR features (one forward pass gives all layers via
output_hidden_states), then score each layer's brain-prediction r against the
Algonauts BOLD on a subset. The result is a modality x layer matrix of how much
each layer contributes to predicting cortex — the same shape as MIRAGE's
per-modality cross-attention figure, but grounded in encoding performance.

Run on EC2 (GPU). Writes JSON {modality: [per-layer median r]} to --out.
"""
import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

AL = Path('/home/ubuntu/.brainio/algonauts2025')
TR_SEC = 1.49


def log(m, t0=[None]):
    if t0[0] is None:
        t0[0] = time.time()
    print(f'[{time.time()-t0[0]:6.1f}s] {m}', flush=True)


def stack_hrf(X, run_idx, W=5, D=3):
    n, f = X.shape
    out = np.zeros((n, W * f), dtype=np.float32)
    for off in range(W):
        sh = D + (W - 1 - off)
        for i in range(n):
            s = i - sh
            if 0 <= s < n and run_idx[s] == run_idx[i]:
                out[i, off*f:(off+1)*f] = X[s]
    return out


def ridge_r(X, Y, run_idx, alpha=1000.0, nsplits=5, seed=0):
    Xs = stack_hrf(X, run_idx)
    rng = np.random.RandomState(seed); folds = rng.randint(0, nsplits, len(Xs))
    pred = np.zeros_like(Y)
    for fo in range(nsplits):
        te = folds == fo; tr = ~te
        xm, ym = Xs[tr].mean(0, keepdims=True), Y[tr].mean(0, keepdims=True)
        Xtr = Xs[tr] - xm
        W_ = np.linalg.solve(Xtr.T @ Xtr + alpha*np.eye(Xtr.shape[1]), Xtr.T @ (Y[tr]-ym))
        pred[te] = (Xs[te]-xm) @ W_ + ym
    P = pred - pred.mean(0, keepdims=True); Yc = Y - Y.mean(0, keepdims=True)
    r = (P*Yc).sum(0) / (np.sqrt((P**2).sum(0)*(Yc**2).sum(0)) + 1e-9)
    return float(np.median(r))


def per_tr_from_frames(hidden_per_frame, n_TRs, fps):
    # hidden_per_frame: (n_layers, n_frames, dim). Aggregate frames -> TR.
    nL, nF, dim = hidden_per_frame.shape
    fpt = fps * TR_SEC
    out = np.zeros((nL, n_TRs, dim), dtype=np.float32)
    for tr in range(n_TRs):
        f0, f1 = int(round(tr*fpt)), int(round((tr+1)*fpt))
        f0, f1 = min(f0, nF-1), min(max(f1, f0+1), nF)
        out[:, tr] = hidden_per_frame[:, f0:f1].mean(1)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n_clips', type=int, default=25)
    ap.add_argument('--out', default='/tmp/layer_contribution.json')
    args = ap.parse_args()
    import brainscore
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    log('load assembly')
    b = brainscore.load_benchmark('Algonauts2025-friends-sub01')
    a_stim = np.array([str(s) for s in b.assembly['stimulus_id'].values])
    a_run = np.array([str(r) for r in b.assembly['run'].values])
    a_t = np.array([int(t) for t in b.assembly['t_within_run'].values])
    Y_all = np.asarray(b.assembly.values, dtype=np.float32)

    clips = list(dict.fromkeys(a_stim.tolist()))[:args.n_clips]
    mask = np.isin(a_stim, clips)
    Y = Y_all[mask]
    sub_stim, sub_run = a_stim[mask], a_run[mask]
    seen, run_idx = {}, np.empty(mask.sum(), dtype=np.int64)
    for i, (s, r) in enumerate(zip(sub_stim, sub_run)):
        seen.setdefault((s, r), len(seen)); run_idx[i] = seen[(s, r)]
    log(f'{len(clips)} clips, {Y.shape[0]} TRs, {Y.shape[1]} parcels')

    results = {}

    # ---- AUDIO: Wav2Vec2 per-layer ----
    log('audio: Wav2Vec2')
    import soundfile as sf
    from transformers import Wav2Vec2Model, AutoFeatureExtractor
    fe = AutoFeatureExtractor.from_pretrained('facebook/wav2vec2-base-960h')
    am = Wav2Vec2Model.from_pretrained('facebook/wav2vec2-base-960h').to(device).eval()
    nL_a = am.config.num_hidden_layers + 1
    perTR = {L: [] for L in range(nL_a)}
    for clip in clips:
        n_clip = int((sub_stim == clip).sum())
        wavp = AL / 'audio_wav' / f'{clip}.wav'
        if not wavp.exists():
            for L in range(nL_a):
                perTR[L].append(np.zeros((n_clip, am.config.hidden_size), np.float32)); continue
        wav, sr = sf.read(str(wavp))
        if wav.ndim > 1:
            wav = wav.mean(1)
        feats = [[] for _ in range(nL_a)]
        for st in range(0, len(wav), 30*sr):
            ch = wav[st:st+30*sr]
            if len(ch) < sr:
                continue
            inp = fe(ch, sampling_rate=sr, return_tensors='pt').to(device)
            with torch.no_grad():
                hs = am(**inp, output_hidden_states=True).hidden_states
            for L in range(nL_a):
                feats[L].append(hs[L][0].cpu().float().numpy())
        fr = am.config.hidden_size  # placeholder
        for L in range(nL_a):
            arr = np.concatenate(feats[L], 0) if feats[L] else np.zeros((1, am.config.hidden_size), np.float32)
            fpt = (len(arr) / max(1, n_clip))
            tr_feat = np.zeros((n_clip, arr.shape[1]), np.float32)
            for tr in range(n_clip):
                f0, f1 = int(tr*fpt), int((tr+1)*fpt)
                tr_feat[tr] = arr[f0:min(max(f1, f0+1), len(arr))].mean(0) if len(arr) else 0
            perTR[L].append(tr_feat)
    audio_scores = []
    for L in range(nL_a):
        X = np.concatenate(perTR[L], 0)
        audio_scores.append(ridge_r(X, Y, run_idx))
    results['audio'] = audio_scores
    log(f'audio per-layer: {[round(x,3) for x in audio_scores]}')
    del am; torch.cuda.empty_cache()

    # ---- TEXT: MiniLM per-layer (from cached per-clip final not enough; re-encode TR text) ----
    log('text: MiniLM')
    import pandas as pd
    from transformers import AutoTokenizer, AutoModel
    stim_df = pd.read_csv(AL / 'algonauts2025_stim_friends.csv')
    stim_df['stimulus_id'] = stim_df['stimulus_id'].astype(str)
    tok = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
    tm = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2').to(device).eval()
    nL_t = tm.config.num_hidden_layers + 1
    # Read per-TR text from each clip's transcript TSV (text_per_tr column).
    perTR_t = {L: [] for L in range(nL_t)}
    for clip in clips:
        n_clip = int((sub_stim == clip).sum())
        texts = ['.'] * n_clip
        row = stim_df[stim_df['stimulus_id'] == clip]
        if len(row):
            tp = row['transcript_path'].iloc[0]
            if isinstance(tp, str) and os.path.exists(tp):
                tdf = pd.read_csv(tp, sep='\t')
                tt = [(str(x).strip() if isinstance(x, str) and str(x).strip() else '.')
                      for x in tdf['text_per_tr'].tolist()]
                for i in range(n_clip):
                    texts[i] = tt[i] if i < len(tt) else '.'
        inp = tok(texts, padding=True, truncation=True, max_length=64, return_tensors='pt').to(device)
        with torch.no_grad():
            hs = tm(**inp, output_hidden_states=True).hidden_states
        for L in range(nL_t):
            perTR_t[L].append(hs[L].mean(1).cpu().float().numpy())  # mean-pool tokens
    results['text'] = [ridge_r(np.concatenate(perTR_t[L], 0), Y, run_idx) for L in range(nL_t)]
    log(f'text per-layer: {[round(x,3) for x in results["text"]]}')
    del tm; torch.cuda.empty_cache()

    # ---- VIDEO: CLIP per-layer ----
    log('video: CLIP')
    from PIL import Image
    from transformers import CLIPVisionModel, AutoProcessor
    cp = AutoProcessor.from_pretrained('openai/clip-vit-base-patch32')
    cm = CLIPVisionModel.from_pretrained('openai/clip-vit-base-patch32').to(device).eval()
    nL_v = cm.config.num_hidden_layers + 1
    perTR_v = {L: [] for L in range(nL_v)}
    for clip in clips:
        n_clip = int((sub_stim == clip).sum())
        fdir = AL / 'frames' / clip
        frames = sorted(fdir.glob('*.jpg')) + sorted(fdir.glob('*.png'))
        if not frames:
            for L in range(nL_v):
                perTR_v[L].append(np.zeros((n_clip, cm.config.hidden_size), np.float32)); continue
        # sample up to ~3 frames/TR
        idxs = np.linspace(0, len(frames)-1, min(len(frames), n_clip*2)).astype(int)
        imgs = [Image.open(frames[i]).convert('RGB') for i in idxs]
        hid = [[] for _ in range(nL_v)]
        for k in range(0, len(imgs), 16):
            inp = cp(images=imgs[k:k+16], return_tensors='pt').to(device)
            with torch.no_grad():
                hs = cm(**inp, output_hidden_states=True).hidden_states
            for L in range(nL_v):
                hid[L].append(hs[L][:, 0].cpu().float().numpy())  # CLS
        fpt = len(idxs) / max(1, n_clip)
        for L in range(nL_v):
            arr = np.concatenate(hid[L], 0)
            tr_feat = np.stack([arr[int(tr*fpt):min(max(int((tr+1)*fpt), int(tr*fpt)+1), len(arr))].mean(0)
                                for tr in range(n_clip)])
            perTR_v[L].append(tr_feat)
    results['video'] = [ridge_r(np.concatenate(perTR_v[L], 0), Y, run_idx) for L in range(nL_v)]
    log(f'video per-layer: {[round(x,3) for x in results["video"]]}')

    with open(args.out, 'w') as f:
        json.dump(results, f, indent=2)
    log(f'written {args.out}')


if __name__ == '__main__':
    main()
