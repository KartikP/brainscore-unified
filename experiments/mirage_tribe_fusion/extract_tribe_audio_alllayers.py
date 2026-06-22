"""TRIBEv2 audio tower, ALL layers: per-TR Wav2Vec-Bert features at every layer.
Saves (n_TRs, nL, 1024) float16. Adapted from extract_tribe_audio.py."""
import argparse, subprocess, time
from pathlib import Path
from collections import Counter
import numpy as np, pandas as pd
TR_SEC = 1.49; SR = 16000


def log(m, t0): print(f'[{time.time()-t0:7.1f}s] {m}', flush=True)


def extract_wav(mkv, wav):
    if wav.exists() and wav.stat().st_size > 1024: return
    wav.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(['ffmpeg','-y','-loglevel','error','-i',str(mkv),'-ac','1','-ar',str(SR),str(wav)], check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stim_csv', default='/home/ubuntu/.brainio/algonauts2025/algonauts2025_stim_friends.csv')
    ap.add_argument('--audio_root', default='/home/ubuntu/.brainio/algonauts2025/audio_wav')
    ap.add_argument('--features_root', default='/home/ubuntu/.brainio/algonauts2025/tribe_audio_alllayers')
    ap.add_argument('--assembly_path', default='/home/ubuntu/.brainio/algonauts2025/algonauts2025_friends_sub01.nc')
    ap.add_argument('--clip_list', default='/home/ubuntu/clip_list.txt')
    ap.add_argument('--chunk_sec', type=float, default=20.0)
    args = ap.parse_args()
    t0 = time.time()

    import torch, soundfile as sf, xarray as xr
    from transformers import SeamlessM4TFeatureExtractor, Wav2Vec2BertModel
    da = xr.open_dataarray(args.assembly_path)
    stim_to_n = dict(Counter(list(da['stimulus_id'].values)))
    subset = set(l.strip() for l in open(args.clip_list) if l.strip()) if args.clip_list else None
    fe = SeamlessM4TFeatureExtractor.from_pretrained('facebook/w2v-bert-2.0')
    model = Wav2Vec2BertModel.from_pretrained('facebook/w2v-bert-2.0', output_hidden_states=True).eval().to('cuda')
    H = model.config.hidden_size; nL = model.config.num_hidden_layers + 1
    log(f'w2v-bert H={H} nL={nL}', t0)
    df = pd.read_csv(args.stim_csv)
    feat_root = Path(args.features_root); feat_root.mkdir(parents=True, exist_ok=True)
    n_done = n_skip = 0
    for _, srow in df.iterrows():
        sid = srow['stimulus_id']; n_TRs = stim_to_n.get(sid)
        if n_TRs is None or (subset and sid not in subset): continue
        fp = feat_root / f'{sid}.npy'
        if fp.exists(): n_skip += 1; continue
        mkv = Path(srow['video_path'])
        if not mkv.exists(): continue
        wav = Path(args.audio_root)/f'{sid}.wav'
        try: extract_wav(mkv, wav)
        except subprocess.CalledProcessError: continue
        w, sr = sf.read(str(wav), dtype='float32')
        if w.ndim > 1: w = w.mean(1)
        if sr != SR: continue
        chunk = int(args.chunk_sec*SR); per_layer_frames = [[] for _ in range(nL)]; rate = None
        with torch.no_grad():
            for s in range(0, len(w), chunk):
                c = w[s:s+chunk]
                if len(c) < SR: continue
                inp = fe(c, sampling_rate=SR, return_tensors='pt').to('cuda')
                out = model(**inp)
                for li, hs in enumerate(out.hidden_states):
                    per_layer_frames[li].append(hs[0].float().cpu().numpy())  # (T,H)
                if rate is None: rate = out.hidden_states[0].shape[1]/(len(c)/SR)
        if rate is None: continue
        feats = [np.concatenate(fr, 0) for fr in per_layer_frames]  # list nL of (T_total,H)
        fpr = rate*TR_SEC
        per = np.zeros((n_TRs, nL, H), np.float16)
        for li in range(nL):
            F = feats[li]
            for i in range(n_TRs):
                a=int(round(i*fpr)); b=min(int(round((i+1)*fpr)), len(F))
                if a>=len(F): per[i,li]=per[i-1,li] if i>0 else per[i,li]; continue
                per[i,li]=F[min(a,len(F)-1):b].mean(0).astype(np.float16)
        np.save(fp, per); n_done += 1
        log(f'done={n_done} skip={n_skip} ({sid} -> {per.shape})', t0)
    log(f'FINAL done={n_done} skip={n_skip}', t0)


if __name__ == '__main__':
    main()
