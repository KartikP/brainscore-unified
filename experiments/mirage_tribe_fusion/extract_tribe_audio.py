"""TRIBEv2 audio tower: per-TR Wav2Vec-Bert-2.0 features for Algonauts clips.

Adapted from the Phase-3b Wav2Vec2 extractor. Two differences:
  * facebook/w2v-bert-2.0 needs SeamlessM4TFeatureExtractor + Wav2Vec2BertModel
    directly (AutoProcessor fails on its CTC tokenizer — #18 finding).
  * frame rate is measured empirically per chunk (n_frames / chunk_seconds)
    rather than hardcoded, so per-TR binning is exact for this model.

Saves (n_TRs, 1024) float32 .npy keyed by stim_id. Subset via --clip_list.
Run in the tribe_test env (transformers 4.57.6) on GPU.
"""
import argparse, subprocess, time
from pathlib import Path
from collections import Counter
import numpy as np
import pandas as pd

TR_SEC = 1.49
SR = 16000


def log(m, t0): print(f'[{time.time()-t0:7.1f}s] {m}', flush=True)


def extract_wav(mkv, wav):
    if wav.exists() and wav.stat().st_size > 1024: return
    wav.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(['ffmpeg','-y','-loglevel','error','-i',str(mkv),
                    '-ac','1','-ar',str(SR),str(wav)], check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stim_csv', default='/home/ubuntu/.brainio/algonauts2025/algonauts2025_stim_friends.csv')
    ap.add_argument('--audio_root', default='/home/ubuntu/.brainio/algonauts2025/audio_wav')
    ap.add_argument('--features_root', default='/home/ubuntu/.brainio/algonauts2025/tribe_audio_w2vbert')
    ap.add_argument('--assembly_path', default='/home/ubuntu/.brainio/algonauts2025/algonauts2025_friends_sub01.nc')
    ap.add_argument('--clip_list', default='', help='optional newline file of stim_ids to restrict to (the subset)')
    ap.add_argument('--chunk_sec', type=float, default=20.0)
    args = ap.parse_args()
    t0 = time.time()

    import torch
    import soundfile as sf
    from transformers import SeamlessM4TFeatureExtractor, Wav2Vec2BertModel
    import xarray as xr

    da = xr.open_dataarray(args.assembly_path)
    stim_to_n = dict(Counter(list(da['stimulus_id'].values)))
    subset = None
    if args.clip_list:
        subset = set(l.strip() for l in open(args.clip_list) if l.strip())
    log(f'{len(stim_to_n)} clips in assembly; subset={len(subset) if subset else "ALL"}', t0)

    fe = SeamlessM4TFeatureExtractor.from_pretrained('facebook/w2v-bert-2.0')
    model = Wav2Vec2BertModel.from_pretrained('facebook/w2v-bert-2.0').eval()
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'; model = model.to(dev)
    HID = model.config.hidden_size
    log(f'w2v-bert hidden={HID}', t0)

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

        chunk = int(args.chunk_sec*SR); frames=[]; rate=None
        with torch.no_grad():
            for s in range(0, len(w), chunk):
                c = w[s:s+chunk]
                if len(c) < SR: continue
                inp = fe(c, sampling_rate=SR, return_tensors='pt').to(dev)
                out = model(**inp).last_hidden_state[0].cpu().numpy()  # (T,HID)
                frames.append(out)
                if rate is None: rate = len(out)/(len(c)/SR)  # empirical Hz
        if not frames: continue
        feats = np.concatenate(frames, 0)
        fpr = rate*TR_SEC
        per = np.zeros((n_TRs, HID), np.float32)
        for i in range(n_TRs):
            a=int(round(i*fpr)); b=min(int(round((i+1)*fpr)), len(feats))
            if a>=len(feats): per[i]=per[i-1] if i>0 else per[i]; continue
            per[i]=feats[min(a,len(feats)-1):b].mean(0)
        np.save(fp, per); n_done+=1
        if n_done%10==0: log(f'done={n_done} skip={n_skip} (rate~{rate:.1f}Hz, {sid}->{per.shape})', t0)
    log(f'FINAL done={n_done} skip={n_skip}', t0)


if __name__ == '__main__':
    main()
