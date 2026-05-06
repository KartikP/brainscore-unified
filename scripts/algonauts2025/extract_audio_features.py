"""Extract per-TR Wav2Vec2-base features for every Friends/Movie10 clip.

For each clip:
1. ffmpeg → 16 kHz mono wav (cached)
2. Wav2Vec2-base forward in 30-sec chunks
3. Mean-pool the model's frame-rate output (50 Hz) into per-TR bins
4. Save (n_TRs, 768) float32 .npy keyed by clip_id

Run on EC2 (GPU). ~30 min for ~330 clips.
"""
import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

TR_SEC = 1.49
SR = 16000  # Wav2Vec2 expects 16 kHz mono


def log(msg, t0):
    print(f'[{time.time() - t0:7.1f}s] {msg}', flush=True)


def extract_wav(mkv_path: Path, wav_path: Path):
    """ffmpeg: 16 kHz mono wav. Skip if cached."""
    if wav_path.exists() and wav_path.stat().st_size > 1024:
        return
    wav_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        'ffmpeg', '-y', '-loglevel', 'error',
        '-i', str(mkv_path),
        '-ac', '1', '-ar', str(SR),
        str(wav_path),
    ]
    subprocess.run(cmd, check=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stim_csv',
                        default='/home/ubuntu/.brainio/algonauts2025/algonauts2025_stim_friends.csv')
    parser.add_argument('--audio_root',
                        default='/home/ubuntu/.brainio/algonauts2025/audio_wav')
    parser.add_argument('--features_root',
                        default='/home/ubuntu/.brainio/algonauts2025/audio_features_wav2vec2')
    parser.add_argument('--assembly_path',
                        default='/home/ubuntu/.brainio/algonauts2025/algonauts2025_friends_sub01.nc')
    parser.add_argument('--chunk_sec', type=float, default=30.0)
    args = parser.parse_args()

    t0 = time.time()
    log('imports...', t0)
    import torch
    import soundfile as sf
    from transformers import AutoFeatureExtractor, AutoModel
    import xarray as xr

    log('load assembly to get n_TRs per clip...', t0)
    da = xr.open_dataarray(args.assembly_path)
    sids = list(da['stimulus_id'].values)
    from collections import Counter
    stim_to_n_TRs = dict(Counter(sids))
    log(f'  {len(stim_to_n_TRs)} clips in assembly', t0)

    log('load Wav2Vec2-base model...', t0)
    fe = AutoFeatureExtractor.from_pretrained('facebook/wav2vec2-base')
    model = AutoModel.from_pretrained('facebook/wav2vec2-base').eval()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    HIDDEN = model.config.hidden_size  # 768
    FRAME_RATE_HZ = SR / 320  # Wav2Vec2 stride: 320 samples @ 16 kHz → 50 Hz
    log(f'  hidden={HIDDEN}, frame_rate={FRAME_RATE_HZ:.1f} Hz', t0)

    df = pd.read_csv(args.stim_csv)
    audio_root = Path(args.audio_root)
    feat_root = Path(args.features_root)
    feat_root.mkdir(parents=True, exist_ok=True)

    n_done, n_skipped, n_missing_video = 0, 0, 0
    for ridx, srow in df.iterrows():
        stim_id = srow['stimulus_id']
        n_TRs = stim_to_n_TRs.get(stim_id)
        if n_TRs is None:
            continue

        feat_path = feat_root / f'{stim_id}.npy'
        if feat_path.exists():
            n_skipped += 1
            continue

        mkv_path = Path(srow['video_path'])
        if not mkv_path.exists():
            n_missing_video += 1
            continue
        wav_path = audio_root / f'{stim_id}.wav'
        try:
            extract_wav(mkv_path, wav_path)
        except subprocess.CalledProcessError as e:
            print(f'  ffmpeg fail on {stim_id}: {e}', flush=True)
            continue

        try:
            wav, sr = sf.read(str(wav_path), dtype='float32')
        except Exception as e:
            print(f'  soundfile fail on {stim_id}: {e}', flush=True)
            continue
        if wav.ndim > 1:
            wav = wav.mean(axis=1)
        if sr != SR:
            print(f'  WARNING: {stim_id} sr={sr} != {SR}, ffmpeg should have'
                  ' resampled', flush=True)
            continue

        # Process in chunks to fit GPU mem; collect per-frame features.
        chunk_samples = int(args.chunk_sec * SR)
        feats_per_frame = []
        with torch.no_grad():
            for start in range(0, len(wav), chunk_samples):
                chunk = wav[start:start + chunk_samples]
                if len(chunk) < SR:  # <1 sec, skip (model needs context)
                    continue
                inputs = fe(chunk, sampling_rate=SR, return_tensors='pt')
                input_values = inputs['input_values'].to(device)
                out = model(input_values).last_hidden_state  # (1, T_frames, 768)
                feats_per_frame.append(out[0].cpu().numpy())
        if not feats_per_frame:
            print(f'  {stim_id}: no audio chunks (<1 sec)', flush=True)
            continue
        feats = np.concatenate(feats_per_frame, axis=0)  # (T_total, 768)

        # Aggregate to per-TR. frame index → TR: floor(frame_idx / FRAME_RATE_HZ / TR_SEC).
        # Use the assembly's n_TRs as ground truth and mean-pool frames into bins.
        frames_per_tr = FRAME_RATE_HZ * TR_SEC  # ~74.5
        per_tr = np.zeros((n_TRs, HIDDEN), dtype=np.float32)
        for tr_i in range(n_TRs):
            f0 = int(round(tr_i * frames_per_tr))
            f1 = int(round((tr_i + 1) * frames_per_tr))
            f1 = min(f1, len(feats))
            if f0 >= len(feats):
                # Audio shorter than fMRI run — repeat last bin's mean
                if tr_i > 0:
                    per_tr[tr_i] = per_tr[tr_i - 1]
                continue
            f0 = min(f0, len(feats) - 1)
            per_tr[tr_i] = feats[f0:f1].mean(axis=0)

        np.save(feat_path, per_tr)
        n_done += 1
        if n_done % 10 == 0 or ridx == len(df) - 1:
            log(f'  done={n_done} skipped={n_skipped} '
                f'missing_video={n_missing_video} '
                f'(latest: {stim_id} → {feats.shape} → {per_tr.shape})', t0)

    log(f'FINAL: done={n_done}, skipped(cached)={n_skipped}, '
        f'missing_video={n_missing_video}', t0)


if __name__ == '__main__':
    main()
