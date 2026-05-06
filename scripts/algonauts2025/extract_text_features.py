"""Extract per-TR sentence-transformer features for every Friends/Movie10 clip.

For each clip:
1. Load .tsv (text_per_tr column, one row per TR)
2. Replace blank rows with empty string (model handles)
3. Embed each TR's text with sentence-transformers/all-MiniLM-L6-v2 (384-dim)
4. Save (n_TRs, 384) float32 .npy keyed by clip_id

Run on EC2. Fast — ~5 min for all clips with batch encoding.
"""
import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

TR_SEC = 1.49


def log(msg, t0):
    print(f'[{time.time() - t0:7.1f}s] {msg}', flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stim_csv',
                        default='/home/ubuntu/.brainio/algonauts2025/algonauts2025_stim_friends.csv')
    parser.add_argument('--features_root',
                        default='/home/ubuntu/.brainio/algonauts2025/text_features_minilm')
    parser.add_argument('--assembly_path',
                        default='/home/ubuntu/.brainio/algonauts2025/algonauts2025_friends_sub01.nc')
    parser.add_argument('--model',
                        default='sentence-transformers/all-MiniLM-L6-v2')
    parser.add_argument('--context_TRs', type=int, default=5,
                        help='Concatenate this many TRs of text for each window (right-aligned).')
    args = parser.parse_args()

    t0 = time.time()
    log('imports...', t0)
    import torch
    from sentence_transformers import SentenceTransformer
    import xarray as xr

    log('load assembly to get n_TRs per clip...', t0)
    da = xr.open_dataarray(args.assembly_path)
    sids = list(da['stimulus_id'].values)
    from collections import Counter
    stim_to_n_TRs = dict(Counter(sids))
    log(f'  {len(stim_to_n_TRs)} clips in assembly', t0)

    log(f'load text model {args.model}...', t0)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = SentenceTransformer(args.model, device=device)
    HIDDEN = model.get_sentence_embedding_dimension()  # 384
    log(f'  hidden={HIDDEN}', t0)

    df = pd.read_csv(args.stim_csv)
    feat_root = Path(args.features_root)
    feat_root.mkdir(parents=True, exist_ok=True)

    n_done, n_skipped, n_missing_tsv = 0, 0, 0
    for ridx, srow in df.iterrows():
        stim_id = srow['stimulus_id']
        n_TRs = stim_to_n_TRs.get(stim_id)
        if n_TRs is None:
            continue

        feat_path = feat_root / f'{stim_id}.npy'
        if feat_path.exists():
            n_skipped += 1
            continue

        tsv_path = Path(srow['transcript_path'])
        if not tsv_path.exists():
            n_missing_tsv += 1
            continue
        tsv = pd.read_csv(tsv_path, sep='\t', keep_default_na=False)
        # Each row of the tsv corresponds to one TR.
        text_per_tr = list(tsv['text_per_tr'])
        # Pad/truncate to assembly's n_TRs (very rare drift).
        if len(text_per_tr) < n_TRs:
            text_per_tr += [''] * (n_TRs - len(text_per_tr))
        else:
            text_per_tr = text_per_tr[:n_TRs]

        # For better signal, accumulate the last `context_TRs` of text
        # into each window (sentence-transformer benefits from longer
        # input than a 1-2 word per-TR snippet).
        windowed = []
        for t in range(n_TRs):
            start = max(0, t - args.context_TRs + 1)
            chunk = ' '.join([s for s in text_per_tr[start:t + 1]
                              if isinstance(s, str)]).strip()
            windowed.append(chunk if chunk else ' ')

        embs = model.encode(windowed, batch_size=128,
                            convert_to_numpy=True,
                            show_progress_bar=False).astype(np.float32)
        np.save(feat_path, embs)
        n_done += 1
        if n_done % 25 == 0 or ridx == len(df) - 1:
            log(f'  done={n_done} skipped={n_skipped} '
                f'missing_tsv={n_missing_tsv} '
                f'(latest: {stim_id} → {embs.shape})', t0)

    log(f'FINAL: done={n_done}, skipped(cached)={n_skipped}, '
        f'missing_tsv={n_missing_tsv}', t0)


if __name__ == '__main__':
    main()
