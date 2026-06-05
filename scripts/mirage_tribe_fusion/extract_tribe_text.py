"""TRIBEv2 text tower: per-TR Llama-3.2-3B features for Algonauts clips.

Adapted from the Phase-3b MiniLM extractor. Instead of a sentence-transformer,
runs Llama-3.2-3B (causal LM, gated — needs HF auth) with output_hidden_states
and takes the LAST-token hidden state at a chosen layer for each windowed text
snippet (causal-LM convention: the final token has attended to the whole window).

Saves (n_TRs, 3072) float32 .npy keyed by stim_id. Subset via --clip_list.
Run in the tribe_test env (HF_TOKEN set) on GPU.
"""
import argparse, time
from pathlib import Path
from collections import Counter
import numpy as np
import pandas as pd

TR_SEC = 1.49


def log(m, t0): print(f'[{time.time()-t0:7.1f}s] {m}', flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stim_csv', default='/home/ubuntu/.brainio/algonauts2025/algonauts2025_stim_friends.csv')
    ap.add_argument('--features_root', default='/home/ubuntu/.brainio/algonauts2025/tribe_text_llama3b')
    ap.add_argument('--assembly_path', default='/home/ubuntu/.brainio/algonauts2025/algonauts2025_friends_sub01.nc')
    ap.add_argument('--clip_list', default='')
    ap.add_argument('--model', default='meta-llama/Llama-3.2-3B')
    ap.add_argument('--layer', type=int, default=14, help='hidden-state layer to read (0..28)')
    ap.add_argument('--context_TRs', type=int, default=5)
    args = ap.parse_args()
    t0 = time.time()

    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import xarray as xr

    da = xr.open_dataarray(args.assembly_path)
    stim_to_n = dict(Counter(list(da['stimulus_id'].values)))
    subset = set(l.strip() for l in open(args.clip_list) if l.strip()) if args.clip_list else None
    log(f'{len(stim_to_n)} clips; subset={len(subset) if subset else "ALL"}', t0)

    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float16,
                                                 output_hidden_states=True).eval().to(dev)
    HID = model.config.hidden_size
    log(f'llama hidden={HID}, layers={model.config.num_hidden_layers}, reading layer {args.layer}', t0)

    df = pd.read_csv(args.stim_csv)
    feat_root = Path(args.features_root); feat_root.mkdir(parents=True, exist_ok=True)
    n_done = n_skip = 0
    for _, srow in df.iterrows():
        sid = srow['stimulus_id']; n_TRs = stim_to_n.get(sid)
        if n_TRs is None or (subset and sid not in subset): continue
        fp = feat_root / f'{sid}.npy'
        if fp.exists(): n_skip += 1; continue
        tsv_path = Path(srow['transcript_path'])
        if not tsv_path.exists(): continue
        tsv = pd.read_csv(tsv_path, sep='\t', keep_default_na=False)
        tpt = list(tsv['text_per_tr'])
        tpt = tpt[:n_TRs] + ['']*max(0, n_TRs-len(tpt))
        windows = []
        for t in range(n_TRs):
            s = max(0, t-args.context_TRs+1)
            chunk = ' '.join(x for x in tpt[s:t+1] if isinstance(x, str)).strip()
            windows.append(chunk if chunk else ' ')

        per = np.zeros((n_TRs, HID), np.float32)
        B = 16
        with torch.no_grad():
            for b in range(0, n_TRs, B):
                batch = windows[b:b+B]
                enc = tok(batch, return_tensors='pt', padding=True, truncation=True,
                          max_length=128).to(dev)
                out = model(**enc)
                hs = out.hidden_states[args.layer]              # (B, T, HID)
                # last NON-pad token per row (left-to-right; pad is on the right)
                lengths = enc['attention_mask'].sum(1) - 1      # index of last real token
                idx = lengths.clamp(min=0)
                last = hs[torch.arange(hs.size(0)), idx]        # (B, HID)
                per[b:b+len(batch)] = last.float().cpu().numpy()
        np.save(fp, per); n_done += 1
        if n_done % 10 == 0: log(f'done={n_done} skip={n_skip} ({sid}->{per.shape})', t0)
    log(f'FINAL done={n_done} skip={n_skip}', t0)


if __name__ == '__main__':
    main()
