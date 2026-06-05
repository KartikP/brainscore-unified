"""Within-Qwen fusion ablation: per-TR Qwen3-Omni thinker hidden states at ALL
49 layers (embeddings + 48 LM layers), mean-pooled over tokens.

Layer 0 = the modality tower streams BEFORE any LM cross-attention (fusion OFF);
later layers = progressively cross-modally fused (fusion ON). Scoring each layer
isolates the contribution of fusion within the SAME backbone, same readout dim,
same pooling — no backbone-identity confound.

Saves (n_TRs, 49, 2048) float16 .npy keyed by stim_id. bf16, 4-GPU. gemma4 env.
"""
import argparse, subprocess, time
from pathlib import Path
from collections import Counter
import numpy as np

TR_SEC = 1.49


def log(m, t0): print(f'[{time.time()-t0:7.1f}s] {m}', flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stim_csv', default='/home/ubuntu/.brainio/algonauts2025/algonauts2025_stim_friends.csv')
    ap.add_argument('--features_root', default='/home/ubuntu/.brainio/algonauts2025/qwen_alllayers')
    ap.add_argument('--assembly_path', default='/home/ubuntu/.brainio/algonauts2025/algonauts2025_friends_sub01.nc')
    ap.add_argument('--clip_list', default='/home/ubuntu/clip_list.txt')
    ap.add_argument('--clip_cache', default='/opt/dlami/nvme/qwen_win2')
    ap.add_argument('--model', default='Qwen/Qwen3-Omni-30B-A3B-Thinking')
    ap.add_argument('--window_sec', type=float, default=3.0)
    args = ap.parse_args()
    t0 = time.time()

    import torch
    from transformers import Qwen3OmniMoeForConditionalGeneration, Qwen3OmniMoeProcessor
    from qwen_omni_utils import process_mm_info
    import xarray as xr

    da = xr.open_dataarray(args.assembly_path)
    stim_to_n = dict(Counter(list(da['stimulus_id'].values)))
    subset = set(l.strip() for l in open(args.clip_list) if l.strip()) if args.clip_list else None
    log(f'{len(stim_to_n)} clips; subset={len(subset) if subset else "ALL"}', t0)

    proc = Qwen3OmniMoeProcessor.from_pretrained(args.model)
    model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
        args.model, device_map='auto', dtype=torch.bfloat16,
        attn_implementation='sdpa', low_cpu_mem_usage=True).eval()
    H = model.config.thinker_config.text_config.hidden_size
    nL = model.config.thinker_config.text_config.num_hidden_layers + 1   # +embeddings
    log(f'model loaded (bf16, sharded); H={H}, n_hidden_states={nL}', t0)

    import pandas as pd
    df = pd.read_csv(args.stim_csv)
    feat_root = Path(args.features_root); feat_root.mkdir(parents=True, exist_ok=True)
    win_dir = Path(args.clip_cache); win_dir.mkdir(parents=True, exist_ok=True)
    n_done = n_skip = 0
    clips = [s for s in dict.fromkeys(list(da['stimulus_id'].values)) if (subset is None or s in subset)]
    for stim in clips:
        srow = df[df['stimulus_id'] == stim]
        if srow.empty: continue
        srow = srow.iloc[0]
        n_TRs = stim_to_n[stim]
        fp = feat_root / f'{stim}.npy'
        if fp.exists(): n_skip += 1; continue
        mkv = Path(srow['video_path'])
        if not mkv.exists(): continue
        text_per_tr = ['']*n_TRs
        tp = srow.get('transcript_path', '')
        if isinstance(tp, str) and Path(tp).exists():
            tsv = pd.read_csv(tp, sep='\t', keep_default_na=False)
            tt = list(tsv['text_per_tr']); text_per_tr = (tt[:n_TRs] + ['']*max(0, n_TRs-len(tt)))

        per = np.zeros((n_TRs, nL, H), np.float16)
        with torch.no_grad():
            for tr in range(n_TRs):
                center = tr*TR_SEC + TR_SEC/2; start = max(0.0, center - args.window_sec/2)
                clip = win_dir / f'{stim}_{tr}.mp4'
                subprocess.run(['ffmpeg','-y','-loglevel','error','-ss',f'{start:.2f}',
                                '-t',f'{args.window_sec:.2f}','-i',str(mkv),'-r','2',
                                '-vf','scale=224:224','-c:a','aac',str(clip)], capture_output=True)
                if not clip.exists() or clip.stat().st_size < 1024:
                    per[tr] = per[tr-1] if tr>0 else per[tr]; continue
                content = [{'type':'video','video':str(clip)}]
                txt = text_per_tr[tr].strip()
                if txt: content.append({'type':'text','text':txt})
                convo = [{'role':'user','content':content}]
                tx = proc.apply_chat_template(convo, add_generation_prompt=True, tokenize=False)
                audios, images, videos = process_mm_info(convo, use_audio_in_video=True)
                inp = proc(text=tx, audio=audios, images=images, videos=videos,
                           return_tensors='pt', padding=True, use_audio_in_video=True)
                inp = {k:(v.to('cuda:0') if hasattr(v,'to') else v) for k,v in inp.items()}
                inp = {k:(v.to(torch.bfloat16) if (hasattr(v,'is_floating_point') and v.is_floating_point()) else v)
                       for k,v in inp.items()}
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    out = model.thinker(**inp, output_hidden_states=True, return_dict=True,
                                        use_audio_in_video=True)
                # (49, H): mean over tokens at every hidden state
                per[tr] = np.stack([h[0].float().mean(0).cpu().numpy() for h in out.hidden_states]).astype(np.float16)
                clip.unlink(missing_ok=True)
        np.save(fp, per); n_done += 1
        log(f'done={n_done} skip={n_skip} ({stim} -> {per.shape})', t0)
    log(f'FINAL done={n_done} skip={n_skip}', t0)


if __name__ == '__main__':
    main()
