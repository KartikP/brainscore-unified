"""MIRAGE native arm: per-TR Qwen3-Omni-30B post-fusion thinker features.

For each clip, slides a short window over the TR grid; per TR it feeds that
window's video+audio (+transcript text) through the omni thinker and pools the
text-LM hidden states (mean over tokens) at a chosen layer -> one vector per TR.
This is the "native fusion" feature: video/audio/text interact INSIDE the LM,
unlike the TRIBEv2 post-hoc arm that runs three separate encoders.

bf16, sharded across all visible GPUs (no quantization — the 4-bit MoE load
crashes). Saves (n_TRs, hidden) float32 .npy keyed by stim_id. Subset via
--clip_list. Run in the gemma4 env, HF_HOME=/home/ubuntu/hf_cache, all 4 GPUs.
"""
import argparse, subprocess, time, os
from pathlib import Path
from collections import Counter
import numpy as np
import pandas as pd

TR_SEC = 1.49


def log(m, t0): print(f'[{time.time()-t0:7.1f}s] {m}', flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stim_csv', default='/home/ubuntu/.brainio/algonauts2025/algonauts2025_stim_friends.csv')
    ap.add_argument('--features_root', default='/home/ubuntu/.brainio/algonauts2025/qwen_native_l16')
    ap.add_argument('--assembly_path', default='/home/ubuntu/.brainio/algonauts2025/algonauts2025_friends_sub01.nc')
    ap.add_argument('--clip_list', default='')
    ap.add_argument('--clip_cache', default='/opt/dlami/nvme/qwen_win')  # temp video windows
    ap.add_argument('--model', default='Qwen/Qwen3-Omni-30B-A3B-Thinking')
    ap.add_argument('--layer', type=int, default=24, help='thinker text-LM hidden-state layer to pool (0..48)')
    ap.add_argument('--window_sec', type=float, default=3.0, help='context window fed per TR')
    ap.add_argument('--win_fps', type=int, default=2)
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
    log('model loaded (bf16, sharded)', t0)

    df = pd.read_csv(args.stim_csv)
    feat_root = Path(args.features_root); feat_root.mkdir(parents=True, exist_ok=True)
    win_dir = Path(args.clip_cache); win_dir.mkdir(parents=True, exist_ok=True)
    n_done = n_skip = 0
    for _, srow in df.iterrows():
        sid = srow['stimulus_id']; n_TRs = stim_to_n.get(sid)
        if n_TRs is None or (subset and sid not in subset): continue
        fp = feat_root / f'{sid}.npy'
        if fp.exists(): n_skip += 1; continue
        mkv = Path(srow['video_path'])
        if not mkv.exists(): continue
        # per-TR transcript (optional)
        text_per_tr = ['']*n_TRs
        tp = srow.get('transcript_path', '')
        if isinstance(tp, str) and Path(tp).exists():
            tsv = pd.read_csv(tp, sep='\t', keep_default_na=False)
            tt = list(tsv['text_per_tr']); text_per_tr = (tt[:n_TRs] + ['']*max(0, n_TRs-len(tt)))

        per = np.zeros((n_TRs, model.config.thinker_config.text_config.hidden_size), np.float32)
        with torch.no_grad():
            for tr in range(n_TRs):
                center = tr * TR_SEC + TR_SEC/2
                start = max(0.0, center - args.window_sec/2)
                clip = win_dir / f'{sid}_{tr}.mp4'
                subprocess.run(['ffmpeg','-y','-loglevel','error','-ss',f'{start:.2f}',
                                '-t',f'{args.window_sec:.2f}','-i',str(mkv),
                                '-r',str(args.win_fps),'-vf','scale=224:224','-c:a','aac',str(clip)],
                               capture_output=True)
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
                per[tr] = out.hidden_states[args.layer][0].float().mean(0).cpu().numpy()
                clip.unlink(missing_ok=True)
        np.save(fp, per); n_done += 1
        log(f'done={n_done} skip={n_skip} ({sid} -> {per.shape})', t0)
    log(f'FINAL done={n_done} skip={n_skip}', t0)


if __name__ == '__main__':
    main()
