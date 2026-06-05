"""Smoke test: load Qwen3-Omni-30B-A3B-Thinking (4-bit), feed one real
video+audio+text window, extract the thinker's 48-layer post-fusion LM hidden
states. Reports load time, peak GPU mem, hidden-state shapes, and per-window
forward time -> drives the cost projection for the full extraction.

Run in the gemma4 env with HF_HOME=/opt/dlami/nvme/hf_cache.
"""
import time, sys, subprocess, os, glob, traceback, faulthandler
faulthandler.enable()   # dump C-level stack on segfault/abort (catch silent deaths)
t0 = time.time()
def log(m): print(f'[{time.time()-t0:7.1f}s] {m}', flush=True)

import torch
from transformers import (Qwen3OmniMoeForConditionalGeneration,
                          Qwen3OmniMoeProcessor, BitsAndBytesConfig)

MODEL = 'Qwen/Qwen3-Omni-30B-A3B-Thinking'
CLIP = '/opt/dlami/nvme/smoke_clip.mp4'

# --- cut a 3s clip from a real movie video (video + audio) ---
src = None
for pat in ['/home/ubuntu/algonauts_2025/stimuli/movies/**/*.mkv']:
    hits = glob.glob(pat, recursive=True)
    if hits: src = hits[0]; break
log(f'source video: {src}')
subprocess.run(['ffmpeg','-y','-ss','60','-t','3','-i',src,'-r','2','-vf','scale=224:224',
                '-c:a','aac', CLIP], capture_output=True)
log(f'cut 3s clip -> {CLIP} ({os.path.getsize(CLIP)//1024} KB)')

# --- load model in bf16, sharded across all visible GPUs (4x L40S = 192GB) ---
# No quantization: the 4-bit MoE-expert quantization path hard-crashes at ~48%
# on a single 48GB GPU. 30B bf16 (~60GB) shards cleanly across 4 GPUs.
log('loading processor...')
proc = Qwen3OmniMoeProcessor.from_pretrained(MODEL)
log(f'visible GPUs: {torch.cuda.device_count()}')
log('loading model bf16 (device_map auto, multi-GPU shard)...')
try:
    model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
        MODEL, device_map='auto', dtype=torch.bfloat16,
        attn_implementation='sdpa', low_cpu_mem_usage=True)
except Exception:
    log('LOAD FAILED:\n' + traceback.format_exc()); raise
model.eval()
try:
    from collections import Counter
    dm = Counter(str(v) for v in model.hf_device_map.values())
    log(f'device placement (modules per device): {dict(dm)}')
except Exception as e:
    log(f'device_map introspect skipped: {e}')
log(f'model loaded. per-GPU mem: ' +
    ', '.join(f'{i}:{torch.cuda.max_memory_allocated(i)/1e9:.0f}G'
              for i in range(torch.cuda.device_count())))
log(f'thinker text layers: {model.config.thinker_config.text_config.num_hidden_layers}, '
    f'dim {model.config.thinker_config.text_config.hidden_size}')

# --- build one window's input ---
from qwen_omni_utils import process_mm_info
convo = [{'role':'user','content':[
    {'type':'video','video': CLIP},
    {'type':'text','text':'A scene from a television show.'}]}]
text = proc.apply_chat_template(convo, add_generation_prompt=True, tokenize=False)
audios, images, videos = process_mm_info(convo, use_audio_in_video=True)
inputs = proc(text=text, audio=audios, images=images, videos=videos,
              return_tensors='pt', padding=True, use_audio_in_video=True)
inputs = {k:(v.to('cuda:0') if hasattr(v,'to') else v) for k,v in inputs.items()}
# processor returns float32 pixels/audio; model is bf16 -> cast floats to match
inputs = {k:(v.to(torch.bfloat16) if (hasattr(v,'is_floating_point') and v.is_floating_point()) else v)
          for k,v in inputs.items()}
log(f'input keys: {list(inputs.keys())}')

# --- forward through the thinker, get 48-layer hidden states, timed ---
torch.cuda.reset_peak_memory_stats()
for trial in range(2):  # 2nd trial = warm timing
    t1 = time.time()
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
        out = model.thinker(**inputs, output_hidden_states=True, return_dict=True,
                            use_audio_in_video=True)
    torch.cuda.synchronize()
    dt = time.time()-t1
    hs = out.hidden_states
    log(f'trial {trial}: forward {dt:.2f}s | hidden_states: {len(hs)} tensors, '
        f'each {tuple(hs[-1].shape)} | peak mem {torch.cuda.max_memory_allocated()/1e9:.1f} GB')

# pooled feature per layer (mean over tokens) -> (n_layers, dim)
import numpy as np
pooled = np.stack([h[0].float().mean(0).cpu().numpy() for h in hs])
log(f'pooled per-layer feature: {pooled.shape}  (n_layers+1, dim)')
log(f'DONE. per-window forward ~{dt:.2f}s')
