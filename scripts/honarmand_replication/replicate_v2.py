"""Faithful Honarmand et al. (2026) dyslexia-induction replication — v2.

Improvements over v1 (which found no selective deficit on 3B):
  * RICHER LOCALIZER. Word images vs a multi-category non-word set (scrambled
    words + line-drawing objects), closer to the Saygin VWFA localizer, so the
    selected units are word-form-specific rather than generic edge detectors.
  * BIGGER MODEL. --model_id defaults to Qwen2.5-VL-7B (escalate to 72B on a
    multi-GPU instance); the v1 finding suggested the effect is scale-dependent.
  * SELF-CONTAINED ABLATION. Direct forward hooks that zero selected gate_proj
    units — works on any HF model size without the brainscore wrapper.
  * MULTI-SEED averaging with mean +/- SD, like the paper's 20-seed protocol.

Scores ROAR by GENERATION (no readout refitting) + a non-reading control
(shape counting). Looks for the dissociation: VWF-selective ablation -> ROAR
crosses the 0.65 dyslexia threshold while the control is spared; random
ablation -> non-selective.

Run on EC2 (GPU). Writes JSON to --out.
"""
import argparse
import json
import os
import re
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw


def log(msg, t0=[None]):
    if t0[0] is None:
        t0[0] = time.time()
    print(f'[{time.time() - t0[0]:7.1f}s] {msg}', flush=True)


def scramble_image(img, grid=8, seed=0):
    rng = np.random.RandomState(seed)
    a = np.asarray(img.convert('RGB')); H, W, _ = a.shape
    ph, pw = H // grid, W // grid
    patches = [a[r*ph:(r+1)*ph, c*pw:(c+1)*pw] for r in range(grid) for c in range(grid)]
    order = rng.permutation(len(patches)); out = a.copy()
    for idx, p in enumerate(order):
        r, c = idx // grid, idx % grid
        out[r*ph:(r+1)*ph, c*pw:(c+1)*pw] = patches[p]
    return Image.fromarray(out)


def object_image(seed, size=(300, 120)):
    """Crude line-drawing 'object' — outlined shapes on white, no letter forms."""
    rng = np.random.RandomState(seed)
    img = Image.new('RGB', size, (255, 255, 255)); d = ImageDraw.Draw(img)
    for _ in range(rng.randint(2, 5)):
        x, y = rng.randint(10, size[0]-60), rng.randint(10, size[1]-60)
        w, h = rng.randint(25, 55), rng.randint(25, 55)
        shape = rng.randint(3)
        if shape == 0:
            d.ellipse([x, y, x+w, y+h], outline=(0, 0, 0), width=3)
        elif shape == 1:
            d.rectangle([x, y, x+w, y+h], outline=(0, 0, 0), width=3)
        else:
            d.line([x, y, x+w, y+h], fill=(0, 0, 0), width=3)
            d.line([x, y+h, x+w, y], fill=(0, 0, 0), width=3)
    return img


def make_control_items(n=15):
    rng = np.random.RandomState(123)
    cols = [(220, 60, 50), (50, 110, 220), (40, 180, 80)]
    items = []
    for _ in range(n):
        k = rng.randint(1, 5)
        img = Image.new('RGB', (300, 300), (245, 245, 245)); d = ImageDraw.Draw(img)
        for _ in range(k):
            x, y = rng.randint(20, 230), rng.randint(20, 230)
            d.ellipse([x, y, x+45, y+45], fill=cols[rng.randint(3)])
        items.append((img, str(k)))
    return items


ROAR_PROMPT = ("Is the letter string in this image a real English word or a "
               "made-up pseudo word? Answer with exactly one word: real or pseudo.")


class Gen:
    def __init__(self, model, processor, device):
        self.model, self.processor, self.device = model, processor, device

    def ask(self, img, prompt, max_new_tokens=6):
        msg = [{'role': 'user', 'content': [
            {'type': 'image', 'image': img}, {'type': 'text', 'text': prompt}]}]
        text = self.processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[text], images=[img], return_tensors='pt').to(self.device)
        with torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        gen = out[0][inputs['input_ids'].shape[1]:]
        return self.processor.decode(gen, skip_special_tokens=True).strip().lower()


def score_roar(gen, test_items):
    correct, by = [], {'real': [], 'pseudo': []}
    for img, label in test_items:
        ans = gen.ask(img, ROAR_PROMPT)
        m = re.search(r'\b(real|pseudo)\b', ans)
        pred = m.group(1) if m else ('pseudo' if 'pseudo' in ans else 'real')
        ok = (pred == label); correct.append(ok); by[label].append(ok)
    return {'raw': float(np.mean(correct)),
            'real': float(np.mean(by['real'])) if by['real'] else None,
            'pseudo': float(np.mean(by['pseudo'])) if by['pseudo'] else None}


def score_control(gen, control_items):
    c = []
    for img, ans in control_items:
        out = gen.ask(img, "How many shapes are in this image? Answer with just a number.")
        m = re.search(r'\d+', out); c.append(bool(m and m.group(0) == ans))
    return float(np.mean(c))


def gate_layers_of(model):
    return [n for n, _ in model.named_modules()
            if re.search(r'language_model\.layers\.\d+\.mlp\.gate_proj$', n)]


def localize(gen, model, gate_layers, word_imgs, nonword_imgs):
    """t-stat per gate_proj unit: word vs (pooled) non-word."""
    cap = {L: [] for L in gate_layers}
    mods = dict(model.named_modules())
    handles = []
    for L in gate_layers:
        def hook(_m, _i, o, _L=L):
            h = o[0] if isinstance(o, tuple) else o
            cap[_L].append(h.detach().to('cpu', torch.float32).mean(1).squeeze(0).numpy())
        handles.append(mods[L].register_forward_hook(hook))
    try:
        for img in word_imgs + nonword_imgs:
            gen.ask(img, 'What is in this image?', max_new_tokens=1)
    finally:
        for h in handles:
            h.remove()
    nw = len(word_imgs)
    tmap = {}
    for L in gate_layers:
        acts = np.stack(cap[L]); w, s = acts[:nw], acts[nw:]
        se = np.sqrt(w.var(0)/nw + s.var(0)/len(s) + 1e-8)
        tmap[L] = (w.mean(0) - s.mean(0)) / se
    return tmap


def install_ablation(model, per_layer):
    """Zero selected gate_proj units via forward hooks. Returns handles."""
    mods = dict(model.named_modules())
    handles = []
    for L, idxs in per_layer.items():
        idx = torch.tensor(idxs, dtype=torch.long)
        def hook(_m, _i, o, _idx=idx):
            if isinstance(o, tuple):
                h = o[0]; h[..., _idx.to(h.device)] = 0.0; return (h,) + tuple(o[1:])
            o[..., _idx.to(o.device)] = 0.0; return o
        handles.append(mods[L].register_forward_hook(hook))
    return handles


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model_id', default='Qwen/Qwen2.5-VL-7B-Instruct')
    ap.add_argument('--n_localizer', type=int, default=60)
    ap.add_argument('--n_test', type=int, default=50)
    ap.add_argument('--mask_sizes', default='0.01,0.0689,0.15')
    ap.add_argument('--seeds', type=int, default=3)
    ap.add_argument('--out', default='/tmp/honarmand_v2.json')
    args = ap.parse_args()

    import brainscore
    log(f'loading benchmark stimuli')
    benchmark = brainscore.load_benchmark('Yeatman2021-lexical_decision-image')
    train, test = benchmark._train_stimuli, benchmark._test_stimuli
    real_train = train[train['image_label'] == 'real']['image_file_name'].tolist()
    test_real = test[test['image_label'] == 'real']['image_file_name'].tolist()[:args.n_test]
    test_pseudo = test[test['image_label'] == 'pseudo']['image_file_name'].tolist()[:args.n_test]
    test_items = ([(Image.open(p).convert('RGB'), 'real') for p in test_real]
                  + [(Image.open(p).convert('RGB'), 'pseudo') for p in test_pseudo])
    control_items = make_control_items()

    from transformers import AutoProcessor
    try:
        from transformers import Qwen2_5_VLForConditionalGeneration as VLM
    except Exception:
        from transformers import AutoModelForVision2Seq as VLM
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    log(f'loading {args.model_id}')
    processor = AutoProcessor.from_pretrained(args.model_id)
    model = VLM.from_pretrained(args.model_id, torch_dtype=torch.float16,
                               device_map='auto').eval()
    gen = Gen(model, processor, device)
    gate_layers = gate_layers_of(model)
    log(f'{len(gate_layers)} gate_proj layers')

    masks = [float(x) for x in args.mask_sizes.split(',')]
    acc = {'model_id': args.model_id, 'n_gate_layers': len(gate_layers),
           'threshold': 0.65, 'seeds': args.seeds, 'per_seed': []}

    for seed in range(args.seeds):
        log(f'==== SEED {seed} ====')
        rng = np.random.RandomState(seed)
        word_paths = list(rng.choice(real_train, size=min(args.n_localizer, len(real_train)), replace=False))
        word_imgs = [Image.open(p).convert('RGB') for p in word_paths]
        # multi-category non-word: half scrambled words, half line-drawing objects
        half = len(word_imgs) // 2
        nonword = ([scramble_image(im, seed=seed*1000+i) for i, im in enumerate(word_imgs[:half])]
                   + [object_image(seed*1000+i) for i in range(len(word_imgs) - half)])
        tmap = localize(gen, model, gate_layers, word_imgs, nonword)
        upl = tmap[gate_layers[0]].shape[0]
        total = upl * len(gate_layers)
        flat = np.concatenate([tmap[L] for L in gate_layers])
        order_desc = np.argsort(flat)[::-1]
        log(f'  localized {upl}/layer x {len(gate_layers)} = {total}')

        def to_layers(flat_idx):
            per = {}
            for fi in flat_idx:
                li, ui = divmod(int(fi), upl); per.setdefault(gate_layers[li], []).append(ui)
            return per

        def run(per_layer):
            hs = install_ablation(model, per_layer)
            try:
                r = score_roar(gen, test_items); c = score_control(gen, control_items)
            finally:
                for h in hs:
                    h.remove()
            return {'roar': r, 'control': c}

        srow = {'baseline': run({})}
        log(f"  baseline ROAR={srow['baseline']['roar']['raw']:.3f} ctrl={srow['baseline']['control']:.2f}")
        rng2 = np.random.RandomState(1000 + seed)
        for ms in masks:
            k = int(round(ms * total))
            vwf = to_layers(order_desc[:k])
            rnd = to_layers(rng2.choice(total, size=k, replace=False))
            v, r = run(vwf), run(rnd)
            srow[f'{ms}'] = {'vwf': v, 'random': r, 'k': k}
            log(f"  ms={ms}: VWF ROAR={v['roar']['raw']:.3f}/ctrl={v['control']:.2f} "
                f"| RANDOM ROAR={r['roar']['raw']:.3f}/ctrl={r['control']:.2f}")
        acc['per_seed'].append(srow)
        with open(args.out, 'w') as f:
            json.dump(acc, f, indent=2)

    # aggregate
    def agg(key, sub):
        vals = []
        for s in acc['per_seed']:
            node = s[key] if key == 'baseline' else s[key][sub]
            vals.append(node['roar']['raw'] if 'roar' in node else node)
        return float(np.mean(vals)), float(np.std(vals))
    log('=== AGGREGATE (mean +/- sd) ===')
    bm, bs = agg('baseline', None)
    log(f'baseline ROAR={bm:.3f}+-{bs:.3f}')
    summary = {'baseline_roar': bm}
    for ms in masks:
        vm, vs = agg(f'{ms}', 'vwf'); rm, rs = agg(f'{ms}', 'random')
        summary[f'{ms}'] = {'vwf_roar': vm, 'vwf_sd': vs, 'random_roar': rm, 'random_sd': rs}
        log(f'ms={ms}: VWF={vm:.3f}+-{vs:.3f} | RANDOM={rm:.3f}+-{rs:.3f}')
    acc['summary'] = summary
    with open(args.out, 'w') as f:
        json.dump(acc, f, indent=2)
    log(f'written {args.out}')


if __name__ == '__main__':
    main()
