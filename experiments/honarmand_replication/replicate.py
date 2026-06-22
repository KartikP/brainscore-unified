"""Faithful replication of Honarmand et al. (2026, ICLR) "Inducing Dyslexia in
Vision Language Models" on Qwen2.5-VL-3B — fixing the localizer + scoring path
so our random control behaves like theirs.

What our earlier sweep got wrong (and this fixes):
  * LOCALIZER. The earlier sweep contrasted real-word vs PSEUDO-word images,
    which finds real/pseudo *discriminator* units, not word-form units. Honarmand
    contrasts word images vs NON-word images (faces/objects/scrambled — the
    Saygin VWFA localizer). We use word vs scrambled-word images here, so we find
    visual-word-form (VWF)-selective units, the canonical-dyslexia substrate.
  * SCORING PATH. The benchmark's readout refits a logistic classifier on the
    (post-ablation) features, which can *compensate* for the lesion and mask it.
    Honarmand scores the model's own generated answer. We score by GENERATION
    here (the model answers "real or pseudo?"), so the lesion shows directly.
  * SITE. We ablate the MLP gate_proj across ALL language-decoder blocks (their
    site), not 5 late MLP layers — so a random lesion of the same size causes the
    non-selective ~general degradation they report, instead of being inert.

Outputs the dissociation: VWF-selective ablation -> selective READING deficit
(ROAR drops below the 65% dyslexia threshold) with the non-reading control
spared; random ablation of equal size -> smaller / non-selective effect.

Run on EC2 (GPU). Writes JSON to --out.
"""
import argparse
import json
import re
import time

import numpy as np
import torch
from PIL import Image


def log(msg, t0=[None]):
    if t0[0] is None:
        t0[0] = time.time()
    print(f'[{time.time() - t0[0]:6.1f}s] {msg}', flush=True)


def scramble_image(img: Image.Image, grid: int = 8, seed: int = 0) -> Image.Image:
    """Patch-scramble: shuffle a grid of patches. Destroys letter forms while
    preserving low-level luminance/colour statistics — the Saygin scrambled-word
    control."""
    rng = np.random.RandomState(seed)
    a = np.asarray(img.convert('RGB'))
    H, W, _ = a.shape
    ph, pw = H // grid, W // grid
    patches = [a[r*ph:(r+1)*ph, c*pw:(c+1)*pw] for r in range(grid) for c in range(grid)]
    order = rng.permutation(len(patches))
    out = a.copy()
    for idx, p in enumerate(order):
        r, c = idx // grid, idx % grid
        out[r*ph:(r+1)*ph, c*pw:(c+1)*pw] = patches[p]
    return Image.fromarray(out)


def make_control_items():
    """A tiny non-reading visual task: count coloured shapes. Reading-selective
    ablation should leave this intact; general damage should hurt it."""
    from PIL import ImageDraw
    items = []
    rng = np.random.RandomState(0)
    colors = [(220, 60, 50), (50, 110, 220), (40, 180, 80)]
    for i in range(12):
        n = rng.randint(1, 5)
        img = Image.new('RGB', (300, 300), (245, 245, 245))
        d = ImageDraw.Draw(img)
        for _ in range(n):
            x, y = rng.randint(20, 230), rng.randint(20, 230)
            col = colors[rng.randint(len(colors))]
            d.ellipse([x, y, x + 45, y + 45], fill=col)
        items.append((img, str(n)))
    return items


class QwenGen:
    """Thin generation helper over the (possibly ablated) Qwen model."""

    def __init__(self, model, processor, device):
        self.model, self.processor, self.device = model, processor, device

    def ask(self, img, prompt, max_new_tokens=6):
        msg = [{'role': 'user', 'content': [
            {'type': 'image', 'image': img}, {'type': 'text', 'text': prompt}]}]
        text = self.processor.apply_chat_template(msg, tokenize=False,
                                                  add_generation_prompt=True)
        inputs = self.processor(text=[text], images=[img],
                                return_tensors='pt').to(self.device)
        with torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=max_new_tokens,
                                      do_sample=False)
        gen = out[0][inputs['input_ids'].shape[1]:]
        return self.processor.decode(gen, skip_special_tokens=True).strip().lower()


ROAR_PROMPT = ("Is the letter string in this image a real English word or a "
               "made-up pseudo word? Answer with exactly one word: real or pseudo.")


def score_roar_generation(gen, test_items):
    """test_items: list of (img, label in {'real','pseudo'}). Returns dict."""
    correct, by = [], {'real': [], 'pseudo': []}
    for img, label in test_items:
        ans = gen.ask(img, ROAR_PROMPT)
        m = re.search(r'\b(real|pseudo)\b', ans)
        pred = m.group(1) if m else ('pseudo' if 'pseudo' in ans else 'real')
        ok = (pred == label)
        correct.append(ok); by[label].append(ok)
    return {
        'raw': float(np.mean(correct)),
        'real': float(np.mean(by['real'])) if by['real'] else None,
        'pseudo': float(np.mean(by['pseudo'])) if by['pseudo'] else None,
    }


def score_control(gen, control_items):
    correct = []
    for img, ans_true in control_items:
        ans = gen.ask(img, "How many shapes are in this image? Answer with just a number.")
        m = re.search(r'\d+', ans)
        correct.append(bool(m and m.group(0) == ans_true))
    return float(np.mean(correct))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n_localizer', type=int, default=60)
    ap.add_argument('--n_test', type=int, default=50, help='per class (real/pseudo)')
    ap.add_argument('--mask_sizes', default='0.01,0.03,0.0689,0.10,0.15')
    ap.add_argument('--seed', type=int, default=0,
                    help='controls localizer word subsample, scramble, and random '
                         'ablation — vary it to average over seeds like Honarmand.')
    ap.add_argument('--out', default='/tmp/honarmand_replication.json')
    args = ap.parse_args()
    SEED = args.seed

    import brainscore
    import brainscore_vision  # noqa: registers qwen
    from brainscore_core.model_interface import StateChange, Selection, Perturbation
    from brainscore.perturbation import build_pytorch_ablation_fn

    log('loading benchmark + model')
    benchmark = brainscore.load_benchmark('Yeatman2021-lexical_decision-image')
    bs = brainscore.load_model('qwen2.5-vl-3b')
    bs._state_change_fn = build_pytorch_ablation_fn(bs._model)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    qwen = bs._model.to(device).eval()
    processor = bs._preprocessors['vision']._processor
    gen = QwenGen(qwen, processor, device)
    log(f'model on {device}')

    # ---- stimuli ----
    train = benchmark._train_stimuli
    test = benchmark._test_stimuli
    real_train = train[train['image_label'] == 'real']['image_file_name'].tolist()
    rng = np.random.RandomState(SEED)
    word_paths = list(rng.choice(real_train, size=min(args.n_localizer, len(real_train)), replace=False))
    word_imgs = [Image.open(p).convert('RGB') for p in word_paths]
    nonword_imgs = [scramble_image(im, seed=SEED * 1000 + i) for i, im in enumerate(word_imgs)]
    log(f'localizer: {len(word_imgs)} word vs {len(nonword_imgs)} scrambled')

    test_real = test[test['image_label'] == 'real']['image_file_name'].tolist()[:args.n_test]
    test_pseudo = test[test['image_label'] == 'pseudo']['image_file_name'].tolist()[:args.n_test]
    test_items = ([(Image.open(p).convert('RGB'), 'real') for p in test_real]
                  + [(Image.open(p).convert('RGB'), 'pseudo') for p in test_pseudo])
    control_items = make_control_items()

    # ---- locate gate_proj layers (Honarmand's site) ----
    gate_layers = [n for n, _ in qwen.named_modules()
                   if re.search(r'language_model\.layers\.\d+\.mlp\.gate_proj$', n)]
    log(f'{len(gate_layers)} gate_proj layers (Honarmand MLP site)')

    # ---- localize: t-stat per unit per layer (word vs scrambled) ----
    captured = {L: [] for L in gate_layers}
    handles = []
    for L in gate_layers:
        mod = dict(qwen.named_modules())[L]
        def hook(_m, _i, output, _L=L):
            h = output[0] if isinstance(output, tuple) else output
            captured[_L].append(h.detach().to('cpu', torch.float32).mean(dim=1).squeeze(0).numpy())
        handles.append(mod.register_forward_hook(hook))
    try:
        for img in word_imgs + nonword_imgs:
            gen.ask(img, 'What is in this image?', max_new_tokens=1)
    finally:
        for h in handles:
            h.remove()

    nw = len(word_imgs)
    tstat_by_layer = {}
    for L in gate_layers:
        acts = np.stack(captured[L])
        w, s = acts[:nw], acts[nw:]
        se = np.sqrt(w.var(0)/nw + s.var(0)/len(s) + 1e-8)
        tstat_by_layer[L] = (w.mean(0) - s.mean(0)) / se
    units_per_layer = tstat_by_layer[gate_layers[0]].shape[0]
    total_units = units_per_layer * len(gate_layers)
    log(f'localized: {units_per_layer} units/layer x {len(gate_layers)} = {total_units} total')

    # flat (layer_idx, unit) ranking by descending t-stat (VWF-selective = high)
    flat_t = np.concatenate([tstat_by_layer[L] for L in gate_layers])
    order_desc = np.argsort(flat_t)[::-1]   # most word-selective first

    def to_per_layer(flat_indices):
        per = {}
        for fi in flat_indices:
            li, ui = divmod(int(fi), units_per_layer)
            per.setdefault(gate_layers[li], []).append(ui)
        return per

    def ablate_score(per_layer, label):
        for L, us in per_layer.items():
            bs.process(StateChange(kind='ablation',
                                   target=Selection(layer=L, indices=us),
                                   perturbation=Perturbation(kind='zero')))
        roar = score_roar_generation(gen, test_items)
        ctrl = score_control(gen, control_items)
        bs.reset()
        log(f'  {label}: ROAR raw={roar["raw"]:.3f} (real={roar["real"]:.2f} '
            f'pseudo={roar["pseudo"]:.2f})  control={ctrl:.2f}')
        return {'roar': roar, 'control': ctrl}

    results = {'total_units': int(total_units), 'units_per_layer': int(units_per_layer),
               'n_gate_layers': len(gate_layers), 'dyslexia_threshold': 0.65,
               'conditions': {}}

    log('baseline (no ablation)')
    results['baseline'] = ablate_score({}, 'baseline')

    rng2 = np.random.RandomState(SEED)
    for ms in [float(x) for x in args.mask_sizes.split(',')]:
        k = int(round(ms * total_units))
        log(f'mask_size={ms:.4f} -> {k} units')
        vwf = to_per_layer(order_desc[:k])
        rnd = to_per_layer(rng2.choice(total_units, size=k, replace=False))
        results['conditions'][f'{ms}'] = {
            'mask_size': ms, 'k': k,
            'vwf_selective': ablate_score(vwf, f'VWF-selective ms={ms}'),
            'random': ablate_score(rnd, f'random        ms={ms}'),
        }
        with open(args.out, 'w') as f:
            json.dump(results, f, indent=2)

    # summary
    log('=== DISSOCIATION SUMMARY ===')
    b = results['baseline']
    log(f'baseline: ROAR={b["roar"]["raw"]:.3f} control={b["control"]:.2f}')
    for ms, c in results['conditions'].items():
        v, r = c['vwf_selective'], c['random']
        log(f'ms={ms}: VWF ROAR={v["roar"]["raw"]:.3f}/ctrl={v["control"]:.2f}  '
            f'| RANDOM ROAR={r["roar"]["raw"]:.3f}/ctrl={r["control"]:.2f}')
    with open(args.out, 'w') as f:
        json.dump(results, f, indent=2)
    log(f'written {args.out}')


if __name__ == '__main__':
    main()
