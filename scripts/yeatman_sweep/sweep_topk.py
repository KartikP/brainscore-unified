"""Sweep TOP_K to find the regime where selectivity matters.

For each K in a range, run:
  - canonical lesion (top-K positive Cohen's d)
  - random control (random K units)
  - inverse-sign lesion (top-K negative Cohen's d)
across the 5 late MLP layers, score via benchmark(bs_model), reset.

Output: /tmp/sweep_results.json with raw/real/pseudo accuracies per
condition per K.

Runtime: ~90 min on Mac MPS.
"""
import json
import random
import time
import numpy as np
import torch
from PIL import Image

import brainscore
import brainscore_vision  # registers Qwen
from brainscore_core.model_interface import StateChange, Selection, Perturbation
from brainscore.perturbation import build_pytorch_ablation_fn


def resolve(model, path):
    for p in path.split('.'):
        model = model[int(p)] if p.isdigit() else getattr(model, p)
    return model


def main():
    t0 = time.time()
    log = lambda msg: print(f'[{time.time()-t0:6.1f}s] {msg}', flush=True)

    log('loading benchmark + model')
    benchmark = brainscore.load_benchmark('Yeatman2021-lexical_decision-image')
    bs_model = brainscore.load_model('qwen2.5-vl-3b')
    bs_model._state_change_fn = build_pytorch_ablation_fn(bs_model._model)

    device = ('mps' if torch.backends.mps.is_available()
              else 'cuda' if torch.cuda.is_available() else 'cpu')
    qwen = bs_model._model.to(device).eval()
    processor = bs_model._preprocessors['vision']._processor

    log(f'model on {device}')

    # ---- Localize once ----
    log('subsampling localizer (50 real + 50 pseudo)')
    train_stim = benchmark._train_stimuli
    real_paths = train_stim[train_stim['image_label'] == 'real']['image_file_name'].tolist()
    pseudo_paths = train_stim[train_stim['image_label'] == 'pseudo']['image_file_name'].tolist()
    random.seed(0)
    loc_real = [Image.open(p).convert('RGB') for p in random.sample(real_paths, 50)]
    loc_pseudo = [Image.open(p).convert('RGB') for p in random.sample(pseudo_paths, 50)]

    LAYERS = [f'model.language_model.layers.{i}.mlp' for i in (26, 28, 30, 32, 34)]
    PROMPT = 'What word is shown in this image?'

    log('localizing (multi-hook)')
    captured = {lp: [] for lp in LAYERS}
    handles = []
    for lp in LAYERS:
        layer = resolve(qwen, lp)
        def hook(_m, _i, output, _lp=lp):
            h = output[0] if isinstance(output, tuple) else output
            captured[_lp].append(h.detach().to('cpu', dtype=torch.float32)
                                  .mean(dim=1).squeeze(0).numpy())
        handles.append(layer.register_forward_hook(hook))
    try:
        for img in loc_real + loc_pseudo:
            msg = [{'role': 'user', 'content': [
                {'type': 'image', 'image': img}, {'type': 'text', 'text': PROMPT}]}]
            text = processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
            inputs = processor(text=[text], images=[img], return_tensors='pt').to(device)
            with torch.no_grad():
                qwen.generate(**inputs, max_new_tokens=1, do_sample=False)
    finally:
        for h in handles:
            h.remove()

    selectivity_by_layer = {}
    for lp in LAYERS:
        acts = np.stack(captured[lp])
        real_act, pseudo_act = acts[:50], acts[50:]
        pooled = np.sqrt((real_act.var(0) + pseudo_act.var(0)) / 2 + 1e-6)
        selectivity_by_layer[lp] = (real_act.mean(0) - pseudo_act.mean(0)) / pooled

    n_units = selectivity_by_layer[LAYERS[0]].shape[0]
    log(f'localizer done; n_units per layer = {n_units}')

    # ---- Helper: install lesion + score + reset ----
    def lesion_score(per_layer_units, label):
        for L, u in per_layer_units.items():
            bs_model.process(StateChange(
                kind='ablation',
                target=Selection(layer=L, indices=u),
                perturbation=Perturbation(kind='zero'),
            ))
        s = benchmark(bs_model)
        bs_model.reset()
        result = {
            'raw': float(s.attrs['raw']),
            'real': s.attrs['accuracy_real'],
            'pseudo': s.attrs['accuracy_pseudo'],
        }
        log(f'  {label}: raw={result["raw"]:.3f}  '
            f'real={result["real"]:.3f}  pseudo={result["pseudo"]:.3f}')
        return result

    # ---- Baseline ----
    log('scoring baseline')
    s = benchmark(bs_model)
    baseline = {
        'raw': float(s.attrs['raw']),
        'real': s.attrs['accuracy_real'],
        'pseudo': s.attrs['accuracy_pseudo'],
    }
    log(f'baseline: raw={baseline["raw"]:.3f}  '
        f'real={baseline["real"]:.3f}  pseudo={baseline["pseudo"]:.3f}')

    # ---- Sweep ----
    K_VALUES = [500, 750, 1000, 1250]
    results = {'baseline': baseline, 'sweep': {}}

    for k in K_VALUES:
        log(f'TOP_K = {k}')
        # canonical (top-K real-selective)
        canonical_units = {L: np.argsort(selectivity_by_layer[L])[-k:].tolist()
                           for L in LAYERS}
        canonical = lesion_score(canonical_units, f'canonical k={k}')

        # random
        random.seed(0)
        random_units = {L: random.sample(range(n_units), k) for L in LAYERS}
        rnd = lesion_score(random_units, f'random    k={k}')

        # inverse (top-K pseudo-selective = bottom-K of selectivity)
        inverse_units = {L: np.argsort(selectivity_by_layer[L])[:k].tolist()
                         for L in LAYERS}
        inverse = lesion_score(inverse_units, f'inverse   k={k}')

        results['sweep'][k] = {
            'canonical': canonical,
            'random': rnd,
            'inverse': inverse,
            'gap_canonical_vs_random': rnd['raw'] - canonical['raw'],
            'gap_canonical_vs_inverse': inverse['raw'] - canonical['raw'],
        }

        # Save incrementally so partial results survive crashes
        with open('/tmp/sweep_results.json', 'w') as f:
            json.dump(results, f, indent=2)

    log('DONE')
    log('--- summary ---')
    for k, r in results['sweep'].items():
        log(f'k={k}: canonical={r["canonical"]["raw"]:.3f} '
            f'random={r["random"]["raw"]:.3f} '
            f'inverse={r["inverse"]["raw"]:.3f} '
            f'gap_can_vs_rand={r["gap_canonical_vs_random"]:+.3f} '
            f'gap_can_vs_inv={r["gap_canonical_vs_inverse"]:+.3f}')


if __name__ == '__main__':
    main()
