"""Cross-benchmark test of the layer-mapping hypotheses on MajajHong2015 V4 + IT
(image-only, single-neuron electrophysiology, two SEPARATE narrow regions) — the
direct test of whether "one layer is enough / low-dim" was an artifact of the
broad Lahner fMRI ROI, plus a regression-vs-RSA (geometry) metric comparison.

For CLIP ViT-B/32, per layer (12 encoder layers, mean-pooled to 768 units/layer
to be comparable to the V-JEPA2 run), for EACH region:
  - regression per-layer sweep (RidgeCV) → best layer  [does V4 vs IT differ?]
  - budget-matched curve (within-layer top-K vs random) → one layer enough here?
  - effective dimensionality of brain responses + best-layer features
  - RSA per-layer sweep (RDM correlation) → best layer under GEOMETRY
  - RSA unit-selection: full vs top-256 vs random-256 at the RSA-best layer
    → does "random scores high / selection doesn't matter" survive RSA?

Raw scores (per-neuroid Pearson r for regression; Spearman RDM r for RSA); no
ceiling normalization — the qualitative questions (best layer per region, one-
layer sufficiency, regression vs RSA) don't need it. Run on EC2 (GPU for CLIP).
"""
import json, os, sys, time
sys.path.insert(0, '/home/ubuntu/brain-score-unified/unified')
import numpy as np
t0 = time.time()
def log(m): print(f'[{time.time()-t0:7.1f}s] {m}', flush=True)
LAYER_IDX = list(range(12))
BUDGETS = [16, 32, 64, 128, 256, 512, 768]
ALPHA_GRID = (1., 10., 100., 1000., 10000., 100000.)
OUT_DIR = '/tmp/vjepa2_sweep'
CACHE = '/tmp/clip_majaj_feats.npz'


def extract_clip(paths):
    """Mean-pooled per-layer CLIP vision features: {layer: (n_stim, 768)}."""
    if os.path.exists(CACHE):
        d = np.load(CACHE)
        return {k: d[k] for k in d.files if k.startswith('encoder')}
    import torch
    from transformers import CLIPModel, CLIPProcessor
    from PIL import Image
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    vm = CLIPModel.from_pretrained('openai/clip-vit-base-patch32').vision_model.eval().to(dev)
    proc = CLIPProcessor.from_pretrained('openai/clip-vit-base-patch32')
    store = {}
    def mk(i):
        def hook(_m, _inp, out):
            o = out[0] if isinstance(out, tuple) else out      # (B, 50, 768)
            store[i] = o.mean(1).detach().float().cpu().numpy()  # mean over tokens
        return hook
    for i in LAYER_IDX:
        vm.encoder.layers[i].register_forward_hook(mk(i))
    feats = {i: [] for i in LAYER_IDX}
    B = 64
    for s in range(0, len(paths), B):
        imgs = [Image.open(p).convert('RGB') for p in paths[s:s + B]]
        px = proc(images=imgs, return_tensors='pt')['pixel_values'].to(dev)
        with torch.no_grad():
            vm(pixel_values=px)
        for i in LAYER_IDX:
            feats[i].append(store[i])
        if s % 640 == 0:
            log(f'  CLIP {s}/{len(paths)}')
    out = {f'encoder.layers.{i}': np.concatenate(feats[i]) for i in LAYER_IDX}
    np.savez(CACHE, **out)
    return out


def main():
    import warnings; warnings.filterwarnings('ignore')
    os.makedirs(OUT_DIR, exist_ok=True)
    import pandas as pd
    from brainscore_vision import load_dataset
    from brainscore.tools import (explore_layer_mapping, score_budget_curve,
                                   effective_dimensionality, rsa_layer_sweep,
                                   compute_rdm, rsa_score)
    from brainscore.tools.layer_mapping import per_voxel_train_test

    log('load MajajHong2015.public assembly...')
    asm = load_dataset('MajajHong2015.public').squeeze()
    if 'time_bin' in asm.dims:
        asm = asm.mean('time_bin')
    region = np.asarray(asm['region'].values)                  # per neuroid
    sids = [str(s) for s in asm['stimulus_id'].values]         # per presentation
    vals = np.asarray(asm.values)
    if vals.shape[0] == len(region):                           # (neuroid, presentation)
        vals = vals.T                                          # -> (presentation, neuroid)
    df = pd.DataFrame(vals); df['sid'] = sids
    avg = df.groupby('sid').mean()
    stim_order = list(avg.index)
    Y_all = avg.values                                         # (n_stim, n_neuroid)
    ss = asm.stimulus_set
    paths = [str(ss.get_stimulus(s)) for s in stim_order]
    log(f'  {len(stim_order)} stimuli; {len(region)} neuroids; regions {sorted(set(region))}')

    log('extract CLIP per-layer features (mean-pooled)...')
    feats = extract_clip(paths)                                # {layer: (n_stim, 768)}
    layers = [f'encoder.layers.{i}' for i in LAYER_IDX]
    log(f'  features per layer: {feats[layers[0]].shape}')

    results = {}
    for reg in sorted(set(region)):                            # 'V4', 'IT'
        cols = np.where(region == reg)[0]
        Y = Y_all[:, cols]
        log(f'=== region {reg}: {Y.shape[1]} neuroids ===')
        res = explore_layer_mapping(feats, Y, localizer_frac=0.5, alpha=1.0, seed=0)
        L, T = res.localizer_idx, res.test_idx

        # regression per-layer sweep + best layer + budget curve + dims
        reg_sweep = {l: round(r, 4) for l, r in zip(res.layer_order, res.per_layer_r)}
        curve = score_budget_curve(feats, Y, res, budgets=BUDGETS, top_n_layers=3,
                                   alpha_grid=ALPHA_GRID, n_null_seeds=5)
        eff_brain = effective_dimensionality(Y)
        eff_feat = effective_dimensionality(feats[res.best_layer])

        # RSA per-layer sweep (on the held-out TEST stimuli, to mirror regression)
        feats_T = {l: feats[l][T] for l in layers}
        rsa_sweep = {l: round(v, 4) for l, v in
                     rsa_layer_sweep(feats_T, Y[T]).items()}
        rsa_best = max(rsa_sweep, key=rsa_sweep.get)

        # RSA unit-selection at the RSA-best layer: full vs top-256 vs random-256
        brain_rdm = compute_rdm(Y[T])
        Xb = feats[rsa_best]
        full_rsa = rsa_score(compute_rdm(Xb[T]), brain_rdm)
        topk = res.top_units(rsa_best, 256)
        top_rsa = rsa_score(compute_rdm(Xb[T][:, topk]), brain_rdm)
        rng = np.random.RandomState(0)
        rand_rsa = float(np.mean([
            rsa_score(compute_rdm(Xb[T][:, rng.choice(Xb.shape[1], 256, replace=False)]), brain_rdm)
            for _ in range(5)]))

        results[reg] = {
            'n_neuroids': int(Y.shape[1]),
            'regression': {
                'per_layer_r': reg_sweep, 'best_layer': res.best_layer,
                'best_r': round(res.best_r, 4),
                'whole_layer_r': curve['whole_layer_r'],
                'within_layer': curve['within_layer'],   # K-curve: selection vs random
                'eff_dim_brain': round(eff_brain, 2),
                'eff_dim_best_layer_features': round(eff_feat, 2),
            },
            'rsa': {
                'per_layer': rsa_sweep, 'best_layer': rsa_best,
                'best_r': rsa_sweep[rsa_best],
                'unit_selection_at_best': {
                    'full_768': round(full_rsa, 4),
                    'top_256': round(top_rsa, 4),
                    'random_256': round(rand_rsa, 4)},
            },
        }
        log(f'  regression best layer={res.best_layer} (r={res.best_r:.3f}); '
            f'RSA best layer={rsa_best} (r={rsa_sweep[rsa_best]:.3f})')
        log(f'  eff-dim brain={eff_brain:.1f}, features={eff_feat:.1f}')
        log(f'  RSA@{rsa_best}: full={full_rsa:.3f} top256={top_rsa:.3f} rand256={rand_rsa:.3f}')

    json.dump({'model': 'clip-vit-b-32', 'benchmark': 'MajajHong2015.public',
               'n_stimuli': len(stim_order), 'layers': layers, 'budgets': BUDGETS,
               'regions': results}, open(f'{OUT_DIR}/majaj_results.json', 'w'), indent=2)
    log('DONE.')
    for reg, r in results.items():
        log(f"  {reg}: reg-best={r['regression']['best_layer']} "
            f"rsa-best={r['rsa']['best_layer']} "
            f"eff-dim-brain={r['regression']['eff_dim_brain']}")


if __name__ == '__main__':
    main()
