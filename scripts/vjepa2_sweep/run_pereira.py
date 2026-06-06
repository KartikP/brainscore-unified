"""Cross-MODALITY test on Pereira2018 243sentences (text → language-network fMRI).

Does the vision story — low-dimensional brain signal, one layer per region,
selection only helps at small budgets, RSA more selection-sensitive — also hold
for LANGUAGE? GPT-2 (causal LM, a different architecture family) per layer
(transformer.h.0-11, last-token, 768/layer) vs the human language network.

Same suite as the vision runs: regression per-layer sweep + budget curve +
effective dim; RSA per-layer sweep + unit selection (full/top/random). Raw
scores. Light (243 sentences, GPT-2 small) — runs on CPU or GPU.
"""
import json, os, sys, time
sys.path.insert(0, '/home/ubuntu/brain-score-unified/unified')
import numpy as np
t0 = time.time()
def log(m): print(f'[{time.time()-t0:7.1f}s] {m}', flush=True)
BUDGETS = [16, 32, 64, 128, 256, 512]
ALPHA_GRID = (1., 10., 100., 1000., 10000., 100000.)
OUT_DIR = '/tmp/vjepa2_sweep'


def extract_gpt2(sentences):
    import torch
    from transformers import GPT2Model, GPT2Tokenizer
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    m = GPT2Model.from_pretrained('gpt2').eval().to(dev)
    tok = GPT2Tokenizer.from_pretrained('gpt2')
    store = {}
    for i in range(12):
        m.h[i].register_forward_hook(
            (lambda i: lambda _m, _x, o: store.__setitem__(
                i, (o[0] if isinstance(o, tuple) else o)))(i))
    feats = {i: [] for i in range(12)}
    for s, sent in enumerate(sentences):
        ids = tok(str(sent), return_tensors='pt').to(dev)
        with torch.no_grad():
            m(**ids)
        for i in range(12):
            feats[i].append(store[i][0, -1].detach().float().cpu().numpy())  # last token
        if s % 60 == 0:
            log(f'  GPT-2 {s}/{len(sentences)}')
    return {f'h.{i}': np.stack(feats[i]) for i in range(12)}


def main():
    import warnings; warnings.filterwarnings('ignore')
    os.makedirs(OUT_DIR, exist_ok=True)
    from brainscore_language import load_dataset
    from brainscore.tools import (explore_layer_mapping, score_budget_curve,
                                  effective_dimensionality, rsa_layer_sweep,
                                  compute_rdm, rsa_score)

    log('load Pereira2018.language (243sentences)...')
    data = load_dataset('Pereira2018.language').sel(experiment='243sentences').dropna('neuroid')
    # orient to (presentation, neuroid)
    dims = data.dims
    Y = np.asarray(data.values, np.float64)
    n_pres = data.sizes['presentation']
    if Y.shape[0] != n_pres:
        Y = Y.T
    sentences = [str(s) for s in data['stimulus'].values]
    log(f'  {Y.shape[0]} sentences, {Y.shape[1]} language-network voxels')

    log('extract GPT-2 per-layer features (last token)...')
    feats = extract_gpt2(sentences)
    layers = [f'h.{i}' for i in range(12)]
    log(f'  features per layer: {feats[layers[0]].shape}')

    res = explore_layer_mapping(feats, Y, localizer_frac=0.5, alpha=1.0, seed=0)
    T = res.test_idx
    curve = score_budget_curve(feats, Y, res, budgets=BUDGETS, top_n_layers=3,
                               alpha_grid=ALPHA_GRID, n_null_seeds=5)
    feats_T = {l: feats[l][T] for l in layers}
    rsa = {l: round(v, 4) for l, v in rsa_layer_sweep(feats_T, Y[T]).items()}
    rsa_best = max(rsa, key=rsa.get)
    brain_rdm = compute_rdm(Y[T]); Xb = feats[rsa_best]
    nb = min(256, Xb.shape[1])
    rng = np.random.RandomState(0)
    sel = {'full': round(rsa_score(compute_rdm(Xb[T]), brain_rdm), 4),
           f'top_{nb}': round(rsa_score(compute_rdm(Xb[T][:, res.top_units(rsa_best, nb)]), brain_rdm), 4),
           f'random_{nb}': round(float(np.mean([
               rsa_score(compute_rdm(Xb[T][:, rng.choice(Xb.shape[1], nb, replace=False)]), brain_rdm)
               for _ in range(5)])), 4)}

    out = {'benchmark': 'Pereira2018.243sentences', 'model': 'gpt2',
           'n_stimuli': Y.shape[0], 'n_voxels': int(Y.shape[1]),
           'regression': {
               'per_layer_r': {l: round(r, 4) for l, r in zip(res.layer_order, res.per_layer_r)},
               'best_layer': res.best_layer, 'best_r': round(res.best_r, 4),
               'best_layer_depth_frac': round(layers.index(res.best_layer) / 11, 2),
               'whole_layer_r': curve['whole_layer_r'], 'within_layer': curve['within_layer'],
               'eff_dim_brain': round(effective_dimensionality(Y), 2),
               'eff_dim_best_layer': round(effective_dimensionality(feats[res.best_layer]), 2)},
           'rsa': {'per_layer': rsa, 'best_layer': rsa_best, 'best_r': rsa[rsa_best],
                   'best_layer_depth_frac': round(layers.index(rsa_best) / 11, 2),
                   'unit_selection_at_best': sel}}
    json.dump(out, open(f'{OUT_DIR}/pereira_results.json', 'w'), indent=2)
    rg = out['regression']
    log(f"DONE. reg-best={rg['best_layer']} (depth {rg['best_layer_depth_frac']}, r={rg['best_r']}); "
        f"whole={rg['whole_layer_r']}; eff-dim brain={rg['eff_dim_brain']} feat={rg['eff_dim_best_layer']}")
    log(f"  budget K16: sel={curve['within_layer'][0]['r']} rand={curve['within_layer'][0]['random_null']}; "
        f"K{curve['within_layer'][-1]['k']}: sel={curve['within_layer'][-1]['r']} rand={curve['within_layer'][-1]['random_null']}")
    log(f"  RSA best={rsa_best} (r={rsa[rsa_best]}); selection {sel}")


if __name__ == '__main__':
    main()
