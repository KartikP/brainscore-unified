"""Multi-architecture layer-mapping comparison on MajajHong2015 V4 + IT.

Tests whether the Pass-1 findings (low-dim brain signal; one layer per region;
selection only helps at small budgets; RSA more selection-sensitive than
regression) depend on the model architecture / training objective, and whether a
CNN — which HAS a spatial hierarchy, unlike CLIP's ViT — separates V4 (early) from
IT (late) where CLIP did not.

Models (pass via --models, default clip,resnet50,dinov2):
  clip      ViT-B/32, image-text contrastive       (encoder.layers.0-11, 768, mean-pooled)
  resnet50  CNN, ImageNet-supervised               (layer1-4, spatial-mean-pooled)
  dinov2    ViT-B/14, self-supervised (DINOv2)      (encoder.layer.0-11, 768, mean-pooled)
  vjepa2    ViT-L, self-supervised VIDEO on stills  (CAVEAT: degenerate temporal input)

Per model, per region: regression per-layer sweep + best layer + budget curve +
effective dim; RSA per-layer sweep + best layer + unit-selection (full/top/random).
Raw scores. Run on EC2 (GPU).
"""
import argparse, json, os, sys, time
sys.path.insert(0, '/home/ubuntu/brain-score-unified/unified')
import numpy as np
t0 = time.time()
def log(m): print(f'[{time.time()-t0:7.1f}s] {m}', flush=True)
BUDGETS = [16, 32, 64, 128, 256, 512]
ALPHA_GRID = (1., 10., 100., 1000., 10000., 100000.)
OUT_DIR = '/tmp/vjepa2_sweep'


# ── per-architecture feature extractors → {layer: (n_stim, dim)} ─────────────

def _batched(paths, fn, B=64):
    outs = None
    for s in range(0, len(paths), B):
        d = fn(paths[s:s + B])
        if outs is None:
            outs = {k: [] for k in d}
        for k in d:
            outs[k].append(d[k])
        if s % 640 == 0:
            log(f'    {s}/{len(paths)}')
    return {k: np.concatenate(v) for k, v in outs.items()}


def extract_clip(paths):
    import torch
    from transformers import CLIPModel, CLIPProcessor
    from PIL import Image
    dev = 'cuda'
    vm = CLIPModel.from_pretrained('openai/clip-vit-base-patch32').vision_model.eval().to(dev)
    proc = CLIPProcessor.from_pretrained('openai/clip-vit-base-patch32')
    store = {}
    for i in range(12):
        vm.encoder.layers[i].register_forward_hook(
            (lambda i: lambda _m, _x, o: store.__setitem__(
                i, (o[0] if isinstance(o, tuple) else o).mean(1).detach().float().cpu().numpy()))(i))
    def step(bp):
        imgs = [Image.open(p).convert('RGB') for p in bp]
        px = proc(images=imgs, return_tensors='pt')['pixel_values'].to(dev)
        with torch.no_grad(): vm(pixel_values=px)
        return {f'encoder.layers.{i}': store[i] for i in range(12)}
    return _batched(paths, step)


def extract_resnet50(paths):
    import torch, torchvision
    from torchvision import transforms
    from PIL import Image
    dev = 'cuda'
    m = torchvision.models.resnet50(weights='IMAGENET1K_V2').eval().to(dev)
    tf = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224),
                             transforms.ToTensor(),
                             transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    store = {}
    for name in ['layer1', 'layer2', 'layer3', 'layer4']:
        getattr(m, name).register_forward_hook(
            (lambda n: lambda _m, _x, o: store.__setitem__(
                n, o.mean(dim=(2, 3)).detach().float().cpu().numpy()))(name))
    def step(bp):
        x = torch.stack([tf(Image.open(p).convert('RGB')) for p in bp]).to(dev)
        with torch.no_grad(): m(x)
        return {n: store[n] for n in ['layer1', 'layer2', 'layer3', 'layer4']}
    return _batched(paths, step)


def extract_dinov2(paths):
    import torch
    from transformers import AutoModel, AutoImageProcessor
    from PIL import Image
    dev = 'cuda'
    m = AutoModel.from_pretrained('facebook/dinov2-base').eval().to(dev)
    proc = AutoImageProcessor.from_pretrained('facebook/dinov2-base')
    store = {}
    for i in range(12):
        m.encoder.layer[i].register_forward_hook(
            (lambda i: lambda _m, _x, o: store.__setitem__(
                i, (o[0] if isinstance(o, tuple) else o).mean(1).detach().float().cpu().numpy()))(i))
    def step(bp):
        imgs = [Image.open(p).convert('RGB') for p in bp]
        px = proc(images=imgs, return_tensors='pt')['pixel_values'].to(dev)
        with torch.no_grad(): m(pixel_values=px)
        return {f'encoder.layer.{i}': store[i] for i in range(12)}
    return _batched(paths, step)


def extract_vjepa2(paths):
    """V-JEPA2 on repeated-frame stills (degenerate temporal input — caveated)."""
    import torch
    from transformers import AutoModel, AutoVideoProcessor
    from PIL import Image
    dev = 'cuda'
    m = AutoModel.from_pretrained('facebook/vjepa2-vitl-fpc64-256',
                                  torch_dtype=torch.float16).eval().to(dev)
    proc = AutoVideoProcessor.from_pretrained('facebook/vjepa2-vitl-fpc64-256')
    LYS = list(range(0, 24, 2))                       # every other layer to bound cost
    store = {}
    for i in LYS:
        m.encoder.layer[i].register_forward_hook(
            (lambda i: lambda _m, _x, o: store.__setitem__(
                i, (o[0] if isinstance(o, tuple) else o).mean(1).detach().float().cpu().numpy()))(i))
    def step(bp):
        out = {}
        for p in bp:                                  # 1 clip at a time (memory)
            img = np.array(Image.open(p).convert('RGB').resize((256, 256)))
            vid = np.repeat(img[None], 64, axis=0)    # 64 identical frames
            px = proc(list(vid), return_tensors='pt')['pixel_values_videos'].to(dev).half()
            with torch.no_grad(): m(px, skip_predictor=True)
            for i in LYS:
                out.setdefault(f'encoder.layer.{i}', []).append(store[i])
        return {k: np.concatenate(v) for k, v in out.items()}
    return _batched(paths, step, B=16)


EXTRACTORS = {'clip': extract_clip, 'resnet50': extract_resnet50,
              'dinov2': extract_dinov2, 'vjepa2': extract_vjepa2}


# ── per-region analysis (same suite as Pass 1) ───────────────────────────────

def analyze(feats, Y, layers):
    from brainscore.tools import (explore_layer_mapping, score_budget_curve,
                                  effective_dimensionality, rsa_layer_sweep,
                                  compute_rdm, rsa_score)
    res = explore_layer_mapping(feats, Y, localizer_frac=0.5, alpha=1.0, seed=0)
    T = res.test_idx
    curve = score_budget_curve(feats, Y, res, budgets=[b for b in BUDGETS],
                               top_n_layers=min(3, len(layers)), alpha_grid=ALPHA_GRID,
                               n_null_seeds=5)
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
    return {
        'regression': {
            'per_layer_r': {l: round(r, 4) for l, r in zip(res.layer_order, res.per_layer_r)},
            'best_layer': res.best_layer, 'best_r': round(res.best_r, 4),
            'best_layer_depth_frac': round(layers.index(res.best_layer) / max(1, len(layers) - 1), 2),
            'whole_layer_r': curve['whole_layer_r'], 'within_layer': curve['within_layer'],
            'eff_dim_brain': round(effective_dimensionality(Y), 2),
            'eff_dim_best_layer': round(effective_dimensionality(feats[res.best_layer]), 2)},
        'rsa': {'per_layer': rsa, 'best_layer': rsa_best, 'best_r': rsa[rsa_best],
                'best_layer_depth_frac': round(layers.index(rsa_best) / max(1, len(layers) - 1), 2),
                'unit_selection_at_best': sel}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--models', default='clip,resnet50,dinov2')
    args = ap.parse_args()
    import warnings; warnings.filterwarnings('ignore')
    os.makedirs(OUT_DIR, exist_ok=True)
    import pandas as pd
    from brainscore_vision import load_dataset

    log('load MajajHong2015.public...')
    asm = load_dataset('MajajHong2015.public').squeeze()
    if 'time_bin' in asm.dims: asm = asm.mean('time_bin')
    region = np.asarray(asm['region'].values)
    sids = [str(s) for s in asm['stimulus_id'].values]
    vals = np.asarray(asm.values)
    if vals.shape[0] == len(region): vals = vals.T
    df = pd.DataFrame(vals); df['sid'] = sids
    avg = df.groupby('sid').mean(); stim_order = list(avg.index)
    Y_all = avg.values
    ss = asm.stimulus_set
    paths = [str(ss.get_stimulus(s)) for s in stim_order]
    log(f'  {len(stim_order)} stimuli, regions {sorted(set(region))}')

    out = {'benchmark': 'MajajHong2015.public', 'n_stimuli': len(stim_order), 'models': {}}
    for mname in args.models.split(','):
        log(f'### model {mname}: extract features ###')
        cache = f'/tmp/feats_{mname}_majaj.npz'
        if os.path.exists(cache):
            d = np.load(cache); feats = {k: d[k] for k in d.files}
        else:
            feats = EXTRACTORS[mname](paths); np.savez(cache, **feats)
        layers = list(feats.keys())
        log(f'  {len(layers)} layers; dims {[feats[l].shape[1] for l in layers]}')
        out['models'][mname] = {}
        for reg in sorted(set(region)):
            Y = Y_all[:, np.where(region == reg)[0]]
            r = analyze(feats, Y, layers)
            out['models'][mname][reg] = r
            rg = r['regression']; rs = r['rsa']
            log(f"  {mname}/{reg}: reg-best={rg['best_layer']} "
                f"(depth {rg['best_layer_depth_frac']}, r={rg['best_r']}) "
                f"| eff-dim-brain={rg['eff_dim_brain']} "
                f"| rsa-best={rs['best_layer']} sel={rs['unit_selection_at_best']}")
        json.dump(out, open(f'{OUT_DIR}/arch_majaj_results.json', 'w'), indent=2)
    log('DONE.')


if __name__ == '__main__':
    main()
