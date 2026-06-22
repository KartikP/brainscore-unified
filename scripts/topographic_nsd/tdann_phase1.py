"""Phase 1: score TDANN (a real topographic vision model) on the NSD-surface
TopographicBenchmark — the first POSITIVE topographic-alignment result.

TDANN is a ResNet-18 trained with a spatial-smoothness loss (Margalit 2024); its units
carry 2-D cortical-sheet positions per layer. We load it WITHOUT VISSL (the checkpoint is a
plain pickled classy_state_dict; strip the `_feature_blocks.` prefix into a torchvision
ResNet-18), record a layer, attach that layer's published positions as tissue_x/tissue_y, and
run it through the real TopographicBenchmark against the NSD fsaverage-surface target.

Expectation: TDANN's spatial loss gives its units genuine topography, so it should CLEAR the
shuffle-coordinate null (signal > 0) — unlike CLIP-on-grid, which sat below it.

Inputs (OSF project 64qv3, fetched via osfclient):
  --ckpt_remote  osfstorage/.../checkpoints/<model>/model_final_checkpoint_phaseNNN.torch
  --pos_remote   osfstorage/.../positions/<model>/.../<layer>.npz
"""
import argparse
import json
import os
import subprocess
import sys

import numpy as np


def osf_fetch(project, remote, local):
    if os.path.exists(local) and os.path.getsize(local) > 0:
        print(f'    cached: {local}', flush=True)
        return local
    os.makedirs(os.path.dirname(local), exist_ok=True)
    print(f'    fetching {remote} -> {local}', flush=True)
    osf_bin = os.path.join(os.path.dirname(sys.executable), 'osf')
    if not os.path.exists(osf_bin):
        osf_bin = 'osf'
    subprocess.run([osf_bin, '-p', project, 'fetch', remote, local], check=True)
    return local


def load_tdann_resnet18(ckpt_path):
    """Load TDANN weights into a torchvision ResNet-18 (fc -> Identity), no VISSL.
    The checkpoint stores classy_state_dict['base_model']['model']['trunk'] with
    `_feature_blocks.` prefixed keys."""
    import torch
    import torch.nn as nn
    from torchvision.models import resnet18

    model = resnet18(weights=None)
    model.fc = nn.Identity()
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    trunk = ckpt['classy_state_dict']['base_model']['model']['trunk']
    # exactly TDANN demo's src/model.load_model_from_checkpoint: strip the
    # 'base_model.' prefix; trunk keys then match torchvision ResNet-18 names.
    sd = {k.split('base_model.')[-1]: v for k, v in trunk.items()
          if k.startswith('base_model') and 'fc.' not in k}
    try:
        model.load_state_dict(sd)                       # strict — must match
        print(f'    loaded TDANN trunk: {len(sd)} keys (strict)', flush=True)
    except RuntimeError as e:
        missing, unexpected = model.load_state_dict(sd, strict=False)
        print(f'    strict failed ({e}); strict=False -> missing={len(missing)} '
              f'unexpected={len(unexpected)}; ckpt key sample={list(trunk.keys())[:4]}', flush=True)
    model.eval()
    return model


def extract_layer_activations(model, image_paths, layer_name, device, batch=32, resize=224):
    """Forward images through the model, capture the named layer's (B,C,H,W) output,
    flatten to (B, C*H*W) in C-major order (matches LayerPositions flat indices)."""
    import torch
    from PIL import Image
    from torchvision import transforms
    tf = transforms.Compose([
        transforms.Resize((resize, resize)), transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    layer = dict(model.named_modules())[layer_name]
    captured = {}
    h = layer.register_forward_hook(lambda m, i, o: captured.__setitem__('a', o.detach()))
    feats = []
    model.to(device)
    with torch.no_grad():
        for s in range(0, len(image_paths), batch):
            imgs = torch.stack([tf(Image.open(p).convert('RGB'))
                                for p in image_paths[s:s + batch]]).to(device)
            model(imgs)
            a = captured['a']                      # (B, C, H, W)
            feats.append(a.reshape(a.shape[0], -1).cpu().numpy())  # C-major flatten
    h.remove()
    return np.concatenate(feats, axis=0)           # (n_images, C*H*W)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--project', default='64qv3')
    ap.add_argument('--ckpt_remote', required=True)
    ap.add_argument('--pos_remote', required=True)
    ap.add_argument('--layer', default='layer4.1')
    ap.add_argument('--region', default='IT')       # NSD-surface brain region (VTC-like)
    ap.add_argument('--subject', default='subj01')
    ap.add_argument('--hemisphere', default='lh')
    ap.add_argument('--nc_threshold', type=float, default=10.0)
    ap.add_argument('--cache', default='/tmp/tdann_data')
    ap.add_argument('--out', default='/tmp/tdann_phase1_result.json')
    args = ap.parse_args()

    import torch
    import brainscore
    from brainscore_vision import load_dataset, load_stimulus_set
    from brainscore.benchmarks.topographic import TopographicBenchmark
    from brainscore.topographic_support import attach_tissue_coords
    from brainscore.metrics.topographic import correlation_distance_profile
    from brainscore_core.supported_data_standards.brainio.assemblies import NeuroidAssembly

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print('[1] fetching TDANN checkpoint + positions from OSF...', flush=True)
    ckpt = osf_fetch(args.project, args.ckpt_remote, f'{args.cache}/ckpt.torch')
    pos = osf_fetch(args.project, args.pos_remote, f'{args.cache}/{args.layer}.npz')

    print('[2] loading TDANN (VISSL-free)...', flush=True)
    model = load_tdann_resnet18(ckpt)

    print('[3] loading layer positions...', flush=True)
    pz = np.load(pos)
    print(f'    npz keys: {list(pz.keys())}', flush=True)
    coords = pz['coordinates'] if 'coordinates' in pz else pz[list(pz.keys())[0]]
    print(f'    positions shape: {coords.shape}', flush=True)

    print('[4] extracting model activations on NSD images...', flush=True)
    stim = load_stimulus_set('Allen2022_fmri_stim_train')
    image_paths = [stim.get_stimulus(sid) for sid in stim['stimulus_id'].values]
    acts = extract_layer_activations(model, image_paths, args.layer, device)
    print(f'    activations: {acts.shape}  (positions: {coords.shape[0]})', flush=True)
    assert acts.shape[1] == coords.shape[0], \
        f'unit count mismatch: acts {acts.shape[1]} vs positions {coords.shape[0]}'

    # model assembly with tissue coords
    model_asm = NeuroidAssembly(
        acts, dims=['presentation', 'neuroid'],
        coords={'stimulus_id': ('presentation', stim['stimulus_id'].values),
                'neuroid_id': ('neuroid', np.arange(acts.shape[1]))})
    model_asm = attach_tissue_coords(model_asm, coords[:, :2])

    # model's own r(d) (should decay if topographic)
    mc_x, mc_y, _ = correlation_distance_profile(acts, coords[:, :2], n_bins=12)
    print('[5] TDANN r(d) profile (should DECAY if topographic):', flush=True)
    for c, m in zip(mc_x, mc_y):
        print(f'    d={c:.3f}  r={m:.4f}', flush=True)

    print('[6] building NSD-surface brain target...', flush=True)
    from nilearn import surface, datasets
    asm = load_dataset('Allen2022_fmri_surface_train')
    if 'time_bin' in asm.dims:
        asm = asm.squeeze('time_bin', drop=True)
    m = ((asm['subject'].values == args.subject) & (asm['region'].values == args.region)
         & (asm['hemisphere'].values == args.hemisphere))
    asm = asm.isel(neuroid=np.where(m)[0])
    keep = np.where(np.asarray(asm['nc_testset'].values) > args.nc_threshold)[0]
    asm = asm.isel(neuroid=keep)
    fs = datasets.fetch_surf_fsaverage('fsaverage')
    lh = np.asarray(surface.load_surf_mesh(fs['infl_left'])[0], dtype=float)
    vidx = np.asarray(asm['vertex_index'].values, dtype=int)
    brain = attach_tissue_coords(asm, lh[vidx])
    print(f'    brain: {brain.sizes["neuroid"]} vertices', flush=True)

    print('[7] scoring TDANN through TopographicBenchmark...', flush=True)

    class _Stub:
        def __init__(self, a): self._a = a
        def start_recording(self, *a, **k): pass
        def process(self, *a, **k): return self._a

    bench = TopographicBenchmark(f'topographic-nsd-{args.region}', brain_assembly=brain,
                                 stimulus_set=stim, region=args.region)
    score = bench(_Stub(model_asm))
    result = {'model': 'tdann-resnet18', 'layer': args.layer, 'region': args.region,
              'subject': args.subject, 'hemisphere': args.hemisphere,
              'n_model_units': int(acts.shape[1]), 'n_brain_vertices': int(brain.sizes['neuroid']),
              'raw': score.attrs['raw'], 'null': score.attrs['null'], 'signal': float(score),
              'tdann_rd_corr': [float(x) for x in mc_y]}
    print('RESULT:', json.dumps(result, indent=2), flush=True)
    json.dump(result, open(args.out, 'w'), indent=2)
    print(f'saved -> {args.out}', flush=True)


if __name__ == '__main__':
    main()
