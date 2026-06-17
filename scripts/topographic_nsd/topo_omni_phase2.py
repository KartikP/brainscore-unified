"""Phase 2: score Topo-Omni (the 2026 flagship topographic multimodal model) on the
NSD-surface TopographicBenchmark.

Topo-Omni is Qwen2.5-Omni-3B fine-tuned with a spatial-smoothness loss onto a single 304x512
cortical sheet (vision rows 0-159/cols 0-255, audio rows 0-159/cols 256-511, language rows
160-303). For image stimuli we record the VISION-band sheet (`out.visual_cortical_sheet`);
each grid cell (row,col) is a unit whose position IS its sheet coordinate — the topographic
unit-coordinate contract met natively. We then score that against the NSD fsaverage-surface
target through the same TopographicBenchmark used for CLIP (-0.155) and TDANN (+0.838).

Load mirrors topo-omni's src/eval/extract/extract_nsd.py (custom Qwen2_5OmniThinker class +
processor; repo root on sys.path so `src.models...` resolves). bf16, fits one A10G.
"""
import argparse
import json
import os
import sys

import numpy as np


def load_topo_omni(run_dir, repo_root, device='cuda'):
    sys.path.insert(0, repo_root)                      # so `src.models...` resolves
    import torch
    from transformers import Qwen2_5OmniProcessor, Qwen2_5OmniThinkerConfig
    from src.models.qwen2_5_omni import Qwen2_5OmniThinkerForConditionalGeneration
    proc = Qwen2_5OmniProcessor.from_pretrained(run_dir)
    cfg = Qwen2_5OmniThinkerConfig.from_pretrained(run_dir)
    cfg.audio_config.is_training = False
    cfg.vision_config.is_training = False
    cfg.text_config.is_training = False
    cfg.apply_spatial_loss = True
    model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
        run_dir, config=cfg, device_map=None, torch_dtype=torch.bfloat16)
    model.to(device).eval()
    return model, proc


def extract_vision_sheet(model, proc, image_paths, device='cuda', batch_size=8):
    """Per image -> the vision-band cortical sheet, mean over the leading (token) dim,
    flattened. Returns (n_images, H*W) responses + (H*W, 2) grid positions (col=x, row=y)."""
    import torch
    from PIL import Image
    from qwen_omni_utils import process_mm_info
    proc.tokenizer.padding_side = 'right'
    sheets = []
    grid_hw = [None]
    for i in range(0, len(image_paths), batch_size):
        imgs = [Image.open(p).convert('RGB') for p in image_paths[i:i + batch_size]]
        chats = [[{'role': 'user', 'content': [{'type': 'image', 'image': im}]}] for im in imgs]
        text = proc.apply_chat_template(chats, add_generation_prompt=False, tokenize=False)
        audios, images, videos = process_mm_info(chats, use_audio_in_video=False)
        inputs = proc(text=text, audio=audios, images=images, videos=videos,
                      return_tensors='pt', padding=True, use_audio_in_video=False).to(device)
        with torch.no_grad():
            out = model(**inputs, return_dict=True)
        # use the UNIFIED sheet (what topo-omni's own extract_nsd.py analyses), not the
        # visual_cortical_sheet (which showed flat r(d) — wrong tensor).
        vs = out.unified_sheet.float().cpu().numpy()    # (B, H, W) — dim0 is the batch
        assert vs.shape[0] == len(imgs), \
            f'sheet batch dim {vs.shape[0]} != n images {len(imgs)} — dim0 is not the batch'
        for b in range(len(imgs)):
            sheets.append(vs[b].reshape(-1))            # one (H*W,) sheet PER image
        grid_hw[0] = vs.shape[1:]
        print(f'    images {i+len(imgs)}/{len(image_paths)}', flush=True)
    H, W = grid_hw[0]
    yy, xx = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
    positions = np.stack([xx.ravel(), yy.ravel()], axis=1).astype(float)  # (H*W, 2): col=x,row=y
    acts = np.stack(sheets, axis=0)                     # (n_images, H*W)
    # keep RESPONSIVE units (non-constant across images) — orientation-agnostic; auto-isolates
    # the band the input modality drives (vision, for images), dropping near-zero audio/lang units.
    std = acts.std(axis=0)
    idx = np.where(std > 1e-6)[0]
    print(f'    responsive units: {len(idx)}/{std.size} (grid {H}x{W})', flush=True)
    # r(d) is a statistical profile over unit pairs; subsample units so the metric's
    # pairwise-index array stays in RAM (115k units -> 6.6B pairs would OOM).
    max_units = 20000
    if len(idx) > max_units:
        rng = np.random.RandomState(0)
        idx = np.sort(rng.choice(idx, size=max_units, replace=False))
        print(f'    subsampled to {max_units} units for r(d)', flush=True)
    return acts[:, idx], positions[idx], (H, W)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run_dir', default='/home/ubuntu/topo_omni_ckpt')
    ap.add_argument('--repo_root', default='/home/ubuntu/topo-omni')
    ap.add_argument('--region', default='IT')
    ap.add_argument('--subject', default='subj01')
    ap.add_argument('--hemisphere', default='lh')
    ap.add_argument('--nc_threshold', type=float, default=10.0)
    ap.add_argument('--out', default='/tmp/topo_omni_phase2_result.json')
    args = ap.parse_args()

    import torch
    import brainscore
    from brainscore_vision import load_dataset, load_stimulus_set
    from brainscore.benchmarks.topographic import TopographicBenchmark
    from brainscore.topographic_support import attach_tissue_coords
    from brainscore.metrics.topographic import correlation_distance_profile
    from brainscore_core.supported_data_standards.brainio.assemblies import NeuroidAssembly

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print('[1] loading Topo-Omni (5B bf16, custom CorticalAdaptor)...', flush=True)
    model, proc = load_topo_omni(args.run_dir, args.repo_root, device)

    print('[2] extracting vision-band cortical sheet on NSD images...', flush=True)
    stim = load_stimulus_set('Allen2022_fmri_stim_train')
    image_paths = [stim.get_stimulus(sid) for sid in stim['stimulus_id'].values]
    acts, positions, (H, W) = extract_vision_sheet(model, proc, image_paths, device)
    print(f'    vision sheet grid: {H}x{W} = {acts.shape[1]} units; acts {acts.shape}', flush=True)

    mc_x, mc_y, _ = correlation_distance_profile(acts, positions, n_bins=12)
    print('[3] Topo-Omni vision-sheet r(d) (should DECAY if topographic):', flush=True)
    for c, m in zip(mc_x, mc_y):
        print(f'    d={c:.3f}  r={m:.4f}', flush=True)

    model_asm = NeuroidAssembly(
        acts, dims=['presentation', 'neuroid'],
        coords={'stimulus_id': ('presentation', stim['stimulus_id'].values),
                'neuroid_id': ('neuroid', np.arange(acts.shape[1]))})
    model_asm = attach_tissue_coords(model_asm, positions)

    print('[4] building NSD-surface brain target...', flush=True)
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
    brain = attach_tissue_coords(asm, lh[np.asarray(asm['vertex_index'].values, dtype=int)])
    print(f'    brain: {brain.sizes["neuroid"]} vertices', flush=True)

    print('[5] scoring Topo-Omni through TopographicBenchmark...', flush=True)

    class _Stub:
        def __init__(self, a): self._a = a
        def start_recording(self, *a, **k): pass
        def process(self, *a, **k): return self._a

    bench = TopographicBenchmark(f'topographic-nsd-{args.region}', brain_assembly=brain,
                                 stimulus_set=stim, region=args.region)
    score = bench(_Stub(model_asm))
    result = {'model': 'topo-omni', 'sheet': 'unified-responsive', 'grid': [int(H), int(W)],
              'region': args.region, 'subject': args.subject, 'hemisphere': args.hemisphere,
              'n_model_units': int(acts.shape[1]), 'n_brain_vertices': int(brain.sizes['neuroid']),
              'raw': score.attrs['raw'], 'null': score.attrs['null'], 'signal': float(score),
              'topo_omni_rd_corr': [float(x) for x in mc_y]}
    print('RESULT:', json.dumps(result, indent=2), flush=True)
    json.dump(result, open(args.out, 'w'), indent=2)
    print(f'saved -> {args.out}', flush=True)


if __name__ == '__main__':
    main()
