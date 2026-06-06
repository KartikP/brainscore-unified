"""The definitive ROI-hierarchy + dimensionality test: Allen2022 NSD fMRI, which
has a true cortical hierarchy as labeled ROIs (V1 → V2 → V4 → IT) and thousands
of natural-scene stimuli. The dataset most likely to BREAK the "one layer / low
effective-dim" story from the narrow benchmarks.

Per ROI (V1, V2, V4, IT), per model (default CLIP + ResNet-50 — ViT vs CNN, the
hierarchy test), the same suite: regression per-layer sweep + best layer +
budget curve + brain effective-dim; RSA per-layer sweep + unit selection.

Key questions:
  - Does brain effective-dim RISE up the hierarchy / with more stimuli?
  - Does V1→early-layer, IT→late-layer SEPARATION finally appear (esp. for the CNN)?
  - Does one-layer-sufficiency still hold per ROI at NSD scale?

Stimuli capped to MAX_STIM (shared across ROIs) to keep RDMs tractable. Raw
scores. Run on EC2 (GPU); first run downloads the NSD assemblies + images.
"""
import argparse, json, os, sys, time
sys.path.insert(0, '/home/ubuntu/brain-score-unified/unified')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
t0 = time.time()
def log(m): print(f'[{time.time()-t0:7.1f}s] {m}', flush=True)
from run_arch_majaj import extract_clip, extract_resnet50, extract_dinov2, analyze
EXTRACTORS = {'clip': extract_clip, 'resnet50': extract_resnet50, 'dinov2': extract_dinov2}
REGIONS = ['V1', 'V2', 'V4', 'IT']
MAX_STIM = 3000
OUT_DIR = '/tmp/vjepa2_sweep'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--models', default='clip,resnet50')
    args = ap.parse_args()
    import warnings; warnings.filterwarnings('ignore')
    os.makedirs(OUT_DIR, exist_ok=True)
    from brainscore_vision.benchmarks.allen2022_fmri.benchmark import load_full_assembly

    log('load Allen2022 ROIs (V1,V2,V4,IT)...')
    region_Y = {}
    canon_sids = None
    ss = None
    for reg in REGIONS:
        a = load_full_assembly(reg)
        sids = [str(s) for s in a['stimulus_id'].values]
        vals = np.asarray(a.values, np.float64)
        if vals.shape[0] != a.sizes['presentation']:
            vals = vals.T                                  # -> (presentation, neuroid)
        # collapse any duplicate stimulus_ids (defensive) by mean
        import pandas as pd
        d = pd.DataFrame(vals); d['sid'] = sids
        g = d.groupby('sid').mean()
        region_Y[reg] = g
        log(f'  {reg}: {g.shape[0]} stimuli x {g.shape[1]} voxels')
        if canon_sids is None:
            canon_sids = list(g.index); ss = a.stimulus_set

    # common stimulus subset shared across all ROIs, capped
    common = [s for s in canon_sids if all(s in region_Y[r].index for r in REGIONS)]
    rng = np.random.RandomState(0)
    if len(common) > MAX_STIM:
        common = list(np.array(common)[np.sort(rng.choice(len(common), MAX_STIM, replace=False))])
    paths = [str(ss.get_stimulus(s)) for s in common]
    log(f'  analysing {len(common)} shared stimuli')

    out = {'benchmark': 'Allen2022_fmri', 'n_stimuli': len(common), 'regions': REGIONS, 'models': {}}
    for mname in args.models.split(','):
        log(f'### model {mname}: extract features ###')
        cache = f'/tmp/feats_{mname}_allen.npz'
        if os.path.exists(cache):
            dd = np.load(cache); feats = {k: dd[k] for k in dd.files}
        else:
            feats = EXTRACTORS[mname](paths); np.savez(cache, **feats)
        layers = list(feats.keys())
        log(f'  {len(layers)} layers; dims {[feats[l].shape[1] for l in layers]}')
        out['models'][mname] = {}
        for reg in REGIONS:
            Y = region_Y[reg].loc[common].values
            r = analyze(feats, Y, layers)
            out['models'][mname][reg] = r
            rg = r['regression']; rs = r['rsa']
            log(f"  {mname}/{reg}: reg-best=d{rg['best_layer_depth_frac']}(r{rg['best_r']}) "
                f"whole={rg['whole_layer_r']} eff-dim-brain={rg['eff_dim_brain']} "
                f"| rsa-best=d{rs['best_layer_depth_frac']}(r{rs['best_r']}) sel={rs['unit_selection_at_best']}")
        json.dump(out, open(f'{OUT_DIR}/allen_results.json', 'w'), indent=2)
    log('DONE.')
    for m in out['models']:
        dims = {reg: out['models'][m][reg]['regression']['eff_dim_brain'] for reg in REGIONS}
        depths = {reg: out['models'][m][reg]['regression']['best_layer_depth_frac'] for reg in REGIONS}
        log(f'  {m}: eff-dim {dims} | best-layer-depth {depths}')


if __name__ == '__main__':
    main()
