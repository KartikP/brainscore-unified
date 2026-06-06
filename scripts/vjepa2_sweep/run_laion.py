"""The richness test: LAION-fMRI per-subject pool (sub-01, ~5,833 high-SNR 7T
stimuli — ~11x the Allen subset), V1/V2/V4/IT, across three architectures
(CLIP contrastive-ViT, ResNet-50 CNN, V-JEPA-2 vision tower = TRIBEv2's vision
backbone on repeated-frame stills).

The question every earlier benchmark left open: with an order of magnitude more
stimuli and higher SNR, does brain effective-dimensionality finally climb toward
the hundreds, and does one-layer-sufficiency BREAK (budget curve stops
saturating / depth starts paying)? Plus the V1->IT hierarchy at scale.

Loads the assembly + DUA-mirrored stimuli straight from brainscore-storage (auth
boto3 / aws cli) by S3 id — no laion_fmri plugin/package needed. Own 50/50
localizer/test split. Raw scores. Run on EC2.

  --smoke : load assembly + stimuli, print diagnostics, no feature extraction.
"""
import argparse, json, os, subprocess, sys, time, zipfile
sys.path.insert(0, '/home/ubuntu/brain-score-unified/unified')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
t0 = time.time()
def log(m): print(f'[{time.time()-t0:7.1f}s] {m}', flush=True)
from run_arch_majaj import extract_clip, extract_resnet50, extract_vjepa2, analyze
EXTRACTORS = {'clip': extract_clip, 'resnet50': extract_resnet50, 'vjepa2': extract_vjepa2}
REGIONS = ['V1', 'V2', 'V4', 'IT']
NC_THRESH = 30.0
MAX_STIM = 6000
BUCKET = 'brainscore-storage/brainscore-vision/benchmarks/LAION_fMRI'
S3 = dict(identifier='LAION_fMRI_persubject_sub-01_Assembly',
          version_id='ZMebfDS6bT7DjDqKcuHmOu9DlRbPstk6',
          sha1='827781fe183ee1744f968517c1ea5afbe7860d4b')
STIM_ZIP = 's3://brainscore-storage/brainscore-vision/benchmarks/LAION_fMRI/stimuli/images_extracted.zip'
STIM_DIR = '/home/ubuntu/laion_stim'
OUT_DIR = '/tmp/vjepa2_sweep'


def fetch_stimuli():
    os.makedirs(STIM_DIR, exist_ok=True)
    ex = os.path.join(STIM_DIR, 'images_extracted')
    if os.path.exists(os.path.join(ex, 'manifest.csv')):
        return ex
    zp = os.path.join(STIM_DIR, 'images_extracted.zip')
    if not os.path.exists(zp):
        log('  downloading 3.2 GB stimulus zip...')
        subprocess.run(['aws', 's3', 'cp', STIM_ZIP, zp], check=True)
    log('  extracting...')
    with zipfile.ZipFile(zp) as zf:
        zf.extractall(STIM_DIR)
    return ex


def load_assembly():
    from brainscore_core.supported_data_standards.brainio.s3 import load_assembly_from_s3
    from brainscore_core.supported_data_standards.brainio.assemblies import NeuroidAssembly
    return load_assembly_from_s3(identifier=S3['identifier'], version_id=S3['version_id'],
                                 sha1=S3['sha1'], bucket=BUCKET, cls=NeuroidAssembly)


def main():
    import warnings; warnings.filterwarnings('ignore')
    ap = argparse.ArgumentParser()
    ap.add_argument('--models', default='clip,resnet50,vjepa2')
    ap.add_argument('--smoke', action='store_true')
    args = ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)
    import pandas as pd

    log('load LAION-fMRI persubject sub-01 assembly...')
    da = load_assembly()
    log(f'  dims={da.dims} sizes={dict(da.sizes)}')
    neuro_coords = [c for c in da.coords if 'neuroid' in str(da[c].dims)]
    pres_coords = [c for c in da.coords if 'presentation' in str(da[c].dims)]
    log(f'  neuroid coords: {neuro_coords}')
    log(f'  presentation coords: {pres_coords}')
    region = np.asarray(da['region'].values)
    region_counts = {r: int((region == r).sum()) for r in sorted(set(region.tolist()))}
    log(f'  region voxel counts: {region_counts}')
    nc_name = next((c for c in neuro_coords if 'nc' in c.lower()), None)
    log(f'  NC coord = {nc_name}')

    ex = fetch_stimuli()
    manifest = pd.read_csv(os.path.join(ex, 'manifest.csv'))
    log(f'  manifest cols={list(manifest.columns)}; rows={len(manifest)}')
    fname_col = 'filename' if 'filename' in manifest.columns else manifest.columns[-1]
    sid2path = {str(s): os.path.join(ex, str(f))
                for s, f in zip(manifest['stimulus_id'], manifest[fname_col])}

    if args.smoke:
        sids = [str(s) for s in da['stimulus_id'].values]
        uniq = list(dict.fromkeys(sids))
        have = [s for s in uniq if s in sid2path and os.path.exists(sid2path[s])]
        log(f'SMOKE: {len(uniq)} unique stimulus_ids; {len(have)} have images on disk')
        if nc_name is not None:
            nc = np.asarray(da[nc_name].values)
            log(f'SMOKE: NC[{nc_name}] median={np.nanmedian(nc):.1f}; '
                f'>= {NC_THRESH}: {int((nc >= NC_THRESH).sum())}/{len(nc)} voxels')
        log('SMOKE OK'); return

    # orient (presentation, neuroid)
    vals = np.asarray(da.values, np.float64)
    npres = da.sizes['presentation']
    if vals.shape[0] != npres:
        vals = vals.T
    sids = [str(s) for s in da['stimulus_id'].values]
    nc = np.asarray(da[nc_name].values)              # per neuroid

    # average reps per stimulus
    dfv = pd.DataFrame(vals); dfv['sid'] = sids
    avg = dfv.groupby('sid').mean()
    stim_order = [s for s in avg.index if s in sid2path and os.path.exists(sid2path[s])]
    rng = np.random.RandomState(0)
    if len(stim_order) > MAX_STIM:
        idx = np.sort(rng.choice(len(stim_order), MAX_STIM, replace=False))
        stim_order = [stim_order[i] for i in idx]
    Y_all = avg.loc[stim_order].values               # (n_stim, n_neuroid)
    paths = [sid2path[s] for s in stim_order]
    log(f'  {len(stim_order)} stimuli with images; Y_all={Y_all.shape}')

    out = {'benchmark': 'LAION_fMRI_persubject_sub-01', 'n_stimuli': len(stim_order),
           'regions': REGIONS, 'nc_threshold': NC_THRESH, 'models': {}}
    for mname in args.models.split(','):
        log(f'### model {mname}: extract features ###')
        cache = f'/tmp/feats_{mname}_laion.npz'
        if os.path.exists(cache):
            dd = np.load(cache); feats = {k: dd[k] for k in dd.files}
        else:
            feats = EXTRACTORS[mname](paths); np.savez(cache, **feats)
        layers = list(feats.keys())
        log(f'  {len(layers)} layers; dims {[feats[l].shape[1] for l in layers]}')
        out['models'][mname] = {}
        for reg in REGIONS:
            cols = np.where((region == reg) & (nc >= NC_THRESH))[0]
            if len(cols) == 0:
                log(f'  {reg}: no voxels above NC threshold, skipping'); continue
            Y = Y_all[:, cols]
            r = analyze(feats, Y, layers)
            out['models'][mname][reg] = r
            rg = r['regression']; rs = r['rsa']
            log(f"  {mname}/{reg} ({Y.shape[1]} vox): reg-best=d{rg['best_layer_depth_frac']}"
                f"(r{rg['best_r']}) whole={rg['whole_layer_r']} eff-dim-brain={rg['eff_dim_brain']}"
                f" | rsa-best=d{rs['best_layer_depth_frac']} sel={rs['unit_selection_at_best']}")
        json.dump(out, open(f'{OUT_DIR}/laion_results.json', 'w'), indent=2)
    log('DONE.')
    for m in out['models']:
        dims = {reg: out['models'][m][reg]['regression']['eff_dim_brain']
                for reg in REGIONS if reg in out['models'][m]}
        depths = {reg: out['models'][m][reg]['regression']['best_layer_depth_frac']
                  for reg in REGIONS if reg in out['models'][m]}
        log(f'  {m}: eff-dim {dims} | best-layer-depth {depths}')


if __name__ == '__main__':
    main()
