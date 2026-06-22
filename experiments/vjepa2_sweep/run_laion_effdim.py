"""Confirmation: does the LAION-fMRI low effective-dim (~10) REPLICATE across
subjects? Brain-only (no model features needed — eff-dim is a property of the
responses), so it's cheap: load each subject's per-subject assembly, filter
nc_12rep>=30, per ROI compute the participation ratio of the rep-averaged
responses.
"""
import json, os, sys, time
sys.path.insert(0, '/home/ubuntu/brain-score-unified/unified')
import numpy as np
import pandas as pd
t0 = time.time()
def log(m): print(f'[{time.time()-t0:7.1f}s] {m}', flush=True)
from brainscore.tools import effective_dimensionality
from brainscore_core.supported_data_standards.brainio.s3 import load_assembly_from_s3
from brainscore_core.supported_data_standards.brainio.assemblies import NeuroidAssembly

BUCKET = 'brainscore-storage/brainscore-vision/benchmarks/LAION_fMRI'
NC_THRESH = 30.0
REGIONS = ['V1', 'V2', 'V4', 'IT']
SUBJECTS = {
    'sub-01': ('ZMebfDS6bT7DjDqKcuHmOu9DlRbPstk6', '827781fe183ee1744f968517c1ea5afbe7860d4b'),
    'sub-03': ('KFwyIrU_1hqbPY64iKxE1CWvAHRmLaL2', 'f699b3aeb168a5a6f57a48704bdb03b0103d34d3'),
    'sub-05': ('As0uxDjHTzjtnGSjEVxBqnIEdK0zHq0d', '73a1ee4f85c41c1c30f771a46396a91c73a85b6c'),
}


def main():
    import warnings; warnings.filterwarnings('ignore')
    out = {}
    for sub, (vid, sha) in SUBJECTS.items():
        log(f'load {sub}...')
        da = load_assembly_from_s3(identifier=f'LAION_fMRI_persubject_{sub}_Assembly',
                                   version_id=vid, sha1=sha, bucket=BUCKET, cls=NeuroidAssembly)
        region = np.asarray(da['region'].values)
        nc = np.asarray(da['nc_12rep'].values)
        sids = [str(s) for s in da['stimulus_id'].values]
        vals = np.asarray(da.values, np.float64)
        if vals.shape[0] != da.sizes['presentation']:
            vals = vals.T
        out[sub] = {}
        for reg in REGIONS:
            cols = np.where((region == reg) & (nc >= NC_THRESH))[0]
            if len(cols) == 0:
                continue
            dfv = pd.DataFrame(vals[:, cols]); dfv['sid'] = sids
            Yr = dfv.groupby('sid').mean().values         # (n_stim, n_vox) rep-averaged
            out[sub][reg] = {'eff_dim': round(effective_dimensionality(Yr), 2),
                             'n_vox': int(len(cols)), 'n_stim': int(Yr.shape[0])}
        log(f'  {sub}: ' + str({r: out[sub][r]['eff_dim'] for r in out[sub]}))
    json.dump(out, open('/tmp/vjepa2_sweep/laion_effdim_subjects.json', 'w'), indent=2)
    log('DONE. effective-dim by subject x ROI:')
    for sub in out:
        log(f'  {sub}: ' + str({r: out[sub][r]['eff_dim'] for r in out[sub]}))


if __name__ == '__main__':
    main()
