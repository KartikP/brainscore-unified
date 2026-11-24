"""A tiny made-up CNeuroMod source in the layout of the DataLad clones (no download).

``make_source(root)`` writes, for every public participant of THINGS and movie10: event files,
masks and the atlas (``anat.atlases``), THINGS GLMsingle trial responses and ``cneuromod2026``
time series. A 4 x 4 x 4 voxel grid; 3 trials per THINGS run, 2 runs, 1 session.
"""
from pathlib import Path

import h5py
import nibabel
import numpy as np
import pandas as pd

REAL_NIBABEL = nibabel          # some packaging tests replace nibabel; the CNeuroMod source needs the real one
TR = 1.49
N_TRS = 9                       # 13.41 s per run
AFFINE = np.diag([2.0, 2.0, 2.0, 1.0])
THINGS = ('sub-01', 'sub-02', 'sub-03', 'sub-06')
MOVIE10 = ('sub-01', 'sub-02', 'sub-03', 'sub-05', 'sub-06')
IMAGE_BYTES = b'\x89PNG made-up image content that must never reach the output'


def _nifti(data, path: Path, dtype=np.uint8) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    nibabel.Nifti1Image(np.asarray(data, dtype=dtype), AFFINE).to_filename(path)


def _grid() -> np.ndarray:
    grid = np.zeros((4, 4, 4), bool)
    grid[1:3, 1:3, 1:3] = True          # 8 voxels
    return grid


def _things_events(session: int, run: int) -> pd.DataFrame:
    names = [f'img{session}{run}{trial}' for trial in range(3)]
    return pd.DataFrame({
        'onset': [0.0, 4.47, 8.94], 'duration': [2.98] * 3,
        'image_path': [f'stimuli/images_fmri/{name[:4]}/{name}.jpg' for name in names],
        'repetition': [1, 2, 1], 'condition': ['unseen', 'seen', 'unseen'], 'response': [1.0, np.nan, -2.0],
        'things_image_nr': [run * 10 + trial for trial in range(3)]})


def _timeseries(path: Path, runs: dict, columns: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, 'w') as handle:
        for (group, name), seed in runs.items():
            handle.require_group(group).create_dataset(
                name, data=np.random.default_rng(seed).normal(size=(N_TRS, columns)).astype(np.float32))


def make_source(root: Path) -> Path:
    grid = _grid()
    v1 = np.zeros((4, 4, 4), bool)
    v1[1, 1:3, 1:3] = True
    ffa = np.zeros((4, 4, 4), bool)
    ffa[2, 1, 1:3] = True
    dseg = np.zeros((4, 4, 4), int)
    dseg[1:3, 1:3, 1] = 1
    dseg[1:3, 1:3, 2] = 1001
    atlases = root / 'anat.atlases'
    labels = ('tpl-MNI152NLin2009cAsym/tpl-MNI152NLin2009cAsym_atlas-Schaefer2018TianS3NettekovenAsym_'
              'desc-1000Parcels7Networks50Subcort128Cereb.tsv')
    (atlas_labels := atlases / labels).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({'index': [1, 2, 1001, 1178], 'name': ['LH_Vis_1', 'LH_Vis_2', 'HIP-head-m-rh', 'Cereb-S5Rv']}
                 ).to_csv(atlas_labels, sep='\t', index=False)
    _nifti(dseg, atlases / 'tpl-MNI152NLin2009cAsym' / 'tpl-MNI152NLin2009cAsym_res-01_atlas-'
           'Schaefer2018TianS3NettekovenAsym_desc-1000Parcels7Networks50Subcort128Cereb_dseg.nii.gz', np.int16)
    (root / 'images').mkdir(parents=True, exist_ok=True)
    (root / 'images' / 'img110.jpg').write_bytes(IMAGE_BYTES)      # a stimulus file the packaging must not copy
    for participant in sorted(set(THINGS) | set(MOVIE10)):
        short = participant.replace('-', '')
        folder = atlases / f'tpl-{short}T1w'
        _nifti(v1, folder / f'tpl-{short}T1w_res-func_atlas-retinoVisionNpythy_label-V1_mask.nii.gz')
        _nifti(ffa, folder / f'tpl-{short}T1w_res-func_atlas-fLocVisionTask_label-faceFFA_mask.nii.gz')
        _nifti(ffa, folder / f'tpl-{short}T1w_res-func_atlas-langToneva_label-IFG_mask.nii.gz')
        _nifti(dseg, atlases / 'tpl-MNI152NLin2009cAsym' /
               f'tpl-MNI152NLin2009cAsym_{participant}_res-func_atlas-Schaefer2018TianS3NettekovenAsym_'
               'desc-1000Parcels7Networks50Subcort128Cereb_dseg.nii.gz', np.int16)
    for participant in THINGS:
        base = f'{participant}_task-things_space-T1w_model-fitHrfGLMdenoiseRR'
        output = root / 'things.glmsingle' / participant / 'glmsingle' / 'output'
        output.mkdir(parents=True, exist_ok=True)
        _nifti(grid, output / f'{base}_stat-noiseCeilings_statmap.nii.gz')
        with h5py.File(output / f'{base}_stat-trialBetas_desc-zscore_statseries.h5', 'w') as handle:
            for run in (1, 2):
                handle.create_dataset(f'1/{run}/betas', data=np.arange(3 * 8, dtype=np.float32).reshape(3, 8) + run)
            handle.create_dataset('mask_array', data=grid.astype(float))
            handle.create_dataset('mask_affine', data=AFFINE)
        func = root / 'things' / participant / 'ses-01' / 'func'
        func.mkdir(parents=True, exist_ok=True)
        for run in (1, 2):
            _things_events(1, run).to_csv(func / f'{participant}_ses-01_task-things_run-{run:02d}_events.tsv',
                                          sep='\t', index=False)
        series = root / 'things.timeseries' / 'timeseries' / 'cneuromod2026' / participant
        stem = f'{participant}_task-things_space-MNI152NLin2009cAsym_atlas-cneuromod26_desc-1134Parcels'
        _nifti(dseg, series / f'{stem}_dseg.nii.gz', np.int16)
        _timeseries(series / f'{stem}_timeseries.h5',
                    {('ses-01', f'ses-01_task-things_run-{run}_timeseries'): run for run in (1, 2)}, 2)
    for segment in ('bourne01', 'life01'):
        events = root / 'movie10' / f'task-{segment}_events.tsv'
        events.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({'onset': [0.0], 'duration': [12.0], 'stim_file': [f'{segment[:-2]}/{segment}.mkv']}
                     ).to_csv(events, sep='\t', index=False)
    for participant in MOVIE10:
        (root / 'movie10' / participant / 'ses-001').mkdir(parents=True, exist_ok=True)
        series = root / 'movie10.timeseries' / 'timeseries' / 'cneuromod2026' / participant
        stem = f'{participant}_task-movie10_space-MNI152NLin2009cAsym_atlas-cneuromod26_desc-1134Parcels'
        _nifti(dseg, series / f'{stem}_dseg.nii.gz', np.int16)
        _timeseries(series / f'{stem}_timeseries.h5',
                    {('ses-001', 'ses-001_task-bourne01_timeseries'): 1,
                     ('ses-001', 'ses-001_task-life01_run-2_timeseries'): 2}, 2)
    return root
