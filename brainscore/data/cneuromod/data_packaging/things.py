"""CNeuroMod-THINGS: what differs from the other CNeuroMod data sets (St-Laurent et al. 2026).

Each event is one image trial (2.98 s image, 1.49 s pause); its stimulus identifier is the THINGS
image file name without the extension (``aardvark_01b``). Each image is shown 3 times
(``repetition`` 1, 2, 3). The GLMsingle trial responses are also read here.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from brainscore.data.cneuromod.data_packaging import shared

NAME = 'things'
REPOSITORY = 'things'
PARTICIPANTS = ('sub-01', 'sub-02', 'sub-03', 'sub-06')
TR_SECONDS = 1.49
STIMULUS_SOURCE = ('THINGS database images (Hebart et al. 2019), https://osf.io/jum2f: academic use only, '
                   'a password zip; get them from OSF under its terms. Brain-Score does not host them.')
STIMULUS_COLUMNS = ('image_path', 'image_nr', 'category_nr', 'exemplar_nr', 'things_category_nr',
                    'things_image_nr', 'things_exemplar_nr')
# The region voxels of the trials, one mask set per region in this order: the participant's own pRF
# and fLoc localizer masks; the Kanwisher group parcels (warped to the participant) only for a region
# without an own mask (sub-06, which did no localizer). ``atlas`` names the source of each voxel.
MASK_ATLASES = ('retinoVisionNpythy', 'fLocVisionTask', 'fLocVisionKanwisher')
GLMSINGLE = 'things.glmsingle'
ANATOMY = 'things.fmriprep'
# The cortex rows: the cortical labels of the FreeSurfer aparc+aseg segmentation (1000-1999 left,
# 2000-2999 right hemisphere), as fMRIPrep gives it on the grid of the BOLD runs.
CORTEX_LABELS = (1000, 3000)
_THIS = sys.modules[__name__]     # this module, as the shared functions take it
_TRIAL_COLUMNS = ('participant', 'session', 'task', 'run', 'event', 'onset', 'duration', 'stimulus_id',
                  'repetition', 'condition', 'response', 'error', 'atypical', 'not_for_memory')


def event_patterns(participant: str, session: str) -> list:
    return [f'{participant}/{session}/func/{participant}_{session}_task-things_run-*_events.tsv']


def run_events(clone: Path, participant: str, run: dict) -> pd.DataFrame:
    """The event table of one run; the stimulus identifier is the image file name without extension."""
    files = [path for path in clone.glob(f'{participant}/ses-*/func/{participant}_ses-*_task-things_run-*_events.tsv')
             if shared.number(shared.entities(path.name)['ses']) == run['session']
             and shared.number(shared.entities(path.name)['run']) == run['run']]
    if len(files) != 1:
        raise FileNotFoundError(f'{participant} session {run["session"]} run {run["run"]}: '
                                f'{len(files)} event files found, 1 expected')
    events = shared.read_tsv(files[0])
    events['stimulus_id'] = [Path(str(path)).stem for path in events['image_path']]
    return events


def answered_viewing(stimulus_id: np.ndarray, response: np.ndarray) -> np.ndarray:
    """For each trial (given in time order) its place among the answered trials of its image: 1, 2, 3, ...;
    0 when the trial has no response. Use this, not the released ``repetition`` column, for the viewing
    order: ``repetition`` is the label of the event files and is 4 or 5 on some trials of out-of-order
    sessions (sub-01 sessions 14-15). The paper's noise ceiling uses the 1st, 2nd and 3rd answered trial.
    """
    ids = pd.Series(np.asarray(stimulus_id).astype(str))
    answered = pd.Series(response).replace(['', 'nan', 'n/a'], np.nan).notna().to_numpy()
    viewing = np.zeros(len(ids), dtype=int)
    viewing[answered] = ids[answered].groupby(ids[answered]).cumcount().to_numpy() + 1
    return viewing


def glmsingle_patterns(participant: str) -> list:
    base = f'{participant}/glmsingle/output/{participant}_task-things_space-T1w_model-fitHrfGLMdenoiseRR'
    return [f'{base}_stat-trialBetas_desc-zscore_statseries.h5', f'{base}_stat-noiseCeilings_statmap.nii.gz']


def cortex_patterns(participant: str) -> list:
    """The fMRIPrep segmentation of the BOLD runs (T1w space); every run gives the same one: one is read."""
    return [f'{participant}/ses-*/func/{participant}_ses-*_task-things_run-*_space-T1w_desc-aparcaseg_dseg.nii.gz']


def cortex_neuroids(dseg: Path, grid) -> tuple:
    """``(neuroid table, grid columns)`` of the cortical voxels of the GLMsingle grid: rows with
    ``region='cortex'``, ``atlas='aparcaseg'``. The segmentation must be on the grid (no resampling)."""
    import nibabel
    image = nibabel.load(dseg)
    if image.shape[:3] != grid.shape[:3] or not np.allclose(image.affine, grid.affine):
        raise ValueError(f'{dseg.name} is not on the grid of the trial responses (shape {image.shape[:3]} '
                         f'against {grid.shape[:3]}): the cortex rows are read without resampling. '
                         'Use the space-T1w segmentation of a THINGS run.')
    labels = np.asarray(image.dataobj).astype(int)[np.asarray(grid.dataobj) > 0]
    low, high = CORTEX_LABELS
    inside = np.flatnonzero((labels >= low) & (labels < high))
    return shared.voxel_rows('aparcaseg', 'cortex', inside, grid), inside


def glmsingle_trials(clones: Path, participant: str, session: int | None, masks: list, cortex: Path,
                     out: Path) -> list:
    """Write the per-trial response assembly of each session (one response per image per trial),
    the region voxels from the masks of ``MASK_ATLASES`` and the cortex voxels of the segmentation
    ``cortex`` (``cortex_neuroids``), with the run's event table beside it.

    The betas file: group = session, subgroup = run, dataset ``betas`` (60 trials x voxels of
    the file's own mask ``mask_array``/``mask_affine``), trials in onset order.
    """
    import h5py
    import nibabel
    trials_file, noise_ceiling = [clones / GLMSINGLE / pattern for pattern in glmsingle_patterns(participant)]
    bids = clones / REPOSITORY
    shared.copy_files([noise_ceiling], out / participant / 'glmsingle')
    written = []
    with h5py.File(trials_file, 'r') as handle:
        grid = nibabel.Nifti1Image(np.asarray(handle['mask_array']).astype(np.uint8), np.asarray(handle['mask_affine']))
        regions, region_columns = shared.region_neuroids(masks, grid, MASK_ATLASES)
        cortical, cortex_columns = cortex_neuroids(cortex, grid)
        neuroid = pd.concat([regions, cortical], ignore_index=True)
        columns = np.concatenate([region_columns, cortex_columns])
        for group in sorted((name for name in handle if name not in ('mask_array', 'mask_affine')), key=shared.number):
            if session is not None and shared.number(group) != session:
                continue
            values, tables = [], []
            for run_name in sorted(handle[group], key=shared.number):
                run = {'session': shared.number(group), 'task': 'things', 'run': shared.number(run_name)}
                betas = np.asarray(handle[group][run_name]['betas'], dtype=np.float32)
                events = shared.run_table(_THIS, bids, participant, run)
                if len(events) != len(betas):
                    raise ValueError(f'{participant} session {run["session"]} run {run["run"]}: {len(betas)} '
                                     f'trial responses but {len(events)} events')
                values.append(betas[:, columns])
                tables.append(events)
            events = pd.concat(tables, ignore_index=True)
            folder = out / participant / f'ses-{shared.number(group):02d}'
            presentation = events[[c for c in _TRIAL_COLUMNS if c in events]].rename(columns={'event': 'trial'})
            presentation['trial'] = presentation['trial'] + 1
            shared.write_assembly(np.concatenate(values), presentation, neuroid, folder / 'trials.nc')
            shared.write_csv(events, folder / 'events.csv')
            shared.write_csv(shared.stimulus_table(_THIS, events), folder / 'stimuli.csv')
            written.append(folder)
    return written
