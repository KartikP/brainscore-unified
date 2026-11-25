"""Data plugin for the Courtois NeuroMod (CNeuroMod) data sets: one shared packaging for all.

The shared part (``data_packaging/shared.py``) downloads with DataLad, reads the time series,
the masks and the ``cneuromod2026`` atlas, joins each TR to the run's event table and writes the
files; one small module per data set (``things.py``, ``movie10.py``) gives what differs.
Brain-Score hosts none of these files: a user packages them locally. See ``README.md``.
"""
from brainscore import data_builder_registry, data_registry, stimulus_set_registry
from brainscore.data import local
from brainscore.data.packaging import DataBuilder

THINGS_IMAGES = local.register(local.LocalAsset(
    name='things-images',
    kind='stimuli',
    why_local='The THINGS images are for academic use only, behind a password zip on OSF; Brain-Score hosts '
              'no copy.',
    source='https://osf.io/jum2f (THINGS database, Hebart et al. 2019)',
    obtain='`images_THINGS.zip` under its terms; unzip it and put its `object_images` folder (one folder per '
           'category: `aardvark/aardvark_01b.jpg`, ...) at the path above.',
    used_by=[f'StLaurent2026-fMRI-things-{region}' for region in
             ('V1', 'V2', 'V3', 'V4', 'FFA', 'OFA', 'pSTS', 'PPA', 'OPA', 'MPA', 'EBA', 'cortex')],
))

_COMMAND = ('python -m brainscore.data.cneuromod.data_packaging.package --dataset {dataset} --form {form} '
            '--source {{source}} --out {{output}} --part {{part}}')
_SOURCE = ('Courtois NeuroMod, https://docs.cneuromod.ca (the repositories at https://github.com/courtois-neuromod); '
           'the documentation states CC0 for the public participants, and each repository README says '
           '"Redistribution of data outside of the Courtois NeuroMod data bank is prohibited": package locally, '
           'do not share the output.')
_LAYOUT = ('none: the script clones the repositories with DataLad (https, no recursion) and downloads only the '
           'files it needs into `$BRAINSCORE_DATA_HOME/sources/cneuromod`, kept so that a stopped run resumes. '
           'A folder of earlier clones (`{repositories}`) given as the source is read where it is')
_PART = ('A part: `sub-XX` (one participant) or `sub-XX/ses-NN` (one session); sub-04 is restricted and '
         'never packaged.')
_FILES = ('`atlas/` (a copy of the group cneuromod2026 atlas and its labels, with `parcels.csv`, the parcels '
          'present); per participant `masks/` (copies of the region masks); per session `ses-NN/`: ')

data_builder_registry['cneuromod-things'] = lambda: DataBuilder(
    source=_SOURCE + ' THINGS: St-Laurent et al. 2026, Scientific Data 13:141. The images are not included.',
    access='public, no account: masks, atlas, event files, GLMsingle outputs and the fMRIPrep segmentation of '
           'sub-01, 02, 03 and 06.',
    layout=_LAYOUT.format(repositories='things/`, `things.glmsingle/`, `things.fmriprep/`, `anat.atlases/') + '.',
    needs='DataLad and git-annex 10.20230126 or later (tested: 10.20261026 and 10.20261113; from '
          '10.20261113 git-annex asks to confirm the CONP store, which the command does); h5py, nibabel, nilearn; per participant 11.32 GB to '
          'download (the trial-response file holds every session, also for a one-session part) and about 16 GB '
          'of disk; measured for sub-01 on a cloud machine: 403 s, mostly the download (24 MB/s); a '
          'rebuild from the kept download: 3 s per session.',
    writes=_FILES + '`trials.nc` (GLMsingle trial responses: the region voxels, one mask set per region, and '
           'the cortex voxels of the fMRIPrep segmentation, `region=\'cortex\'`), `events.csv` (the '
           'event files, each trial with its stimulus identifier), `stimuli.csv` (identifiers only); and '
           '`glmsingle/` (the released noise-ceiling map). ' + _PART,
    build=_COMMAND.format(dataset='things', form='glmsingle'),
    fetches=True,
)

for _dataset, _participants in (('things', 'sub-01, 02, 03 and 06'), ('movie10', 'sub-01, 02, 03, 05 and 06'),
                                ('mario', 'sub-01, 02, 03, 05 and 06')):
    data_builder_registry[f'cneuromod-{_dataset}-timeseries'] = lambda dataset=_dataset, who=_participants: DataBuilder(
        source=_SOURCE + ' Made with CNeuroMod\'s extraction code (https://github.com/courtois-neuromod/'
                         'cneuromod_extract_tseries, MIT licence) at ae310a53, the commit of the released time series.',
        access=f'public, no account: the time series of {who} are made on this machine from the public fMRIPrep '
               f'outputs ({dataset}.fmriprep) and anat.atlases, at the commits that {dataset}.timeseries names '
               '(its released files need a key from CNeuroMod).',
        layout=_LAYOUT.format(repositories=f'{dataset}/`, `{dataset}.timeseries/`, `anat.atlases/') + ' (and its '
               'released time series files are then read, not made).',
        needs='DataLad and git-annex 10.20230126 or later (checked first); the venv module of the calling Python: '
              'the command makes an environment in `sources/cneuromod/extraction/env` with the pinned versions '
              'of the code and never installs into the calling environment; one session of fMRIPrep outputs on '
              'the disk at a time (dropped after its check). Measured for THINGS sub-01 ses-02 on a cloud machine: '
              '4.76 GB of input, 131 s, 4.2 GB of memory. THINGS: about 169 GB of input and 2 h per participant; '
              'movie10: 475 GB for the 5 participants; mario: 564 GB for the 5 participants (estimates from the '
              'file sizes). Mario: the events are the annotated event files of mario.annotations, and each TR '
              'also has the level, the frame range of its play video and the seconds each button was held.',
        writes=_FILES + '`timeseries_cneuromod2026.nc` (one row per TR: participant, session, task, run, tr, '
               'time, the event and stimulus identifier of the TR), `events.csv`, `stimuli.csv` (identifiers '
               'only), and `sub-XX/atlas/parcels.csv` (the parcels of the participant\'s own atlas). ' + _PART,
        build=_COMMAND.format(dataset=dataset, form='cneuromod2026'),
        fetches=True,
    )


def _trials(participant, region, root=None):
    from .data import load_trials
    return load_trials(participant, region, root)


def _stimuli(root=None):
    from .data import load_stimulus_set
    return load_stimulus_set(root)


data_registry['cneuromod-things'] = _trials
stimulus_set_registry['cneuromod-things'] = _stimuli
