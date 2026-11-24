"""Data plugin for the Courtois NeuroMod (CNeuroMod) data sets: one shared packaging for all.

The shared part (``data_packaging/shared.py``) downloads with DataLad, reads the time series,
the masks and the ``cneuromod2026`` atlas, joins each TR to the run's event table and writes the
files; one small module per data set (``things.py``, ``movie10.py``) gives what differs.
Brain-Score hosts none of these files: a user packages them locally. See ``README.md``.
"""
from brainscore import data_builder_registry
from brainscore.data.packaging import DataBuilder

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
    access='public, no account: masks, atlas, event files and GLMsingle outputs of sub-01, 02, 03 and 06.',
    layout=_LAYOUT.format(repositories='things/`, `things.glmsingle/`, `anat.atlases/') + '.',
    needs='DataLad and git-annex 10.20230126 or later (tested: 10.20261026 and 10.20261113; from '
          '10.20261113 git-annex asks to confirm the CONP store, which the command does); h5py, nibabel, nilearn; per participant 11.32 GB to '
          'download (the trial-response file holds every session, also for a one-session part) and about 12 GB '
          'of disk; measured for sub-01 on a cloud machine: 403 s, mostly the download (24 MB/s); a '
          'rebuild from the kept download: 3 s per session.',
    writes=_FILES + '`trials.nc` (GLMsingle trial responses, region voxels from the masks), `events.csv` (the '
           'event files, each trial with its stimulus identifier), `stimuli.csv` (identifiers only); and '
           '`glmsingle/` (the released noise-ceiling map). ' + _PART,
    build=_COMMAND.format(dataset='things', form='glmsingle'),
    fetches=True,
)

for _dataset, _participants in (('things', 'sub-01, 02, 03 and 06'), ('movie10', 'sub-01, 02, 03, 05 and 06')):
    data_builder_registry[f'cneuromod-{_dataset}-timeseries'] = lambda dataset=_dataset, who=_participants: DataBuilder(
        source=_SOURCE,
        access=f'not public yet: the {dataset}.timeseries files of {who} are on the CNeuroMod S3 store only, '
               'which needs a key from CNeuroMod (https://docs.cneuromod.ca/contents/access.html); the same '
               'command works once they are on the public store.',
        layout=_LAYOUT.format(repositories=f'{dataset}/`, `{dataset}.timeseries/`, `anat.atlases/') + '.',
        needs='DataLad and git-annex; h5py, nibabel, nilearn; the cneuromod2026 form is 0.2 GB per '
              'participant (from the annex keys); time not measured.',
        writes=_FILES + '`timeseries_cneuromod2026.nc` (one row per TR: participant, session, task, run, tr, '
               'time, the event and stimulus identifier of the TR), `events.csv`, `stimuli.csv` (identifiers '
               'only), and `sub-XX/atlas/parcels.csv` (the parcels of the participant\'s own atlas). ' + _PART,
        build=_COMMAND.format(dataset=dataset, form='cneuromod2026'),
        fetches=True,
    )
