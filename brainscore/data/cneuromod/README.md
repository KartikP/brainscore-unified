# CNeuroMod data (Courtois NeuroMod)

One data plugin for all CNeuroMod data sets. A user packages the data locally from the source
with one command; Brain-Score hosts none of these files (the repository READMEs forbid
redistribution outside the CNeuroMod data bank).

```python
from brainscore.data import package_data
package_data('cneuromod-things', part='sub-01/ses-01')   # one session; part='sub-01': one participant
```

The command states what it needs (DataLad, git-annex, the download size, the free disk) before
it downloads. The DataLad clones stay in `$BRAINSCORE_DATA_HOME/sources/cneuromod`, so a
stopped run resumes with the same command. It prints the sha1 of each written file; compare it
with the checksums below. Participant sub-04 is restricted and never packaged.

## What is written

- `atlas/`: a copy of the group `cneuromod2026` atlas (MNI152NLin2009cAsym, 1 mm) with its label
  table, and `parcels.csv`, the parcels present in it.
- `sub-XX/masks/`: copies of the participant's region masks from `anat.atlases` (pRF
  `retinoVisionNpythy`, fLoc `fLocVisionTask`, `fLocVisionKanwisher`, `langToneva`, `yeo7networks`),
  native T1w space, on the grid of the GLMsingle outputs.
- `ses-NN/events.csv`: every column of the run event files, with `participant`, `session`,
  `task`, `run`, `event` (row in the run) and `stimulus_id`.
- `ses-NN/stimuli.csv`: one row per stimulus: identifiers and the place to get the stimulus
  (`source`). No stimulus file is copied.
- THINGS: `ses-NN/trials.nc`, the GLMsingle trial responses (`things.glmsingle`, model
  FITHRF_GLMDENOISE_RR, betas z-scored per voxel over all trials), dims `presentation` (one per
  trial: participant, session, task, run, trial, onset, duration, stimulus_id, repetition,
  condition, response, ...) x `neuroid` (one per region mask and voxel: region, atlas, voxel,
  i, j, k); and `glmsingle/`, the released noise-ceiling map.
- Time series (when public): `sub-XX/atlas/parcels.csv`, the parcels of the participant's own
  atlas in the column order of its time series, and `ses-NN/timeseries_cneuromod2026.nc`, one row per TR (participant,
  session, task, run, tr, time = tr x 1.49 s, event, stimulus_id, time_in_event) x one column per
  parcel. The extraction (`cneuromod_extract_tseries`, denoise `simple+gsr`) drops no volume, so
  TR n starts at n x 1.49 s; a TR takes the event whose [onset, onset + duration) holds its start.

## The atlas has 1134 parcels, its label table 1178 rows

The label table lists 1000 Schaefer, 50 Tian and 128 Nettekoven labels (1178). The group atlas file
(`res-01`) holds 1134 of them: 44 labels are absent, 4 cortical (366, 471, 476, 985: two
Default PCC and one Control cingulate parcel on the left, one Default PCC on the right) and 40
cerebellar (`Cereb-*`), so 996 + 50 + 88 = 1134, as the documentation states. Read from the files
on 2026-11-24. Whether the grey-matter mask of a participant removes more parcels is not known:
the per-participant atlases are on the S3 store only.

## Checksums (sub-01, session 1; `--part sub-01/ses-01`)

Built twice from the public files on a cloud machine on 2026-11-24, equal bytes both times. Every
file's sha1 is in `data_packaging/expected.json`; the command compares each written file with it.

| File | sha1 |
| --- | --- |
| `sub-01/ses-01/trials.nc` (180 trials x 17290 region voxels) | `20f7fd2cf977fc24cde56f878750d60380069e0e` |
| `sub-01/ses-01/events.csv` | `6ccb86ffc4ba1808b15e0c19e104b47e4bb6fbf0` |
| `sub-01/ses-01/stimuli.csv` (120 images) | `82bac6c9402245711e4a5c838fd24e6c694c56dd` |
| `atlas/parcels.csv` | `73255080a671a6d875a7808ac0c2ce43ee1e824e` |

Time and size: 11.32 GB downloaded (one trial-response file per participant holds all 36
sessions), 403 s for the first build, 3 s for a rebuild from the kept download; 16 MB written.

## Stimuli and terms

- THINGS images: https://osf.io/jum2f, academic use only (a password zip); get them there and
  match them by `stimulus_id` (the image file name) or `things_image_nr`.
- movie10 films: copyright kept by their owners.

## Add a data set

Write `data_packaging/<name>.py` with `NAME`, `REPOSITORY`, `PARTICIPANTS`, `TR_SECONDS`,
`STIMULUS_SOURCE`, `STIMULUS_COLUMNS`, `event_patterns(participant, session)` and
`run_events(clone, participant, run)` (a table with `onset`, `duration`, `stimulus_id`,
`repetition` and its own columns); then add one registry line in `__init__.py`. Nothing in
`shared.py` changes (see `movie10.py`).

## Citation

St-Laurent, M. et al. (2026). CNeuroMod-THINGS, a densely-sampled fMRI dataset for visual
neuroscience. Scientific Data 13, 141. https://doi.org/10.1038/s41597-026-06591-y

## Rebuild

### cneuromod-movie10-timeseries

- Source: Courtois NeuroMod, https://docs.cneuromod.ca (the repositories at https://github.com/courtois-neuromod); the documentation states CC0 for the public participants, and each repository README says "Redistribution of data outside of the Courtois NeuroMod data bank is prohibited": package locally, do not share the output.
- Access: not public yet: the movie10.timeseries files of sub-01, 02, 03, 05 and 06 are on the CNeuroMod S3 store only, which needs a key from CNeuroMod (https://docs.cneuromod.ca/contents/access.html); the same command works once they are on the public store.
- Needs: DataLad and git-annex; h5py, nibabel, nilearn; the cneuromod2026 form is 0.2 GB per participant (from the annex keys); time not measured.
- Command: `python -m brainscore.data package cneuromod-movie10-timeseries`. Source folder: none: the script clones the repositories with DataLad (https, no recursion) and downloads only the files it needs into `$BRAINSCORE_DATA_HOME/sources/cneuromod`, kept so that a stopped run resumes. A folder of earlier clones (`movie10/`, `movie10.timeseries/`, `anat.atlases/`) given as the source is read where it is. Output (or `--output <new folder>`): `~/brainscore-data/packaged/cneuromod-movie10-timeseries`. One part only: add `--part <part>` (a part as Writes states).
- Writes: `atlas/` (a copy of the group cneuromod2026 atlas and its labels, with `parcels.csv`, the parcels present); per participant `masks/` (copies of the region masks); per session `ses-NN/`: `timeseries_cneuromod2026.nc` (one row per TR: participant, session, task, run, tr, time, the event and stimulus identifier of the TR), `events.csv`, `stimuli.csv` (identifiers only), and `sub-XX/atlas/parcels.csv` (the parcels of the participant's own atlas). A part: `sub-XX` (one participant) or `sub-XX/ses-NN` (one session); sub-04 is restricted and never packaged.
- Check: the printed counts; no sha1 is recorded for these files

### cneuromod-things

- Source: Courtois NeuroMod, https://docs.cneuromod.ca (the repositories at https://github.com/courtois-neuromod); the documentation states CC0 for the public participants, and each repository README says "Redistribution of data outside of the Courtois NeuroMod data bank is prohibited": package locally, do not share the output. THINGS: St-Laurent et al. 2026, Scientific Data 13:141. The images are not included.
- Access: public, no account: masks, atlas, event files and GLMsingle outputs of sub-01, 02, 03 and 06.
- Needs: DataLad and git-annex 10.20230126 or later; h5py, nibabel, nilearn; per participant 11.32 GB to download (the trial-response file holds every session, also for a one-session part) and about 12 GB of disk; measured for sub-01 on a cloud machine: 403 s, mostly the download (24 MB/s); a rebuild from the kept download: 3 s per session.
- Command: `python -m brainscore.data package cneuromod-things`. Source folder: none: the script clones the repositories with DataLad (https, no recursion) and downloads only the files it needs into `$BRAINSCORE_DATA_HOME/sources/cneuromod`, kept so that a stopped run resumes. A folder of earlier clones (`things/`, `things.glmsingle/`, `anat.atlases/`) given as the source is read where it is. Output (or `--output <new folder>`): `~/brainscore-data/packaged/cneuromod-things`. One part only: add `--part <part>` (a part as Writes states).
- Writes: `atlas/` (a copy of the group cneuromod2026 atlas and its labels, with `parcels.csv`, the parcels present); per participant `masks/` (copies of the region masks); per session `ses-NN/`: `trials.nc` (GLMsingle trial responses, region voxels from the masks), `events.csv` (the event files, each trial with its stimulus identifier), `stimuli.csv` (identifiers only); and `glmsingle/` (the released noise-ceiling map). A part: `sub-XX` (one participant) or `sub-XX/ses-NN` (one session); sub-04 is restricted and never packaged.
- Check: the sha1 and exact content digest of 47 files in `data_packaging/expected.json`; each other file: its counts or byte size in `build.json`

### cneuromod-things-timeseries

- Source: Courtois NeuroMod, https://docs.cneuromod.ca (the repositories at https://github.com/courtois-neuromod); the documentation states CC0 for the public participants, and each repository README says "Redistribution of data outside of the Courtois NeuroMod data bank is prohibited": package locally, do not share the output.
- Access: not public yet: the things.timeseries files of sub-01, 02, 03 and 06 are on the CNeuroMod S3 store only, which needs a key from CNeuroMod (https://docs.cneuromod.ca/contents/access.html); the same command works once they are on the public store.
- Needs: DataLad and git-annex; h5py, nibabel, nilearn; the cneuromod2026 form is 0.2 GB per participant (from the annex keys); time not measured.
- Command: `python -m brainscore.data package cneuromod-things-timeseries`. Source folder: none: the script clones the repositories with DataLad (https, no recursion) and downloads only the files it needs into `$BRAINSCORE_DATA_HOME/sources/cneuromod`, kept so that a stopped run resumes. A folder of earlier clones (`things/`, `things.timeseries/`, `anat.atlases/`) given as the source is read where it is. Output (or `--output <new folder>`): `~/brainscore-data/packaged/cneuromod-things-timeseries`. One part only: add `--part <part>` (a part as Writes states).
- Writes: `atlas/` (a copy of the group cneuromod2026 atlas and its labels, with `parcels.csv`, the parcels present); per participant `masks/` (copies of the region masks); per session `ses-NN/`: `timeseries_cneuromod2026.nc` (one row per TR: participant, session, task, run, tr, time, the event and stimulus identifier of the TR), `events.csv`, `stimuli.csv` (identifiers only), and `sub-XX/atlas/parcels.csv` (the parcels of the participant's own atlas). A part: `sub-XX` (one participant) or `sub-XX/ses-NN` (one session); sub-04 is restricted and never packaged.
- Check: the printed counts; no sha1 is recorded for these files
