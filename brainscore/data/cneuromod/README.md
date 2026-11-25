# CNeuroMod data (Courtois NeuroMod)

One data plugin for all CNeuroMod data sets. A user packages the data locally from the source
with one command; Brain-Score hosts none of these files (the repository READMEs forbid
redistribution outside the CNeuroMod data bank).

```python
from brainscore.data import package_data
package_data('cneuromod-things', part='sub-01/ses-01')   # one session; part='sub-01': one participant
```

The command states what it needs (DataLad, git-annex, the download size, the free disk) before
it downloads. Tested with git-annex 10.20261026 and 10.20261113. From 10.20261113 git-annex does not
use a special remote until it is confirmed; the command confirms the public store
(`git-annex verifyremote conp-ria-storage-http --safe <id>`) only when git-annex asks and the store
is at `https://sftp.conp.ca/users/cneuromod/`. The DataLad clones stay in `$BRAINSCORE_DATA_HOME/sources/cneuromod`, so a
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
  condition, response, ...) x `neuroid` (one per region and voxel: region, atlas, voxel,
  i, j, k); and `glmsingle/`, the released noise-ceiling map. Select a region with
  `trials.sel(region='FFA')` (a `region` coordinate, as in the vision assemblies). Each region has
  one mask set: the participant's own pRF (`retinoVisionNpythy`) and fLoc localizer
  (`fLocVisionTask`) masks; the Kanwisher group parcels (`fLocVisionKanwisher`) only for a region
  without an own mask (sub-06). `atlas` names the source. A voxel in two masks (for example EBA
  and hV4) is one row per region with the same `voxel`, `i`, `j`, `k`.
  The rows with `region='cortex'`, `atlas='aparcaseg'` are the cortical voxels of the fMRIPrep
  segmentation (`things.fmriprep`, `space-T1w_desc-aparcaseg_dseg` of the first THINGS run,
  labels 1000 to 2999), on the GLMsingle grid without resampling (sub-01: 70900 voxels).
  The `repetition` column is the label of the released event files, unchanged. It is not always
  the viewing order: in out-of-order sessions it is 4 or 5 (18 trials of sub-01, sessions 14-15),
  and it counts trials without a response. For the viewing order use
  `data_packaging.things.answered_viewing(stimulus_id, response)`: the place of each trial among
  the answered trials of its image, in time order (0: no response). The Fig. 2a check uses it.
- Time series: `sub-XX/atlas/parcels.csv`, the parcels of the participant's own
  atlas in the column order of its time series, and `ses-NN/timeseries_cneuromod2026.nc`, one row per TR (participant,
  session, task, run, tr, time = tr x 1.49 s, event, stimulus_id, time_in_event) x one column per
  parcel. The extraction (`cneuromod_extract_tseries`, denoise `simple+gsr`) leaves out the volumes
  that fMRIPrep marks non-steady (`non_steady_state_outlier*`; for example sub-01 movie10
  `ses-006_task-wolf16`: 405 of 406 volumes); `tr` is the BOLD volume of a row and it starts at
  tr x 1.49 s; a row takes the event whose [onset, onset + duration) holds its start. The builder
  stores the kept volumes on each run; a released file has no such record and is read as every
  volume kept, so a run of a released file with a non-steady volume is timed wrong.

## Time series made from the fMRIPrep outputs

The released time series (`things.timeseries`, `movie10.timeseries`) need a key from CNeuroMod.
`package_data('cneuromod-things-timeseries', part='sub-01/ses-02')` (or
`cneuromod-movie10-timeseries`) makes them instead from public files, as the release made them
(`data_packaging/extraction.py`):

- The commits: `cneuromod_extract_tseries` at `ae310a53` (the code of the release; another commit
  is refused) and `<data set>.fmriprep` at the commit that `<data set>.timeseries` names (THINGS
  `5155ba0d`, movie10 `638de599`); `anat.atlases` at the commit the code names (`0b78173e`). They
  are printed at the start.
- The inputs, per run of the session: the BOLD, its brain mask and its confounds in the space of the
  form (cneuromod2026: MNI152NLin2009cAsym; voxel_native: T1w); and the atlas files that the code's
  configuration names (`schaefer1000Tian50Nette128`, `gm_epi`).
- The code runs in its own environment (`sources/cneuromod/extraction/env`, the pinned versions of
  its `requirements.txt`; pip cache and temporary files beside it), never in the calling one, from
  its own folder, with denoise `simple+gsr`, as the release.
- One session at a time; after each session every run is checked (rows = BOLD volumes, columns =
  the parcels of the participant's atlas or the voxels of its mask), because the code catches the
  error of a run and goes on. A failed run raises with its name. Then the session's BOLD files are
  dropped. A session whose runs are all in the HDF5 file is skipped, so the same command resumes.
- The files get the released names (`atlas-cneuromod26_desc-1134Parcels`, `desc-voxelwise`), and the
  reader above reads them. The voxel form: `python -m brainscore.data.cneuromod.data_packaging.package
  --dataset things --form voxel_native --source <absent folder> --out <new folder> --part sub-01`.
- Checked on a cloud machine on 2026-11-25 (THINGS sub-01 ses-02, the same code outside this
  command): 190 rows per run, 1134 parcels, 102959 voxels; the participant's atlas and mask have the
  released MD5s.

## Licences

- `cneuromod_extract_tseries`: MIT (its `LICENSE`, "Copyright (c) 2024, CNeuroMod"). It is not
  copied; it is cloned on the user's machine.
- The data: open. The CNeuroMod documentation states CC0 for the public participants; the
  `License` field of `anat.atlases` and `movie10.fmriprep` points to the data transfer agreement
  page; each repository README says "Redistribution of data outside of the Courtois NeuroMod data
  bank is prohibited". Until CNeuroMod states which applies, package locally and do not share the
  output.

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
On 2026-11-24 `trials.nc` changed (one mask set per region, the cortex rows added): rebuilt from the
kept download, every other file had the recorded bytes; the earlier `trials.nc` sha1 was
`20f7fd2cf977fc24cde56f878750d60380069e0e` (17290 region rows).

| File | sha1 |
| --- | --- |
| `sub-01/ses-01/trials.nc` (180 trials x 84543 voxels: 13643 region rows, 70900 cortex rows) | `b0e5f02a398199191f9bd12195709d4358826b8a` |
| `sub-01/ses-01/events.csv` | `6ccb86ffc4ba1808b15e0c19e104b47e4bb6fbf0` |
| `sub-01/ses-01/stimuli.csv` (120 images) | `82bac6c9402245711e4a5c838fd24e6c694c56dd` |
| `atlas/parcels.csv` | `73255080a671a6d875a7808ac0c2ce43ee1e824e` |

Time and size: 11.32 GB downloaded (one trial-response file per participant holds all 36
sessions), 403 s for the first build, 3 s for a rebuild from the kept download; 69 MB written
for one session; 4.2 GB for the 36 sessions of sub-01 (19 s from the kept download).

## Validation: the paper's noise ceiling (Fig. 2a)

`validation/fig2a_noise_ceiling.py` computes the voxelwise noise ceiling of St-Laurent et al.
(2026, p.5-6) from the packaged `trials.nc` files only and draws it per region and participant.
On the full packaging (2026-11-24) it equals the released noise-ceiling map on every region
voxel (largest difference 0.0011 percentage points; the betas are stored as float32) and gives
the paper's image counts and maxima for sub-01, 02 and 03. sub-06 has no pRF masks; its FFA,
EBA and PPA are the Kanwisher group parcels. The figure and tables are in `validation/`.
After the change to one mask set per region (2026-11-24) the check on sub-01 gives the same image
count (3247), maximum (56.51) and region medians; `voxels` in `fig2a_voxels.csv` still counts the
earlier packaging, with the group parcels beside the own masks (sub-01 now: 12417 unique voxels).

## Stimuli and terms

- THINGS images: https://osf.io/jum2f, academic use only (a password zip); get them there and
  match them by `stimulus_id` (the image file name) or `things_image_nr`. For the THINGS benchmarks
  put the unzipped `object_images` folder at `$BRAINSCORE_DATA_HOME/things-images`;
  `load_stimulus_set('cneuromod-things')` checks each of the 4320 images of the sessions against its
  sha1 in `things_images_sha1.csv` (from `images_THINGS.zip`, read 2026-11-24) and names a file that
  is missing or differs.
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

- Source: Courtois NeuroMod, https://docs.cneuromod.ca (the repositories at https://github.com/courtois-neuromod); the documentation states CC0 for the public participants, and each repository README says "Redistribution of data outside of the Courtois NeuroMod data bank is prohibited": package locally, do not share the output. Made with CNeuroMod's extraction code (https://github.com/courtois-neuromod/cneuromod_extract_tseries, MIT licence) at ae310a53, the commit of the released time series.
- Access: public, no account: the time series of sub-01, 02, 03, 05 and 06 are made on this machine from the public fMRIPrep outputs (movie10.fmriprep) and anat.atlases, at the commits that movie10.timeseries names (its released files need a key from CNeuroMod).
- Needs: DataLad and git-annex 10.20230126 or later (checked first); the venv module of the calling Python: the command makes an environment in `sources/cneuromod/extraction/env` with the pinned versions of the code and never installs into the calling environment; one session of fMRIPrep outputs on the disk at a time (dropped after its check). Measured for THINGS sub-01 ses-02 on a cloud machine: 4.76 GB of input, 131 s, 4.2 GB of memory. THINGS: about 169 GB of input and 2 h per participant; movie10: 475 GB for the 5 participants (estimates from the file sizes).
- Command: `python -m brainscore.data package cneuromod-movie10-timeseries`. Source folder: none: the script clones the repositories with DataLad (https, no recursion) and downloads only the files it needs into `$BRAINSCORE_DATA_HOME/sources/cneuromod`, kept so that a stopped run resumes. A folder of earlier clones (`movie10/`, `movie10.timeseries/`, `anat.atlases/`) given as the source is read where it is (and its released time series files are then read, not made). Output (or `--output <new folder>`): `~/brainscore-data/packaged/cneuromod-movie10-timeseries`. One part only: add `--part <part>` (a part as Writes states).
- Writes: `atlas/` (a copy of the group cneuromod2026 atlas and its labels, with `parcels.csv`, the parcels present); per participant `masks/` (copies of the region masks); per session `ses-NN/`: `timeseries_cneuromod2026.nc` (one row per TR: participant, session, task, run, tr, time, the event and stimulus identifier of the TR), `events.csv`, `stimuli.csv` (identifiers only), and `sub-XX/atlas/parcels.csv` (the parcels of the participant's own atlas). A part: `sub-XX` (one participant) or `sub-XX/ses-NN` (one session); sub-04 is restricted and never packaged.
- Check: the printed counts; no sha1 is recorded for these files

### cneuromod-things

- Source: Courtois NeuroMod, https://docs.cneuromod.ca (the repositories at https://github.com/courtois-neuromod); the documentation states CC0 for the public participants, and each repository README says "Redistribution of data outside of the Courtois NeuroMod data bank is prohibited": package locally, do not share the output. THINGS: St-Laurent et al. 2026, Scientific Data 13:141. The images are not included.
- Access: public, no account: masks, atlas, event files, GLMsingle outputs and the fMRIPrep segmentation of sub-01, 02, 03 and 06.
- Needs: DataLad and git-annex 10.20230126 or later (tested: 10.20261026 and 10.20261113; from 10.20261113 git-annex asks to confirm the CONP store, which the command does); h5py, nibabel, nilearn; per participant 11.32 GB to download (the trial-response file holds every session, also for a one-session part) and about 16 GB of disk; measured for sub-01 on a cloud machine: 403 s, mostly the download (24 MB/s); a rebuild from the kept download: 3 s per session.
- Command: `python -m brainscore.data package cneuromod-things`. Source folder: none: the script clones the repositories with DataLad (https, no recursion) and downloads only the files it needs into `$BRAINSCORE_DATA_HOME/sources/cneuromod`, kept so that a stopped run resumes. A folder of earlier clones (`things/`, `things.glmsingle/`, `things.fmriprep/`, `anat.atlases/`) given as the source is read where it is. Output (or `--output <new folder>`): `~/brainscore-data/packaged/cneuromod-things`. One part only: add `--part <part>` (a part as Writes states).
- Writes: `atlas/` (a copy of the group cneuromod2026 atlas and its labels, with `parcels.csv`, the parcels present); per participant `masks/` (copies of the region masks); per session `ses-NN/`: `trials.nc` (GLMsingle trial responses: the region voxels, one mask set per region, and the cortex voxels of the fMRIPrep segmentation, `region='cortex'`), `events.csv` (the event files, each trial with its stimulus identifier), `stimuli.csv` (identifiers only); and `glmsingle/` (the released noise-ceiling map). A part: `sub-XX` (one participant) or `sub-XX/ses-NN` (one session); sub-04 is restricted and never packaged.
- Check: the sha1 and exact content digest of 47 files in `data_packaging/expected.json`; each other file: its counts or byte size in `build.json`

### cneuromod-things-timeseries

- Source: Courtois NeuroMod, https://docs.cneuromod.ca (the repositories at https://github.com/courtois-neuromod); the documentation states CC0 for the public participants, and each repository README says "Redistribution of data outside of the Courtois NeuroMod data bank is prohibited": package locally, do not share the output. Made with CNeuroMod's extraction code (https://github.com/courtois-neuromod/cneuromod_extract_tseries, MIT licence) at ae310a53, the commit of the released time series.
- Access: public, no account: the time series of sub-01, 02, 03 and 06 are made on this machine from the public fMRIPrep outputs (things.fmriprep) and anat.atlases, at the commits that things.timeseries names (its released files need a key from CNeuroMod).
- Needs: DataLad and git-annex 10.20230126 or later (checked first); the venv module of the calling Python: the command makes an environment in `sources/cneuromod/extraction/env` with the pinned versions of the code and never installs into the calling environment; one session of fMRIPrep outputs on the disk at a time (dropped after its check). Measured for THINGS sub-01 ses-02 on a cloud machine: 4.76 GB of input, 131 s, 4.2 GB of memory. THINGS: about 169 GB of input and 2 h per participant; movie10: 475 GB for the 5 participants (estimates from the file sizes).
- Command: `python -m brainscore.data package cneuromod-things-timeseries`. Source folder: none: the script clones the repositories with DataLad (https, no recursion) and downloads only the files it needs into `$BRAINSCORE_DATA_HOME/sources/cneuromod`, kept so that a stopped run resumes. A folder of earlier clones (`things/`, `things.timeseries/`, `anat.atlases/`) given as the source is read where it is (and its released time series files are then read, not made). Output (or `--output <new folder>`): `~/brainscore-data/packaged/cneuromod-things-timeseries`. One part only: add `--part <part>` (a part as Writes states).
- Writes: `atlas/` (a copy of the group cneuromod2026 atlas and its labels, with `parcels.csv`, the parcels present); per participant `masks/` (copies of the region masks); per session `ses-NN/`: `timeseries_cneuromod2026.nc` (one row per TR: participant, session, task, run, tr, time, the event and stimulus identifier of the TR), `events.csv`, `stimuli.csv` (identifiers only), and `sub-XX/atlas/parcels.csv` (the parcels of the participant's own atlas). A part: `sub-XX` (one participant) or `sub-XX/ses-NN` (one session); sub-04 is restricted and never packaged.
- Check: the printed counts; no sha1 is recorded for these files
