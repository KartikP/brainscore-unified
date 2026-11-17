# Load and prepare Lahner2024 data

The registered BOLDMoments stimulus and Lahner2024 response loaders download
versioned assets from Brain-Score storage. Expected hashes and versions are
in `__init__.py`.

```python
import brainscore

# Download the hosted clips and GLM responses once, checked by sha1
stimuli = brainscore.load_stimulus_set("BOLDMoments")
responses = brainscore.load_dataset("Lahner2024-fMRI")
print(len(stimuli), responses.sizes)
```

These are separate assets: movie access does not imply permission to redistribute
human responses, or vice versa. Confirm the applicable terms before sharing.

## Frames and sound: local assets

The benchmarks read two folders that you prepare once from the hosted clips.
`python -m brainscore.data` shows whether they are found.

| Asset | Folder (under `$BRAINSCORE_DATA_HOME`, default `~/brainscore-data`) | Prepare with |
| --- | --- | --- |
| `lahner2024-frames` | `lahner2024-frames` | `python -m brainscore.data.lahner2024.data_packaging.prepare_frames` |
| `lahner2024-sound` | `lahner2024-sound` | `python -m brainscore.data.lahner2024.data_packaging.prepare_audio_tracks` (needs ffmpeg) |

These scripts fill the folder in place and keep the files that exist;
`python -m brainscore.data package lahner2024-frames` (or `-sound`) builds a new
folder instead and prints the counts (see Rebuild below).

A subject that takes only still images reads the frames; a subject that takes
sound reads the sound tracks. A missing file stops the run with the command
that prepares it.

A Lahner2024 benchmark takes no folder argument: for every variant, it reads the
frames at `$BRAINSCORE_DATA_HOME/lahner2024-frames` and the sound at
`$BRAINSCORE_DATA_HOME/lahner2024-sound`.

The GLM-beta assembly and the time-resolved assembly are different targets,
not interchangeable.

## Rebuild

### destrieux-atlas

- Source: the Destrieux 2009 surface atlas on fsaverage5, as nilearn publishes it (`nilearn.datasets.fetch_atlas_surf_destrieux`); its own terms.
- Access: public, no account.
- Needs: nilearn; a download made by nilearn, size not measured.
- Command: none: nothing to package.
- Writes: Nilearn fetches the atlas on first use into `$BRAINSCORE_DATA_HOME/destrieux-atlas`, where the auditory-ROI benchmark reads it as published.
- Check: nothing to compare.

### lahner2024

- Source: BOLD Moments (Lahner et al. 2024), OpenNeuro ds005165 v1.0.4, https://openneuro.org/datasets/ds005165; licence as OpenNeuro states it for ds005165 (not recorded in this plugin). The hosted files (BOLDMoments stimuli and Lahner2024-fMRI GLM responses) came from vision pull request #1249 (YingtianDt), whose packaging code is not in any repository read so far.
- Access: public, no account (the OpenNeuro data); the hosted files download with no account.
- Needs: not known.
- Command: none yet. No script builds the GLM responses and the BOLDMoments stimuli from OpenNeuro. A maintainer needs: the packaging code of vision pull request #1249, or a new script from the ds005165 GLM betas and clips; the rule that keeps 1,026 of the 1,102 videos (933 training-set, 93 test-set), which is not written anywhere found; then a cloud run that compares the rebuilt files with the registered sha1 or, for the netCDF file, with the hosted values.
- Writes: `stimulus_BOLDMoments.csv`, `stimulus_BOLDMoments.zip`, `assy_Lahner2024-fMRI.nc`. Use the hosted files: `brainscore.load_dataset` fetches them.
- Check: sha1 of `stimulus_BOLDMoments.csv` is `0b27388f5898c908f58cd1f21f8f5cb3eda8536e`; sha1 of `stimulus_BOLDMoments.zip` is `dc9c3bf631632cd433d02f2f1847fd33c01ae0b3`; sha1 of `assy_Lahner2024-fMRI.nc` is `2c7f1d2e5724b8cc3c5cf47986e956c4f13001e4`

### lahner2024-frames

- Source: https://openneuro.org/datasets/ds005165 (BOLD Moments; the clips are the hosted stimulus set BOLDMoments, which brainscore.load_stimulus_set fetches); licence as OpenNeuro states it for ds005165 (not recorded in this plugin).
- Access: public, no account: the hosted clips download with no account.
- Needs: opencv-python and Pillow; the hosted clips (fetched once into the brainio cache; size not measured); disk for the frames and time not measured.
- Command: `python -m brainscore.data package lahner2024-frames`. Source folder: none: the script reads the hosted clips through brainscore.load_stimulus_set. Output (or `--output <new folder>`): the folder the benchmarks read, `$BRAINSCORE_DATA_HOME/lahner2024-frames` (`~/brainscore-data/lahner2024-frames` by default).
- Writes: one PNG per clip and time (500, 1500, 2500 ms): `<clip>_t<ms>.png`.
- Check: the printed counts; no sha1 is recorded for these files

### lahner2024-motion

- Source: BOLD Moments (Lahner et al. 2024), OpenNeuro ds005165 v1.0.4, https://openneuro.org/datasets/ds005165; licence as OpenNeuro states it for ds005165 (not recorded in this plugin). The fMRIPrep derivatives `derivatives/versionB/fmriprep` and the raw `events.tsv` files, read from s3://openneuro.org/ds005165 with no account.
- Access: public, no account.
- Needs: boto3; the confounds files of the 520 runs (size and time not measured).
- Command: `python -m brainscore.data package lahner2024-motion`. Source folder: none needed: the script lists and downloads the fMRIPrep confounds files itself into a private work folder (deleted after the build). Output (or `--output <new folder>`): each identical file at `$BRAINIO_HOME/<sha1>/<file name>` (`~/.brainio` by default), where the loader reads it; the build record in `$BRAINIO_HOME/packaged/lahner2024-motion`.
- Writes: `Lahner2024-fMRI-timeresolved-motion.csv`.
- Check: sha1 of `Lahner2024-fMRI-timeresolved-motion.csv` is `827130c78f6b706b52f6d0d28962eb2301bee00b`

### lahner2024-sound

- Source: https://openneuro.org/datasets/ds005165 (BOLD Moments; the clips are the hosted stimulus set BOLDMoments, which brainscore.load_stimulus_set fetches); licence as OpenNeuro states it for ds005165 (not recorded in this plugin).
- Access: public, no account: the hosted clips download with no account.
- Needs: ffmpeg and ffprobe on the PATH; the hosted clips (fetched once into the brainio cache; size not measured); about 100 MB of sound (3 s at 16 kHz, 16 bit, for each of 1,026 clips, computed); time not measured.
- Command: `python -m brainscore.data package lahner2024-sound`. Source folder: none: the script reads the hosted clips through brainscore.load_stimulus_set. Output (or `--output <new folder>`): the folder the benchmarks read, `$BRAINSCORE_DATA_HOME/lahner2024-sound` (`~/brainscore-data/lahner2024-sound` by default).
- Writes: one 16 kHz mono WAV per clip (`<clip>.wav`; a silent file for a clip with no sound) and `manifest.json`.
- Check: the printed counts; no sha1 is recorded for these files

### lahner2024-timeresolved

- Source: BOLD Moments (Lahner et al. 2024), OpenNeuro ds005165 v1.0.4, https://openneuro.org/datasets/ds005165; licence as OpenNeuro states it for ds005165 (not recorded in this plugin). The fMRIPrep derivatives `derivatives/versionB/fmriprep` and the raw `events.tsv` files, read from s3://openneuro.org/ds005165 with no account.
- Access: public, no account.
- Needs: the `aws` command, nibabel and nilearn; about 20 GB of disk at a time (about 200 GB pass through); hours.
- Command: `python -m brainscore.data package lahner2024-timeresolved`. Source folder: none needed: the script lists and downloads the files itself into a private work folder (deleted after the build). A local copy of ds005165 in the source folder, laid out as the bucket (`derivatives/versionB/fmriprep/sub-XX/...`, `sub-XX/ses-XX/func/*_events.tsv`), is read where it is and never changed. Output (or `--output <new folder>`): each identical file at `$BRAINIO_HOME/<sha1>/<file name>` (`~/.brainio` by default), where the loader reads it; the build record in `$BRAINIO_HOME/packaged/lahner2024-timeresolved`.
- Writes: `assy_Lahner2024-fMRI-timeresolved.nc` and `Lahner2024-fMRI-timeresolved-events.csv`.
- Check: sha1 of `assy_Lahner2024-fMRI-timeresolved.nc` is `4d06589dde6dddf273a489a64da1337940d5fafd`; sha1 of `Lahner2024-fMRI-timeresolved-events.csv` is `783c5b33a75f121812a49b8b0a7639b9ed07c6a3`
