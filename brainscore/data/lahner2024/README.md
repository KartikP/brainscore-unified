# Load and prepare Lahner2024 data

The registered BOLDMoments stimulus and Lahner2024 response loaders download
versioned assets from Brain-Score storage. Expected hashes and versions are
in `__init__.py`.

```python
import brainscore

stimuli = brainscore.load_stimulus_set("BOLDMoments")
responses = brainscore.load_dataset("Lahner2024-fMRI")
print(len(stimuli), responses.sizes)
```

These are separate assets: movie access does not imply permission to redistribute
human responses, or vice versa. Confirm the applicable terms before sharing.

## Frames and sound: local assets

The benchmarks read two folders that you prepare once from the hosted clips.
`python -m brainscore.data` shows whether they are found.

| Asset | Folder (variable, default) | Prepare with |
| --- | --- | --- |
| `lahner2024-frames` | `BRAINSCORE_LAHNER_FRAMES_DIR`, `~/brainscore-data/lahner2024/frames` | `python -m brainscore.data.lahner2024.data_packaging.prepare_frames` |
| `lahner2024-sound` | `BRAINSCORE_LAHNER_SOUND_DIR`, `~/.brainio/lahner2024_audio_16k` | `python -m brainscore.data.lahner2024.data_packaging.prepare_audio_tracks` (needs ffmpeg) |

These scripts fill the folder in place and keep the files that exist;
`python -m brainscore.data package lahner2024-frames` (or `-sound`) builds a new
folder instead and prints the counts (see Rebuild below).

A subject that takes only still images reads the frames; a subject that takes
sound reads the sound tracks. A missing file stops the run with the command
that prepares it.

A Lahner2024 benchmark takes no folder argument: for every variant, the frames
folder is set by `BRAINSCORE_LAHNER_FRAMES_DIR` and the sound folder by
`BRAINSCORE_LAHNER_SOUND_DIR`.

The GLM-beta assembly and the time-resolved assembly are different targets,
not interchangeable.

## Rebuild

### destrieux-atlas

- Source: the Destrieux 2009 surface atlas on fsaverage5, as nilearn publishes it (`nilearn.datasets.fetch_atlas_surf_destrieux`); its own terms.
- Access: public, no account.
- Needs: nilearn.
- Command: none: nothing to package.
- Writes: Nilearn fetches the atlas on first use and the auditory-ROI benchmark reads it as published.
- Check: nothing to compare.

### lahner2024

- Source: BOLD Moments (Lahner et al. 2024), OpenNeuro ds005165 v1.0.4, https://openneuro.org/datasets/ds005165; licence as OpenNeuro states it for ds005165 (not recorded in this plugin). The hosted files (BOLDMoments stimuli and Lahner2024-fMRI GLM responses) came from vision pull request #1249 (YingtianDt), whose packaging code is not in any repository read so far.
- Access: public, no account (the OpenNeuro data); the hosted files download with no account.
- Needs: not known.
- Command: none yet. No script builds the GLM responses and the BOLDMoments stimuli from OpenNeuro. A maintainer needs: the packaging code of vision pull request #1249, or a new script from the ds005165 GLM betas and clips; the rule that keeps 1,026 of the 1,102 videos (933 training-set, 93 test-set), which is not written anywhere found; then a cloud run that compares the rebuilt files with the registered sha1 or, for the netCDF file, with the hosted values.
- Writes: `stimulus_BOLDMoments.csv`, `stimulus_BOLDMoments.zip`, `assy_Lahner2024-fMRI.nc`. Use the hosted files: `brainscore.load_dataset` fetches them.
- Check: sha1 of `stimulus_BOLDMoments.csv` is `0b27388f5898c908f58cd1f21f8f5cb3eda8536e`; sha1 of `stimulus_BOLDMoments.zip` is `dc9c3bf631632cd433d02f2f1847fd33c01ae0b3`; sha1 of `assy_Lahner2024-fMRI.nc` is `2c7f1d2e5724b8cc3c5cf47986e956c4f13001e4`

### lahner2024-frames

- Source: https://openneuro.org/datasets/ds005165 (BOLD Moments; the clips are the hosted stimulus set BOLDMoments, which brainscore.load_stimulus_set fetches)
- Access: public, no account: the hosted clips download with no account.
- Needs: opencv-python and Pillow; the hosted clips (fetched once); minutes.
- Command: `python -m brainscore.data package lahner2024-frames`
- Writes: one PNG per clip and time (500, 1500, 2500 ms): `<clip>_t<ms>.png`. Use them: set BRAINSCORE_LAHNER_FRAMES_DIR to the output folder.
- Check: the printed counts; no sha1 is recorded for these files

### lahner2024-motion

- Source: BOLD Moments (Lahner et al. 2024), OpenNeuro ds005165 v1.0.4, https://openneuro.org/datasets/ds005165; licence as OpenNeuro states it for ds005165 (not recorded in this plugin). The fMRIPrep derivatives `derivatives/versionB/fmriprep` and the raw `events.tsv` files, read from s3://openneuro.org/ds005165 with no account.
- Access: public, no account.
- Needs: boto3; a small download; minutes.
- Command: `python -m brainscore.data package lahner2024-motion`
- Writes: `Lahner2024-fMRI-timeresolved-motion.csv`. Use it: put it at `$BRAINIO_HOME/<sha1>/<file name>`.
- Check: sha1 of `Lahner2024-fMRI-timeresolved-motion.csv` is `827130c78f6b706b52f6d0d28962eb2301bee00b`

### lahner2024-sound

- Source: https://openneuro.org/datasets/ds005165 (BOLD Moments; the clips are the hosted stimulus set BOLDMoments, which brainscore.load_stimulus_set fetches)
- Access: public, no account: the hosted clips download with no account.
- Needs: ffmpeg and ffprobe on the PATH; the hosted clips (fetched once); minutes.
- Command: `python -m brainscore.data package lahner2024-sound`
- Writes: one 16 kHz mono WAV per clip (`<clip>.wav`; a silent file for a clip with no sound) and `manifest.json`. Use them: set BRAINSCORE_LAHNER_SOUND_DIR to the output folder.
- Check: the printed counts; no sha1 is recorded for these files

### lahner2024-timeresolved

- Source: BOLD Moments (Lahner et al. 2024), OpenNeuro ds005165 v1.0.4, https://openneuro.org/datasets/ds005165; licence as OpenNeuro states it for ds005165 (not recorded in this plugin). The fMRIPrep derivatives `derivatives/versionB/fmriprep` and the raw `events.tsv` files, read from s3://openneuro.org/ds005165 with no account.
- Access: public, no account.
- Needs: the `aws` command, nibabel and nilearn; about 20 GB of disk at a time (about 200 GB pass through); hours.
- Command: `python -m brainscore.data package lahner2024-timeresolved`
- Writes: `assy_Lahner2024-fMRI-timeresolved.nc` and `Lahner2024-fMRI-timeresolved-events.csv`. Use them: put each at `$BRAINIO_HOME/<sha1>/<file name>`.
- Check: sha1 of `assy_Lahner2024-fMRI-timeresolved.nc` is `4d06589dde6dddf273a489a64da1337940d5fafd`; sha1 of `Lahner2024-fMRI-timeresolved-events.csv` is `783c5b33a75f121812a49b8b0a7639b9ed07c6a3`
