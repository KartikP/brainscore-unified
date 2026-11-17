# Prepare Algonauts2025 data locally

Two folders are involved, both under `$BRAINSCORE_DATA_HOME` (default `~/brainscore-data`):
`sources/algonauts2025` holds the native competition files, which the packaging
command below reads, and `algonauts2025-prepared` holds the converted files that the
benchmarks read. A folder on another drive: put a link at its place, or give the
source to the command with `--source`.

Obtain the competition files through the
[authors' DataLad instructions](https://github.com/courtois-neuromod/algonauts_2025.competitors).
The page's Terms section states that the data is shared under CC0; the download needs no account.
Keep the native `fmri/` and `stimuli/` directories together.

The build reads per-participant HDF5 responses and movie/transcript metadata. It
writes per-TR NetCDF assemblies, stimulus CSVs and the media of the registered
benchmarks. See Rebuild below for its needs.

```bash
# Build from ~/brainscore-data/sources/algonauts2025 into the folder the benchmarks read
python -m brainscore.data package algonauts2025
```

The benchmark's message for missing files prints this command and the folder.

To convert fewer participants or splits, run the script it wraps,
`brainscore.data.algonauts2025.data_packaging.prepare_assembly`, with `--subjects`
and `--splits`.

Movie paths in the CSV point to your native files; keep those files available.
Check loading without a model:

```python
import brainscore

# The film table and the fMRI of participant 1, from the prepared folder
stimuli = brainscore.load_stimulus_set("Algonauts2025-friends")
responses = brainscore.load_dataset("Algonauts2025-friends-sub01")
print(len(stimuli), responses.sizes)
```

## Prepare the stimuli

The benchmarks show one file per scan (TR = 1.49 s), cut from the films. The
films are copyrighted (Friends and the Movie10 films); Brain-Score cannot
redistribute them or files cut from them. Obtain them with the competition
dataset above, under its access terms, and cut them on your machine.
`python -m brainscore.data package algonauts2025` already cuts the media of the
registered benchmarks (frames for every participant, clips and sound for
participant 1). To cut other media or films, run the script it calls:

```bash
# Cut the frames of the films that participant 1 saw
python -m brainscore.data.algonauts2025.data_packaging.prepare_stimuli \
  --split friends --participant 1 --media frames
```

- It needs: the output of `prepare_assembly` for that split and participant
  (the assembly gives the number of scans of each film), the native film files
  that the film table points to, `ffmpeg` and `ffprobe` on the `PATH`.
- `--media frames` writes `frames/<film>/<t + 1>.jpg`: the frame at the middle
  of scan `t`, 224 x 224 (the default benchmarks, `vision:image`).
- `--media clips` writes `clips/<film>/<t>.mp4`: the 1.49 s of scan `t`,
  224 x 224, 30 frames, no sound (the `-video` benchmark).
- `--media clips sound` writes the clips and `sound/<film>/<t>.wav`: the sound
  of the same 1.49 s, 16 kHz, mono, 23840 samples (the `-video-audio`
  benchmark).
- `--movies friends_s01e01a ...` cuts only the named films (for a benchmark
  built with `movies=[...]`). `--prepared-root` names another
  prepared folder than `$BRAINSCORE_DATA_HOME/algonauts2025-prepared`; `--workers` sets the number of ffmpeg
  processes (default 8).

The files are kept per film, so the participants share them. A second run cuts
only what is missing. Where a film ends before the last scan of its run, the
missing files are copies of the last whole file. If ffmpeg cannot cut a film,
the script raises `FileNotFoundError`.

Time and size for the training split of participant 1 (162,671 scans): the
frames took about 12 minutes with 8 workers on a cloud machine, about 10 KB
each (about 1.6 GB). The sound is about 48 KB per scan (about 7.8 GB, computed
from the format). The time of the clips and the sound, and the size of the
clips, were not measured.

Other participants are 2, 3 and 5. Splits `friends_s7` and `ood` contain held-out
response placeholders, not measured responses available for local scoring.
Do not treat those NaNs as an acquired human dataset.

A fresh build has not been compared with a recorded count or checksum.

## Rebuild

### algonauts2025

- Source: the Algonauts 2025 competition data (Courtois NeuroMod): https://github.com/courtois-neuromod/algonauts_2025.competitors; its Terms section states that the data is shared under CC0.
- Access: public, no account: the DataLad instructions of the page above download it with no form and no account (140.6 GB, about 40 minutes measured).
- Needs: DataLad and git-annex (from conda-forge on Ubuntu 22.04), h5py, ffmpeg and ffprobe; 140.6 GB in; out 17.5 GB: 3.36 GB of assemblies and tables, 4.78 GB of clips, 1.51 GB of frames and 7.77 GB of sound; the packaging took 110 minutes (measured once).
- Command: `python -m brainscore.data package algonauts2025` with the source in `~/brainscore-data/sources/algonauts2025` (or `--source <folder>`): the DataLad dataset as it installs: `fmri/sub-0X/...` (HDF5 responses and `target_sample_number/`) and `stimuli/` (`movies/`, `transcripts/`), kept together. Output (or `--output <new folder>`): the folder the benchmarks read, `$BRAINSCORE_DATA_HOME/algonauts2025-prepared` (`~/brainscore-data/algonauts2025-prepared` by default).
- Writes: per participant 1, 2, 3, 5 and split: `algonauts2025_<split>_sub0X.nc` (friends: the fMRI; friends_s7 and ood: placeholders with no fMRI), `algonauts2025_stim_<split>.csv`, and one file per scan of the training films: `frames/` (all participants), `clips/` and `sound/` (participant 1), the media of the six registered benchmarks.
- Check: the printed counts; no sha1 is recorded for these files
