# Prepare Algonauts2025 data locally

Two folders are involved. `BRAINSCORE_ALGONAUTS_ROOT` names the native competition
files (default `~/algonauts_2025`). `BRAINSCORE_ALGONAUTS_PREPARED` names the converted
files that the benchmarks read (default `~/.brainio/algonauts2025`).

Obtain the competition files through the
[authors' DataLad instructions](https://github.com/courtois-neuromod/algonauts_2025.competitors).
Follow their access terms for the movies, transcripts and fMRI separately.
Keep the native `fmri/` and `stimuli/` directories together.

The existing converter reads per-participant HDF5 responses and movie/transcript
metadata. It writes per-TR NetCDF assemblies and stimulus CSVs. It needs the
unified installation plus `h5py`; the source dataset is approximately 130 GB,
and preparing all subject/split outputs was estimated at another 25 GB. Runtime
depends on storage and scope; it has not been qualified for this release.

```bash
python -m brainscore.data.algonauts2025.data_packaging.prepare_assembly \
  --algonauts-root /path/to/native-data \
  --output-root /path/to/new-build \
  --subjects 1 \
  --splits friends

export BRAINSCORE_ALGONAUTS_PREPARED=/path/to/new-build
```

Movie paths in the CSV point to your native files; keep those files available.
Check loading without a model:

```python
import brainscore

stimuli = brainscore.load_stimulus_set("Algonauts2025-friends")
responses = brainscore.load_dataset("Algonauts2025-friends-sub01")
print(len(stimuli), responses.sizes)
```

## Prepare the stimuli

The benchmarks show one file per scan (TR = 1.49 s), cut from the films. The
films are copyrighted (Friends and the Movie10 films); Brain-Score cannot
redistribute them or files cut from them. Obtain them with the competition
dataset above, under its access terms, and cut them on your machine:

```bash
export BRAINSCORE_ALGONAUTS_PREPARED=/path/to/new-build
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
  built with `movies=[...]`). `--prepared-root` overrides
  `BRAINSCORE_ALGONAUTS_PREPARED`; `--workers` sets the number of ffmpeg
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

This converter is not registered with `brainscore.data prepare`. The full
fresh-build validation, content manifest and shared-builder migration are
tracked in the [dataset audit](../../../docs/archive/dataset_preparation.md).
