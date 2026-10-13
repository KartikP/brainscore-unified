# Prepare Algonauts2025 data locally

Obtain the competition files through the
[authors' DataLad instructions](https://github.com/courtois-neuromod/algonauts_2025.competitors).
Follow their access terms for the movies, transcripts and fMRI separately.
Keep the native `fmri/` and `stimuli/` directories together.

The existing converter reads per-subject HDF5 responses and movie/transcript
metadata. It writes per-TR NetCDF assemblies and stimulus CSVs. It needs the
unified installation plus `h5py`; the source dataset is approximately 130 GB,
and preparing all subject/split outputs was estimated at another 25 GB. Runtime
depends on storage and scope; it has not been qualified for this release.

```bash
python -m brainscore.data.algonauts2025.prepare_assembly \
  --algonauts-root /path/to/native-data \
  --output-root /path/to/new-build \
  --subjects 1 \
  --splits friends

export ALGONAUTS_DATA_ROOT=/path/to/new-build
```

Movie paths in the CSV point to your native files; keep those files available.
Check loading without a model:

```python
import brainscore

stimuli = brainscore.load_stimulus_set("Algonauts2025-friends")
responses = brainscore.load_dataset("Algonauts2025-friends-sub01")
print(len(stimuli), responses.sizes)
```

Other subjects are 2, 3 and 5. Splits `friends_s7` and `ood` contain held-out
response placeholders, not measured responses available for local scoring.
Do not treat those NaNs as an acquired human dataset.

This converter is not registered with `brainscore.data prepare`. The full
fresh-build validation, content manifest and shared-builder migration are
tracked in the [dataset audit](../../../docs/dataset_preparation.md).
