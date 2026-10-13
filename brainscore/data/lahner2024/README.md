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

## Existing preparation scripts

| Module under `brainscore.data.lahner2024` | Produces |
| --- | --- |
| `prepare_audio_tracks` | Audio sidecars from the registered movies |
| `prepare_timeresolved_assembly` | Time-resolved responses and event metadata from fMRIPrep surface files |
| `prepare_motion_sidecar` | Motion/confound metadata |

Inspect inputs and output options before preparing data:

```bash
python -m brainscore.data.lahner2024.prepare_timeresolved_assembly --help
python -m brainscore.data.lahner2024.prepare_motion_sidecar --help
python -m brainscore.data.lahner2024.prepare_audio_tracks --help
```

The time-resolved converter accepts `--data-cache`, `--output-path` and
`--events-path`. Do not enable its optional upload flag for a local build.

**Local workflow gap:** the registered loaders still fetch hosted artifacts;
these scripts do not yet form a shared `brainscore.data prepare` workflow with
registered local output loading. The ordinary GLM-beta assembly and the
prepared time-resolved assembly are different targets, not interchangeable.

Before release, complete the local loader/command path, document the exact
native inputs and access steps, measure storage/runtime needs, and validate a
fresh build. See the [dataset audit](../../../docs/dataset_preparation.md).
