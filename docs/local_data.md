# Data you have to supply yourself

Some benchmarks cannot ship their inputs. Usually it is the stimuli — films,
copyrighted text, face photographs — and occasionally the neural data or an
atlas carries its own agreement. Brain-Score can still score those benchmarks;
it just cannot hand you the bytes.

## As a user

Ask what you are missing before starting a run rather than during one:

```bash
python -m brainscore.data
```


Then ask about one of them:

```bash
python -m brainscore.data algonauts2025-root
```

which prints where to get it, why it is not bundled, where to put it, the
environment variable that overrides that location, and — when the download is
not directly readable — the command that converts it.

Every asset can live wherever you already keep it; the environment variable
exists so you never have to move or copy a dataset you already have.

## Build from native files or an access ID

List the available builders:

```bash
python -m brainscore.data list
```

Build from the original authors' files, without manually converting them:

```bash
python -m brainscore.data prepare EWoK-core-1.0 \
  --source /path/to/native-data \
  --output /path/to/new-build
```

A plugin with an author-supported download service can instead accept
`--request-id YOUR_ID`. The identifier's meaning belongs to that provider: an
author-issued retrieval code, an accession, or a dataset repository ID. It is
not a universal authorization token. Authentication and access approval still
follow the provider's process. [EWoK](ewok.md) supports approved Hugging Face
access and native local exports.

The shared command builds in a private staging directory and publishes the
output only after successful conversion. Failed builds leave no partial dataset;
existing outputs are never overwritten. Input files remain unchanged. A
`build.json` records the builder's source and conversion provenance.

Existing asset-status commands and dataset-specific conversion scripts still
work. EWoK is the first plugin using this shared builder; other datasets are not
automatically migrated.

## As a benchmark author

Declare the asset in the manifest at the bottom of `brainscore/data/local.py`:

```python
MY_ASSET = register(LocalAsset(
    name='mydata-stimuli',
    env_var='BRAINSCORE_MYDATA_STIMULI',
    default_path='Downloads/mydata',
    kind='stimuli',
    why_local='Films licensed to the original study, not redistributable.',
    source='https://example.org/request-access',
    obtain='The stimulus archive, about 40 GB, after signing the DUA.',
    prepare='python -m brainscore.data.mydata.prepare --root <root>',
    used_by=['MyData-encoding'],
))
```

Read it with `local.path('mydata-stimuli')`, which returns a `Path` or raises
`LocalDataMissing` carrying those instructions. Do not check for the file
yourself — the whole point is that the failure is identical everywhere.

The manifest is deliberately a plain list in one module rather than a
registration scattered across each reader. Asking what you need should never
require importing the code that needs it, so `python -m brainscore.data` works
on a bare checkout without loading a benchmark, a model, or numpy.

### Conversion scripts

Anything that is not readable as downloaded gets a `prepare_*.py` beside its
data package, runnable as a module and taking the download root as an argument:

| script | turns |
|---|---|
| `data.algonauts2025.prepare_assembly` | a DataLad checkout into per-subject assemblies |
| `data.lahner2024.prepare_audio_tracks` | MP4s into 16 kHz mono WAV sidecars |
| `data.lahner2024.prepare_timeresolved_assembly` | fmriprep surface output into a TR-resolved assembly |
| `data.lahner2024.prepare_motion_sidecar` | fmriprep confounds into a motion sidecar |

Two conventions worth keeping. Write output somewhere derived, never back into
the download, so a user can re-run a conversion without re-downloading 130 GB.
And key any cache on the parameters that produced it — the LeBel loader keys on
its trim, because a cache written under an older convention looks identical on
inspection and simply scores half as well.

### What does not belong here

Writable caches the code creates itself, such as
`BRAINSCORE_LAHNER_FRAMES_DIR` or `BRAINSCORE_LEBEL_CACHE`. Those are outputs,
not things a user obtains, and a missing one is not an error.

### Register a reusable builder

Keep the converter beside the data plugin. Register a lazy factory:

```python
from brainscore.data.preparation import DataBuilder, data_builder_registry


def builder():
    from .prepare import convert, retrieve
    return DataBuilder(build=convert, resolve=retrieve)


data_builder_registry["MyDataset"] = builder
```

- `convert(source, destination)` validates native files, writes the prepared
  dataset under `destination`, and returns JSON provenance. Record source hashes,
  conversion settings, exclusions, and the schema version. Validate shapes,
  unique IDs, and required metadata before returning.
- `retrieve(request_id)` uses the authors' documented service and returns
  `(local_path, provenance)`. Use existing authentication; never accept terms or
  print credentials. Keep private request IDs out of provenance and error text.
  Set `resolve=None` when there is no supported retrieval service.
- Loaders must verify the prepared schema and checksum before scoring. Register
  them in `data_registry` and `stimulus_set_registry` as usual.
- Test both access paths using synthetic fixtures, failed downloads/conversions,
  and score agreement with the native evaluator. Do not commit restricted data.

A LAION-fMRI-style request-ID workflow belongs in that dataset's `retrieve`
function. The command passes the ID through without interpreting or storing it;
it cannot invent a download API where the authors provide none.
