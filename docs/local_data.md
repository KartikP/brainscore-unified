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

```
   asset               kind         path

-- algonauts2025-root  stimuli      /home/you/algonauts_2025
ok lana-atlas          atlas        /home/you/Downloads/20425209/FS
ok lebel2023-pickle    neural data  /home/you/Downloads/assembly_lebel_uts03.pkl

2/3 present; missing: algonauts2025-root
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
