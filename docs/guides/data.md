# Data

A **data plugin** holds the source files of a study: the stimuli that were shown and the responses that were recorded. A **benchmark** is an experiment over them: a protocol, the data it compares against, and a score. Keeping the two apart lets several benchmarks share one dataset. Lahner 2024, for example, backs a whole-cortex benchmark, a visual-ROI benchmark and a multimodal one.

## Load a dataset

This block downloads the hosted LeBel2023 files on first use and caches them.

```python
import brainscore

stimuli = brainscore.load_stimulus_set('LeBel2023-UTS03-words')   # what was shown
responses = brainscore.load_dataset('LeBel2023-UTS03')            # what was recorded
```

`load_dataset` looks in the unified registry first, then in the vision and language registries; `load_stimulus_set` looks in the unified registry, then in the vision registry. Classic datasets such as `MajajHong2015.public` load the same way.

To present your own images instead, build a stimulus set from a table with `stimulus_id` and `image_path` columns ([Wrap your model](wrap_your_model.md#try-it-on-your-own-images)).

A wrapper reads a stimulus file path only from the column names that the core catalog binds to its input (`brainscore_core.io_catalog.stimulus_columns`), tried in the catalog's order: `audio_path`, `audio_file_name`, `audio_file`, `audio` for sound; `video_path`, `video_file_name`, `video` for clips; `image_file_name`, `image_path`, `filename`, `image`, `vision` for images; a table with none of them is refused with that list.

## Datasets in the unified package

| Dataset | Stimulus set | What it is | Files | Used by |
| --- | --- | --- | --- | --- |
| `Lahner2024-fMRI`, `-timeresolved`, `-timeresolved-events`, `-timeresolved-motion` | `BOLDMoments` | fMRI to 1,102 three-second videos (1,000 training-set, 102 test-set), as GLM betas and as TR time series with events and motion confounds; the hosted GLM betas and stimulus set hold 1,026 of the videos (933 and 93) | hosted | `Lahner2024-fMRI-naturalistic*` (GLM betas); the time series back the class `Lahner2024BOLDMoments_timeresolved`, which is not registered |
| `LeBel2023-UTS03` | `LeBel2023-UTS03` (one row per TR), `LeBel2023-UTS03-words` (one row per word) | whole-cortex fMRI while listening to 25 stories, one participant | hosted (default settings); other settings build from the source pickle | `LeBel2023-UTS03-encoding*` |
| `Algonauts2025-{friends,friends-s7,ood}-sub{01,02,03,05}` | `Algonauts2025-{friends,friends-s7,ood}` | Courtois NeuroMod fMRI while watching Friends and films, Schaefer 1,000 parcels | local: CC0, not hosted by Brain-Score | `Algonauts2025-friends-sub*`; the held-out `friends-s7` and `ood` splits have no fMRI and serve prediction export only (`brainscore.tools.algonauts2025_predictions`, see the [benchmark README](../../brainscore/benchmarks/algonauts2025/README.md)) |
| `EWoK-core-1.0` | `EWoK-core-1.0` | world-knowledge contrasts and their answer keys | local: gated text | `EWoK-core-1.0-logprobs`, `-choice` |
| `Yeatman2021` | `Yeatman2021` | word and pseudoword images with human lexical decisions (ROAR) | hosted | `Yeatman2021-lexical_decision-*` |

CogGym's experiments, human results and scoring code come from the authors' repository at a pinned commit, in a folder Brain-Score makes at the first load, with CogGym's own environment ([the CogGym page](../../brainscore/data/coggym/README.md)). `python -m brainscore.data` lists each data plugin with the place of its source, then the files you supply yourself that a plugin declares as a local asset (the Algonauts 2025 files, the EWoK data, the Lahner2024 frames and sound tracks, the language atlas), and whether each is found; `python -m brainscore.data <asset name>` prints how to obtain one.

## Package a dataset from its source

One variable places all local unified data: `BRAINSCORE_DATA_HOME`, an absolute path or one that starts with `~` (default `~/brainscore-data`; an empty or relative value is refused). It holds `sources/<name>`, the original source of a data plugin, and one folder for each local data set, named as the asset: `algonauts2025-prepared`, `ewok-core-1.0`, `lahner2024-frames`, `lahner2024-sound`, `lana-atlas`, and `destrieux-atlas`, which nilearn fills on first use (`python -m brainscore.data` lists the local assets with their places). To keep one data set on another drive, put a link at its place. Hosted files stay in the brainio cache (`$BRAINIO_HOME`, default `~/.brainio`).

`package_data` builds a plugin's files from the original source, the way Brain-Score reads them. It looks for the source in one place, by the same rule for every plugin: the `source` argument, else `$BRAINSCORE_DATA_HOME/sources/<name>`, else `~/brainscore-data/sources/<name>`. It writes where the plugin's benchmarks read the data, so the benchmark runs next; the success message names any step that remains (`rajalingham2018-2afc`: its tool reads the files shipped in the plugin folder, so the build is a check of them). The place is a local folder, `$BRAINSCORE_DATA_HOME/<asset name>` (for example `~/brainscore-data/ewok-core-1.0`), or, for a hosted file, `$BRAINIO_HOME/<sha1>/<file name>` (`~/.brainio` by default), where the loader reads it in place of the download. The `output` argument (`--output`) writes elsewhere. An existing output is refused; the build goes to a private folder beside the output and moves into place only when it is complete and checked. When the check of a completed build fails or stops, nothing reaches the output and the built files stay in the private folder that the call made (mode 0700): the error gives its absolute path. No command reads, publishes from or deletes that folder; a new run builds again, and the folder can be deleted when it is no longer needed. A build that itself fails, or a step after the check that fails, keeps nothing. An empty folder that appears at the output place during the call can be replaced (it holds no data). A hosted file is published last, only when it is the registered file, and never over an existing entry; if a later step fails, the published hosted files stay (they are the files the loader would download) and the error names them.

```python
from brainscore.data import package_data

# Build the EWoK files from the native table in ~/brainscore-data/sources/ewok,
# into the folder the EWoK benchmarks read
output = package_data(name='ewok')

# Or name the source and the output folders
output = package_data(name='lebel2023', source='/data/litcoder', output='/data/lebel-build')
```

The same from a shell:

```bash
python -m brainscore.data                        # each plugin, where its source is expected, and whether it is there
python -m brainscore.data package ewok           # package one
```

When the source is missing, the error says what data is needed and from where, the access it needs, the folder and the files to put in it, and the tools and disk space. After a build, `package_data` prints the size, sha1 and counts (rows, dimension sizes, items) of the first 20 files; `build.json` in the output lists them for every file. It then checks each file against the plugin's expected record (the registered sha1, or the `data_packaging/expected.json` that a cloud run copied from its `build.json`): `identical` (equal sha1), `same content, bytes differ` (only in two cases: the registered copy is on the machine and every value agrees, typed and in place: text, integers, coordinates, types, dimensions, infinities and the places of missing values exactly, each finite float within `TOLERANCE = 1e-6` of its own expected value, scaled by its size when that is above 1; or no copy is here and the exact digest of the parsed content equals the recorded one; a table is read with `keep_default_na=False`; other kinds, such as PNG, WAV and ZIP, are compared by bytes only), or an error that names the file and what differs, and nothing is moved. A file with no record prints `no expected record` and its counts. The bytes of a written file can change between builds of the same content: four rebuilds of the LeBel netCDF assembly gave four sha1 (`same content, bytes differ`), while its tables and the Rajalingham and Lahner motion tables were `identical`. The last lines state where the data is now, whether the benchmarks read it there (or the one step that remains), and the result of the check. A hosted file reaches `$BRAINIO_HOME/<sha1>/<file name>` only when it is `identical`: the loader accepts only the registered bytes. A script reads the source and never changes it, and nothing is uploaded. Each plugin README's `## Rebuild` section states the same facts.

## Add a data plugin

```
brainscore/data/<name>/
    __init__.py          registration: identifiers, loader functions and local assets, no I/O until called
    README.md            where the data come from, their license, how to cite them
    data_packaging/      the script that builds the plugin's files from the source data
```

The folder registers by its presence: every subpackage of `brainscore/data/` is imported with `brainscore`, so no other file changes. A plugin in another package (another repository) registers when you import that package before you load it.

Register a loader for each assembly and each stimulus set:

```python
import numpy as np
import pandas as pd
import brainscore
from brainscore import data_registry, stimulus_set_registry
from brainscore_core.supported_data_standards.brainio.assemblies import NeuroidAssembly
from brainscore_core.supported_data_standards.brainio.stimuli import StimulusSet


def load_stimuli():                 # read your stimulus table here
    return StimulusSet(pd.DataFrame({'stimulus_id': ['s0', 's1'],
                                     'image_path': ['s0.png', 's1.png']}))


def load_responses():               # read your recordings here
    return NeuroidAssembly(np.zeros((2, 3)), dims=['presentation', 'neuroid'],
                           coords={'stimulus_id': ('presentation', ['s0', 's1']),
                                   'condition': ('presentation', ['a', 'a']),
                                   'neuroid_id': ('neuroid', [0, 1, 2])})


# Register both under one identifier, then load
data_registry['MyStudy2026'] = load_responses           # returns a DataAssembly
stimulus_set_registry['MyStudy2026'] = load_stimuli     # returns a StimulusSet
print(brainscore.load_dataset('MyStudy2026').shape)      # (2, 3)
```

Behavioral data (what participants answered) is a `BehavioralAssembly` with one row per
(participant, trial): `stimulus_id` and `participant` on `presentation`, the chosen label as
the value.

`templates/new_data/` is a complete data plugin with loaders for stimuli, recordings and
participant choices, image files and tests. Its test goes to `tests/test_<name>.py`.

Say how your files are built from the source: register a factory of a `DataBuilder` beside the registry lines, with your `data_packaging/` script as its command. `package_data` fills in `{source}` and `{output}`, runs the script, and prints the sha1 and counts of what it wrote.

```python
from brainscore import data_builder_registry
from brainscore.data import DataBuilder

# In brainscore/data/mystudy2026/__init__.py, beside the registry lines: a factory of the facts
data_builder_registry['mystudy2026'] = lambda: DataBuilder(
    source='https://example.org/mystudy2026; licence CC-BY 4.0',
    access='public, no account',
    layout='responses.mat and the stimuli/ folder, as the download unpacks',
    needs='scipy; 2 GB of disk; minutes',
    writes='assy_MyStudy2026.nc, stimulus_MyStudy2026.csv',
    reads='brainio',      # hosted files; or the name of a local asset; or empty for no local read place
    build='python -m brainscore.data.mystudy2026.data_packaging.package --source {source} --out {output}',
    requires=('responses.mat', 'stimuli'),
    sha1={'assy_MyStudy2026.nc': '<the sha1 your loader pins>'},
)

# The README section is generated from the same facts; paste it under "## Rebuild"
print(data_builder_registry['mystudy2026']().readme('mystudy2026'))
```

`requires` names what the source folder must hold; a source that lacks it gets the error with the facts above. A script reads the source and never changes it: it writes into `{output}`, and downloads or intermediate files go into `{work}`, a private folder removed after the build. A dataset that cannot be rebuilt yet states `cannot=` (what is known, what a maintainer needs); one that is read as published leaves out `build`. `tests/test_data_packaging.py` fails a data plugin without a declaration or with a README that differs from it, and prints the block to paste.

Hosted files are fetched once, checked against a sha1 and cached (see `brainscore/data/lebel2023/hosted.py`). Data you may not redistribute stays local: declare it as a `LocalAsset` in your plugin's `__init__.py`, so users get instructions instead of a missing-file error, and `python -m brainscore.data` lists it. Name the command that prepares it in `prepare`. Its place follows from its name: `$BRAINSCORE_DATA_HOME/<asset name>`. Read the location with `local.path(...)` when the data is needed, not at import. A second asset with the same name is refused.

```python
from brainscore.data import local

# In brainscore/data/mystudy2026/__init__.py, beside the registry lines.
MY_FILMS = local.register(local.LocalAsset(
    name='mystudy2026-films',                  # read at $BRAINSCORE_DATA_HOME/mystudy2026-films
    kind='stimuli',
    why_local='The films are licensed to each lab; Brain-Score may not share them.',
    source='https://example.org/mystudy2026',
    obtain='The 12 film files, about 4 GB.',
    prepare='python -m brainscore.data package mystudy2026',
))

try:
    films = local.path(MY_FILMS.name)          # the folder, when it is there
except local.LocalDataMissing as missing:
    print(missing)                             # where to get it, where to put it, the command
```

A benchmark reaches its data only through `load_dataset` / `load_stimulus_set`, or the identifiers and constants a plugin's package exports. It never imports a plugin's modules; `tests/test_data_boundaries.py` checks this.
