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
| `Algonauts2025-{friends,friends-s7,ood}-sub{01,02,03,05}` | `Algonauts2025-{friends,friends-s7,ood}` | Courtois NeuroMod fMRI while watching Friends and films, Schaefer 1,000 parcels | local: data-use agreement | `Algonauts2025-friends-sub*`; the held-out `friends-s7` and `ood` splits have no fMRI and serve prediction export only (`brainscore.tools.algonauts2025_predictions`, see the [benchmark README](../../brainscore/benchmarks/algonauts2025/README.md)) |
| `EWoK-core-1.0` | `EWoK-core-1.0` | world-knowledge contrasts and their answer keys | local: gated text | `EWoK-core-1.0-logprobs`, `-choice` |
| `Yeatman2021` | `Yeatman2021` | word and pseudoword images with human lexical decisions (ROAR) | hosted | `Yeatman2021-lexical_decision-*` |

CogGym's experiments, human results and scoring code come from the authors' repository at a pinned revision; point `BRAINSCORE_COGGYM_CHECKOUT` at that checkout. `python -m brainscore.data` lists the files you supply yourself that a plugin declares as a local asset (the Algonauts 2025 files, the CogGym checkout, the EWoK data, the Lahner2024 frames and sound tracks, the language atlas and the LeBel source pickle), where to get each and whether it is found; `python -m brainscore.data <name>` prints how to obtain one.

## Add a data plugin

```
brainscore/data/<name>/
    __init__.py          registration: identifiers, loader functions and local assets, no I/O until called
    README.md            where the data come from, their license, how to cite them
    data_packaging/      scripts that build the hosted files from the source data
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


data_registry['MyStudy2026'] = load_responses           # returns a DataAssembly
stimulus_set_registry['MyStudy2026'] = load_stimuli     # returns a StimulusSet
print(brainscore.load_dataset('MyStudy2026').shape)      # (2, 3)
```

Behavioral data (what participants answered) is a `BehavioralAssembly` with one row per
(participant, trial): `stimulus_id` and `participant` on `presentation`, the chosen label as
the value.

`templates/new_data/` is a complete data plugin with loaders for stimuli, recordings and
participant choices, image files and tests. Its test goes to `tests/test_<name>.py`.

Hosted files are fetched once, checked against a sha1 and cached (see `brainscore/data/lebel2023/hosted.py`). Data you may not redistribute stays local: declare it as a `LocalAsset` in your plugin's `__init__.py`, so users get instructions instead of a missing-file error, and `python -m brainscore.data` lists it. Put the script that prepares it in your `data_packaging/` folder and name it in `prepare`. Read the location with `local.path(...)` when the data is needed, not at import. A second asset with the same name is refused.

```python
from brainscore.data import local

# In brainscore/data/mystudy2026/__init__.py, beside the registry lines.
MY_FILMS = local.register(local.LocalAsset(
    name='mystudy2026-films',
    env_var='BRAINSCORE_MYSTUDY2026_FILMS',
    default_path='mystudy2026/films',           # under the home folder when the variable is unset
    kind='stimuli',
    why_local='The films are licensed to each lab; Brain-Score may not share them.',
    source='https://example.org/mystudy2026',
    obtain='The 12 film files, about 4 GB.',
    prepare='python -m brainscore.data.mystudy2026.data_packaging.cut_frames',
))

try:
    films = local.path(MY_FILMS.name)          # the folder, when it is there
except local.LocalDataMissing as missing:
    print(missing)                             # where to get it, the variable, the command
```

A benchmark reaches its data only through `load_dataset` / `load_stimulus_set`, or the identifiers and constants a plugin's package exports. It never imports a plugin's modules; `tests/test_data_boundaries.py` checks this.
