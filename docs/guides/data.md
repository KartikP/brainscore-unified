# Data

A **data plugin** holds the source files of a study: the stimuli that were shown and the responses that were recorded. A **benchmark** is an experiment over them: a protocol, the data it compares against, and a score. Keeping the two apart lets several benchmarks share one dataset. Lahner 2024, for example, backs a whole-cortex benchmark, a visual-ROI benchmark, a time-resolved one and a multimodal one.

## Load a dataset

```python
import brainscore

stimuli = brainscore.load_stimulus_set('LeBel2023-UTS03-words')   # what was shown
responses = brainscore.load_dataset('LeBel2023-UTS03')            # what was recorded
```

`load_dataset` and `load_stimulus_set` look in the unified registry first, then in the vision and language registries, so classic datasets such as `MajajHong2015.public` load the same way.

To present your own images instead, build a stimulus set from a table with `stimulus_id` and `image_path` columns ([Wrap your model](wrap_your_model.md#try-it-on-your-own-images)).

## Datasets in the unified package

| Dataset | Stimulus set | What it is | Files | Used by |
| --- | --- | --- | --- | --- |
| `Lahner2024-fMRI`, `-timeresolved`, `-timeresolved-events`, `-timeresolved-motion` | `BOLDMoments` | fMRI to 1,102 three-second videos, as GLM betas and as TR time series with events and motion confounds | hosted | `Lahner2024-fMRI-naturalistic*` |
| `LeBel2023-UTS03` | `LeBel2023-UTS03` (one row per TR), `LeBel2023-UTS03-words` (one row per word) | whole-cortex fMRI while listening to 25 stories, one participant | hosted (default settings); other settings build from the source pickle | `LeBel2023-UTS03-encoding*` |
| `Algonauts2025-{friends,friends-s7,ood}-sub{01,02,03,05}` | `Algonauts2025-{friends,friends-s7,ood}` | Courtois NeuroMod fMRI while watching Friends and films, Schaefer 1,000 parcels | local: data-use agreement | `Algonauts2025-friends-sub*`; the held-out `friends-s7` and `ood` splits have no fMRI and serve prediction export only (`Algonauts2025(split=...)`, see the [benchmark README](../../brainscore/benchmarks/algonauts2025/README.md)) |
| `EWoK-core-1.0` | `EWoK-core-1.0` | world-knowledge contrasts and their answer keys | local: gated text | `EWoK-core-1.0-logprobs`, `-choice` |
| `Yeatman2021` | `Yeatman2021` | word and pseudoword images with human lexical decisions (ROAR) | hosted | `Yeatman2021-lexical_decision-*` |

CogGym's experiments, human results and scoring code come from the authors' repository at a pinned revision; point `BRAINSCORE_COGGYM_CHECKOUT` at that checkout. `python -m brainscore.data local` lists every file you supply yourself, where to get it and whether it is found.

## Add a data plugin

```
brainscore/data/<name>/
    __init__.py          registration: identifiers and loader functions, no I/O until called
    README.md            where the data come from, their license, how to cite them
    data_packaging/      scripts that build the hosted files from the source data
```

Register a loader for each assembly and each stimulus set:

```python
from brainscore import data_registry, stimulus_set_registry

data_registry['MyStudy2026'] = load_responses           # returns a DataAssembly
stimulus_set_registry['MyStudy2026'] = load_stimuli     # returns a StimulusSet
```

Hosted files are fetched once, checked against a sha1 and cached (see `brainscore/data/lebel2023/hosted.py`). Data you may not redistribute stays local: declare it as a `LocalAsset` in `brainscore/data/local.py`, so users get instructions instead of a missing-file error.

A benchmark reaches its data only through `load_dataset` / `load_stimulus_set`, or the identifiers and constants a plugin's package exports. It never imports a plugin's modules; `tests/test_data_boundaries.py` checks this.
