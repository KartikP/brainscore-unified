# Template — a new dataset

For: *"I have my lab's measurements and the stimuli that produced them, and I want a
benchmark to be able to load them."*

This is usually the first seam a neuroscientist needs, and it is the one furthest from
ordinary software work — so the template runs on synthetic files first, and you replace
them a piece at a time.

## What a data plugin provides

Two things, registered under two identifiers:

| | What it is | Registry |
| --- | --- | --- |
| **StimulusSet** | what was shown — one row per stimulus | `stimulus_set_registry` |
| **DataAssembly** | what was measured — `(presentation, neuroid)` | `data_registry` |

Neither scores anything. A benchmark loads both and compares a model against them.

## Install

Copy this folder to `brainscore/data/<your_name>/`, then add one line to
`brainscore/data/__init__.py`:

```python
from . import your_name
```

Then, typically from inside a benchmark:

```python
from brainscore import load_stimulus_set, load_dataset
stimuli  = load_stimulus_set('your-stimuli')
measured = load_dataset('your-measurements')
```

## Replace in this order

1. **`_build_example_files`** — point at your real images/audio/video instead of
   generating them. Everything else keeps working.
2. **`load_stimulus_set`** — read your metadata CSV rather than the constructed rows.
3. **`load_assembly`** — swap the random values for your recordings, keeping the coords.

Run `pytest` after each step. The tests fail with a named reason, not a shape error.

## The four things that make data benchmark-usable

Each corresponds to a failure that is easy to hit and hard to read:

- **A recognized modality column.** `image_file_name`, `sentence`, `audio_path`,
  `video_path`. Dispatch identifies a stimulus set's modality by column *name*; without
  one you get `No recognized modality columns`.
- **`stimulus_paths` pointing at files that exist.** A real model opens them. Paths that
  merely look plausible fail only once someone runs a real model, not in your tests.
- **At least two coords per axis.** With one, brainio never builds the MultiIndex, and
  metrics later fail with `no stimulus_id on the presentation axis`.
- **Whatever your metric stratifies on.** Cross-validated metrics split folds by a
  presentation coord (`object_name` here) and raise
  `Expected stratification coordinate` without it.

## Publishing

The loaders here read local files, which is right while developing. Registered datasets
in this repository pull from S3 with pinned checksums and version ids so everyone scores
against byte-identical data — see the note at the bottom of `data.py`, and
`brainscore/data/roar_yeatman2021/` for a complete worked example.

Pin the checksums. An unpinned dataset can change underneath a published score.
