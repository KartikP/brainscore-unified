# Template: a new dataset

For: *"I have my lab's measurements and the stimuli that produced them, and I want a
benchmark to be able to load them."*

This is usually the first seam a neuroscientist needs, and it is the one furthest from
ordinary software work. The template runs on synthetic files first, and you replace
them a piece at a time.

## What a data plugin provides

What was shown and what was measured, each registered under an identifier:

| | What it is | Registry |
| --- | --- | --- |
| StimulusSet | what was shown, one row per stimulus | `stimulus_set_registry` |
| DataAssembly, neural | what was measured, `(presentation, neuroid)` | `data_registry` |
| DataAssembly, behavior | what participants answered, one row per (participant, trial) | `data_registry` |

The template registers `your-stimuli`, `your-measurements` and `your-choices`. Keep the
ones your study has.

Neither scores anything. A benchmark loads both and compares a model against them.

## Install

Copy this folder to `brainscore/data/<your_name>/` and move `test_data.py` to
`tests/test_<your_name>.py`. Nothing else to edit: the folder registers by its presence.

Then, typically from inside a benchmark (here from the `templates/` folder, where
importing the template registers it):

```python
import new_data
from brainscore import load_stimulus_set, load_dataset

stimuli  = load_stimulus_set('your-stimuli')
measured = load_dataset('your-measurements')
choices  = load_dataset('your-choices')
print(len(stimuli), dict(measured.sizes))   # 32 {'presentation': 32, 'neuroid': 16}
print(dict(choices.sizes))                  # {'presentation': 128}: 4 participants x 32 trials
```

## Replace in this order

1. `_build_example_files`: point at your real images/audio/video instead of
   generating them. Everything else keeps working.
2. `load_stimulus_set`: read your metadata CSV rather than the constructed rows.
3. `load_assembly`: swap the random values for your recordings, keeping the coords.
4. `load_choices`: swap the made-up answers for your participants' answers.

Run `pytest` after each step. The tests fail with a named reason, not a shape error.

As shipped, loading the template writes its 32 example images once to
`your_data_stimuli/` in the Brain-Score home (`BRAINSCORE_HOME`, default `~/.brain-score`).
The template's test points `BRAINSCORE_HOME` at a temporary folder, so a test run leaves
nothing there; keep that in your test while your loader writes files.

## The four things that make data benchmark-usable

Each corresponds to a failure that is easy to hit and hard to read:

- A recognized modality column: `image_path`, `sentence`, `audio_path`,
  `video_path`. The subject finds a stimulus set's modality by column name. Without
  one you get `No column in this stimulus set says what kind of input it holds`.
- `stimulus_paths` pointing at files that exist. A real model opens them. Paths that
  merely look plausible fail only once someone runs a real model, not in your tests.
- At least two coords per axis. With one, brainio never builds the MultiIndex, and
  metrics later fail with `no stimulus_id on the presentation axis`.
- Whatever your metric stratifies on. Cross-validated metrics split folds by a
  presentation coord (`object_name` here) and raise
  `Expected stratification coordinate` without it.

## Data you may not share

Declare it as a local asset in your `__init__.py`, beside the registry lines. Users
then get instructions instead of a missing-file error, and `python -m brainscore.data`
lists it. Put the script that prepares it in `data_packaging/`.

```python
from brainscore.data import local

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

## Publishing

The loaders here read local files, which is right while developing. Registered datasets
in this repository pull from S3 with pinned checksums and version ids so everyone scores
against byte-identical data. See the note at the bottom of `data.py`, and
`brainscore/data/roar_yeatman2021/` for a complete worked example.

Pin the checksums. An unpinned dataset can change underneath a published score.
