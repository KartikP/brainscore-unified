# Brain-Score Unified Model Interface (UMI)

A Brain-Score benchmark is an experiment with brain or behavior data from people and animals, and a metric that compares a model with that data. With UMI an experimenter writes the experiment once and it runs on every model that fits, and a modeler adapts a model once and it takes part in every experiment that fits, with recording and experiment tools for both. A `Subject` connects the model to the experiment; `BrainScoreModel` supplies extraction and task helpers. Use `Experiment` to combine a protocol with recording and intervention tools.

**Start here:** [Lesson 1: your first subject](notebooks/lessons/01_your_first_subject.ipynb), then [Lesson 10: a whole study in one notebook](notebooks/lessons/10_run_an_experiment.ipynb) (design a task, record answers and activity, save the run, turn it into a benchmark). [Score a model](docs/guides/score_a_model.md) scores a first pair with no download. Reference: [Manual](docs/README.md) | [What UMI is](docs/01_what_is_umi.md) | [Core concepts](docs/02_core_concepts.md) | [All lessons](notebooks/README.md#lessons) | [Experiments](notebooks/README.md#experiments)

UMI is a source candidate on the `unified-model-interface-v2` branches; its packages are not published on PyPI.

## Layout

- `brainscore/model_helpers/`: extraction wrappers for image, text, VLM vision, video and audio.
- `brainscore/harnesses/`: environment and subject harnesses (robotics environment harness, a prototype human subject `HumanSubject`), and the adapters for external experiment libraries (`adapter.py` holds what every adapter shares; `coggym.py`, `ewok.py`).
- `brainscore/models/`: registered models (CLIP, Qwen-VL, BLIP-2, V-JEPA, null controls and others).
- `brainscore/benchmarks/`: registered benchmarks (ROAR, induced dyslexia, Lahner2024, LeBel2023, Algonauts2025, CogGym, EWoK, the grid game). MajajHong2015, Pereira2018 and the other Brain-Score 2.0 benchmarks are registered in the vision and language packages; `brainscore.score` runs them for any subject that fits.
- `brainscore/metrics/`: scoring metrics.
- `brainscore/data/`: data plugins (stimuli, recorded responses, and the packaging scripts that build them).
- `brainscore/tools/`: tools to fit, apply and register (linear probe, LEACE, steering, layer mapping, auto-register, Algonauts prediction export).

## Install

Requires Python 3.11, `conda` (miniforge or miniconda) and `git`.

**On a new machine** (recommended): the bootstrap clones all four repositories, with core,
vision and language at the commits named in `install/peer-revisions.json` (the last pinned
set; a branch in work can need later peer commits), builds the pinned environment and
checks it:

```bash
mkdir brainscore-umi && cd brainscore-umi    # empty folder for the four repositories
base=https://raw.githubusercontent.com/KartikP/brainscore-unified/unified-model-interface-v2/install
curl -fsSLO "$base/setup.sh" && curl -fsSLO "$base/environment-unified.yml"
bash setup.sh
conda activate brainscore-unified
```

**The exact scoring platform** (Linux x86_64, Python 3.11, CUDA 12.4), for scores to be
compared bit for bit: the same bootstrap with `--lock` installs every package at the version
of `install/lock-linux-py311-cu124.txt` (230 packages, torch 2.6.0+cu124), and
`python -m brainscore.doctor` then lists the Python and every package that differs from that lock,
and prints the environment fingerprint and the math threads that every score records. Two scores
are equal bit for bit only when these, the accelerator and the code commits are equal
([Score a model](docs/guides/score_a_model.md)):

```bash
bash setup.sh --lock
```

Both commands use conda-forge only, so a fresh miniconda does not stop at the terms of
service of Anaconda's default channels.

**With the four repositories already checked out side by side:**

```bash
cd <workspace-root>          # the directory holding core/ vision/ language/ unified/
conda create -y -n brainscore-unified --override-channels -c conda-forge python=3.11
conda activate brainscore-unified
# Install the four checkouts together, at the tested versions
python -m pip install -c unified/install/v2-constraints.txt -e ./core -e ./vision -e ./language -e "./unified[notebooks,test]"
python -m pip check
python -m brainscore.doctor
```

Install all four repositories in the same pip command: the candidate peer versions come
from these checkouts and are not on PyPI. Keep `-c unified/install/v2-constraints.txt`.
The packages declare only broad bounds (`numpy<2`, `xarray==2022.3.0`, ...); the
constraints file holds the shared dependencies that break imports or move scores when they
drift (numpy, pandas, pyarrow, scipy, scikit-learn, xarray, transformers, torch, ...), not
every package. For every package at one version, use the full lock (`bash setup.sh --lock`
on the scoring platform).
For an immutable checkout set, use [the integration instructions](install/README.md#reproduce-the-reviewed-peer-revisions).

Do not point conda at `install/environment-unified.yml` in place: conda resolves its
relative `-e ./core` entries from the file's own directory, so it fails with
`ERROR: ./core is not a valid editable requirement`. See
[install/README.md](install/README.md) for troubleshooting.

## Local datasets and external experiments

`python -m brainscore.data` lists the files you supply yourself and where to get each
([Data](docs/guides/data.md)). [External libraries](docs/guides/external_libraries.md)
shows how EWoK and CogGym map into benchmarks.

## Capability Status Matrix

Status labels separate real numeric validation from local structural coverage.
`EC2-only` means the full score needs large data, model weights, API calls, or
benchmark compute that does not run on a laptop. `Demo-only` means the
interface is exercised, but the result is not a brain-alignment claim.

| Capability | Example | Status | Evidence |
| --- | --- | --- | --- |
| Vision neural encoding | MajajHong2015 V4/IT | validated | The table of reference scores (`tests/fixtures/reference-scores.json`) holds the leaderboard values and the cloud reference values; normal CI checks the table, the slow tier re-scores the leaderboard rows. |
| Language neural encoding | Pereira2018 | validated | The same table holds cloud reference values for Pereira2018 ridge (GPT-2, CLIP-ViT-B/32); normal CI checks the table. |
| Behavioral lexical decision | ROAR / Yeatman2021 | validated | Registered image/text variants, shared split, ceiling, and CLIP-above-chance scoring are tested. |
| Video neural encoding | Lahner2024 BOLDMoments | EC2-only | Local tests verify registration, temporal preprocessing, and V-JEPA wiring; full scoring is EC2-only. |
| Audio wrapper | Wav2Vec2-style features | structurally-tested | Construction, stimulus columns, aggregation, truncation, and cache keys are unit-tested without weights. |
| Audio+video fMRI | Lahner2024 multimodal | EC2-only | Registry, channel routing, voxel masks and both streams are tested locally on synthetic data; real forward/data runs are EC2-only. |
| Movie multimodal fMRI | Algonauts2025 / CNeuroMod | EC2-only | Six benchmark entries and the banded-ridge scoring are tested locally; the source data is about 130 GB. Each entry uses all released fMRI of one participant (Friends seasons 1 to 6 and Movie10). Three forms exist: one frame per scan (`vision:image`, participants 1, 2, 3 and 5), one 1.49 s clip per scan (`vision:video`, participant 1 only) and the clip with its sound (`vision:video` and `audio`, participant 1 only). Every form predicts all parcels from the model's layers for the regions the benchmark names. No form sends speech text. |
| Intervention | `apply_state_change`, `Intervene` | structurally-tested | Hook install, indexed ablation, concurrent handles, and reset are tested on a toy torch model. |
| Closed-loop action | GridGame `policy` | demo-only | Floor, ceiling, deterministic boards, and registration are tested; the game is an interface demo. |
| Closed-weight API behavior | OpenRouter / OpenAI-compatible | structurally-tested | Provider registration, lazy model construction, parsing, image payloads, and cache behavior are mocked locally. |
