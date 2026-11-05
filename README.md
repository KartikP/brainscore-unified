# Brain-Score Unified Model Interface (UMI)

Run models across compatible Brain-Score benchmarks, record their responses, and attach experiment tools. A `Subject` connects the model to the experiment; `BrainScoreModel` supplies extraction and task helpers. Use `Experiment` to combine a protocol with recording and intervention tools.

**Start here:** [Lesson 1: your first subject](notebooks/lessons/01_your_first_subject.ipynb), then [Lesson 10: a whole study in one notebook](notebooks/lessons/10_run_an_experiment.ipynb) (design a task, record answers and activity, save the run, turn it into a benchmark). Reference: [Manual](docs/README.md) | [What UMI is](docs/01_what_is_umi.md) | [Core concepts](docs/02_core_concepts.md) | [All lessons](notebooks/README.md#lessons) | [Experiments](notebooks/README.md#experiments)

UMI is a v2 source candidate. See the [release policy](docs/archive/production_release.md) for remaining qualification.

## Layout

- `brainscore/model_helpers/` — extraction wrappers: image, text,
  VLM-vision, video, audio, tensor, and the stateful `PolicyWrapper` for
  closed-loop sessions.
- `brainscore/harnesses/` — environment and subject harnesses (robotics
  environment harness; human-harness scaffold).
- `brainscore/models/` — registered models (CLIP, Qwen-VL, BLIP-2, V-JEPA,
  null controls, and others).
- `brainscore/benchmarks/` — registered benchmarks (ROAR, MajajHong-unified,
  Pereira-unified, Lahner2024, Algonauts2025, and others).
- `brainscore/metrics/` — scoring metrics.

## Install

Requires Python 3.11, `conda` (miniforge or miniconda) and `git`.

**On a new machine** (recommended): the bootstrap clones all four repositories, with core,
vision and language at the commits this version of unified was tested with
(`install/peer-revisions.json`), builds the pinned environment and checks it:

```bash
mkdir brainscore-umi && cd brainscore-umi
base=https://raw.githubusercontent.com/KartikP/brainscore-unified/unified-model-interface-v2/install
curl -fsSLO "$base/setup.sh" && curl -fsSLO "$base/environment-unified.yml"
bash setup.sh
conda activate brainscore-unified
```

**With the four repositories already checked out side by side:**

```bash
cd <workspace-root>          # the directory holding core/ vision/ language/ unified/
conda create -y -n brainscore-unified python=3.11
conda activate brainscore-unified
python -m pip install -c unified/install/v2-constraints.txt -e ./core -e ./vision -e ./language -e "./unified[notebooks,test]"
python -m pip check
python -m brainscore.doctor
```

Install all four repositories in the same pip command: the candidate peer versions come
from these checkouts and are not on PyPI. Keep `-c unified/install/v2-constraints.txt`.
The packages declare only broad bounds (`numpy<2`, `xarray==2022.3.0`, ...); the
constraints file holds every shared dependency (pyarrow, torch, transformers, ...) at the
tested version, and without it pip can pick versions that break imports.
For an immutable checkout set, use [the integration instructions](install/README.md#reproduce-the-reviewed-peer-revisions).

Do not point conda at `install/environment-unified.yml` in place: conda resolves its
relative `-e ./core` entries from the file's own directory, so it fails with
`ERROR: ./core is not a valid editable requirement`. See
[install/README.md](install/README.md) for troubleshooting.

## Local datasets and external experiments

Use [local data builders](docs/archive/local_data.md) to prepare author-format files or
retrieve data with a provider-supported request ID. See [EWoK](docs/archive/ewok.md) for
world-knowledge scoring and [CogGym](docs/archive/coggym.md) for cognitive experiments
with UMI recording and intervention tools.

## Capability Status Matrix

Status labels separate real numeric validation from local structural coverage.
`EC2-only` means the full score needs large data, model weights, API calls, or
benchmark compute that must not run on the local Mac. `Demo-only` means the
interface is exercised, but the result is not a brain-alignment claim.

| Capability | Example | Status | Evidence |
| --- | --- | --- | --- |
| Vision neural encoding | MajajHong2015 V4/IT | validated | Baseline manifest plus slow replay test preserve anchored leaderboard scores. |
| Language neural encoding | Pereira2018 | validated | Same regression-baseline path as vision; normal CI checks manifest shape, slow tier re-scores. |
| Behavioral lexical decision | ROAR / Yeatman2021 | validated | Registered image/text variants, shared split, ceiling, and CLIP-above-chance scoring are tested. |
| Video neural encoding | Lahner2024 BOLDMoments | EC2-only | Local tests verify registration, temporal preprocessing, and V-JEPA wiring; full scoring is EC2-only. |
| Audio wrapper | Wav2Vec2-style features | structurally-tested | Construction, stimulus columns, aggregation, truncation, and cache keys are unit-tested without weights. |
| Audio+video fMRI | Lahner2024 multimodal | EC2-only | Registry, mode validation, null controls, and modality tagging are local; real forward/data runs are EC2-only. |
| Movie multimodal fMRI | Algonauts2025 / CNeuroMod | EC2-only | Twelve benchmark entries and scoring kernels are guarded locally; the data assembly is about 100 GB on EC2. |
| State-change perturbation | `process(StateChange)` | structurally-tested | Hook install, indexed ablation, concurrent handles, and reset are tested on a toy torch model. |
| Closed-loop action | GridGame `action_fn` | demo-only | Floor, ceiling, deterministic boards, and registration are tested; the game is an interface demo. |
| Closed-weight API behavior | OpenRouter / OpenAI-compatible | structurally-tested | Provider registration, lazy model construction, parsing, image payloads, and cache behavior are mocked locally. |
