# Brain-Score Unified Model Interface (UMI)

Run models across compatible Brain-Score benchmarks, record their responses, and attach experiment tools. A `Subject` connects the model to the experiment; `BrainScoreModel` supplies extraction and task helpers. Use `Experiment` to combine a protocol with recording and intervention tools.

**Start here:** [Getting started](docs/getting_started.md) | [Concepts](docs/concepts.md) | [Build tools](docs/tool_authoring.md) | [Supported features and environments](docs/supported_features.md)

UMI is a v2 source candidate. See the [release policy](docs/production_release.md) for remaining qualification.

## Layout

- `brainscore/model_helpers/` — extraction wrappers: image, text,
  VLM-vision, video, audio, tensor, and the stateful `PolicyWrapper` for
  closed-loop embodied dispatch.
- `brainscore/harnesses/` — environment and subject harnesses (robotics
  environment harness; human-harness scaffold).
- `brainscore/models/` — registered models (CLIP, Qwen-VL, BLIP-2, V-JEPA,
  null controls, and others).
- `brainscore/benchmarks/` — registered benchmarks (ROAR, MajajHong-unified,
  Pereira-unified, Lahner2024, Algonauts2025, and others).
- `brainscore/metrics/` — scoring metrics.

## Install

Four editable installs in one Python 3.11 environment.

**With the four repositories already checked out side by side:**

```bash
cd <workspace-root>          # the directory holding core/ vision/ language/ unified/
conda create -y -n brainscore-unified python=3.11
conda activate brainscore-unified
python -m pip install -c unified/install/v2-constraints.txt -e ./core -e ./vision -e ./language -e "./unified[notebooks,test]"
python -m pip check
python -m brainscore.doctor
```

Install all four repositories in the same pip command. The candidate peer versions
are supplied by these checkouts; they are not available from PyPI. Installing
only unified, vision, or language into an empty environment will not resolve the
candidate dependencies. Do not remove the exact peer pins to work around this.
For an immutable checkout set, use [the integration instructions](install/README.md#reproduce-the-reviewed-peer-revisions).

 Version pins (`numpy<2`, `xarray==2022.3.0`,
`scikit-learn>=1.7,<1.8`, `transformers>=4.57,<6`) live in the packages themselves, so
pip enforces them without a separate environment file.

**Starting from nothing?** The bootstrap also clones the four repositories:

```bash
mkdir brainscore-umi && cd brainscore-umi
base=https://raw.githubusercontent.com/KartikP/brainscore-unified/unified-model-interface-v2/install
curl -fsSLO "$base/setup.sh" && curl -fsSLO "$base/environment-unified.yml"
bash setup.sh
conda activate brainscore-unified
```

`install/environment-unified.yml` exists for that bootstrap, which places it beside the
repositories before use. Do not point conda at it in place — conda resolves its relative
`-e ./core` entries from the file's own directory, so it fails with
`ERROR: ./core is not a valid editable requirement`. The two-step above avoids the file
entirely and is preferred when a checkout already exists.

Requires Python 3.11 and `conda` (miniforge or miniconda). See
[install/README.md](install/README.md) for troubleshooting.

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
| Embodied action | GridGame `action_fn` | demo-only | Floor, ceiling, deterministic boards, and registration are tested; the game is an interface demo. |
| Closed-weight API behavior | OpenRouter / OpenAI-compatible | structurally-tested | Provider registration, lazy model construction, parsing, image payloads, and cache behavior are mocked locally. |
| Cursor traces | `Nodekit-fitts-pointing` | demo-only | Prototype. Pointer actions replay in headless Chromium on a nodekit site; trace reader, metrics, harness timing and click adapter are tested. No human traces yet, so the ceiling is a placeholder. |

## Cursor traces (nodekit, prototype)

Models play the same browser task people do. A [nodekit](https://github.com/intelligence-observatory/nodekit)
site runs in headless Chromium; the model sees a screenshot of the task board and answers
with pointer samples, which are replayed as real mouse events. nodekit's own runtime logs
them, so model and human traces share one format and one coordinate system (Board
coordinates: origin at the centre, y up, 1024 x 1024 px).

```bash
pip install -e "./unified[browser]"
playwright install chromium
```

```python
from brainscore import load_benchmark, load_model
score = load_benchmark('Nodekit-fitts-pointing')(load_model('nodekit-straight-reach'))
score.attrs['fitts_r'], score.attrs['per_trial'][0], score.attrs['trace']['events'][:3]
```

- **Action format:** `process(EnvironmentStep)` returns an `EnvironmentResponse` whose
  `action` is an `(n, 4)` array of `(dt_ms, x, y, kind)` rows (kind 0 = move, 1 = button
  down, 2 = button up). Build one with `brainscore.harnesses.nodekit_browser.pointer_action`
  or `click`.
- **Timing:** the page clock is frozen between actions, so model thinking time is not
  recorded. Page time advances only by the `dt_ms` the model emits; page times are
  identical from run to run.
- **Models:** `nodekit-random-pointer` (null floor), `nodekit-straight-reach` (reference
  mover that reads the card layout), and
  `brainscore.model_helpers.pointer_policy.build_click_policy` for vision-language models
  (API or local weights).
- **Score:** Pearson r between movement time and Fitts' index of difficulty over completed
  trials. Reference mover: 0.61 (36/36 trials); random null: 0 (0/36).
- **Rebuilding the site** needs nodekit, which requires Python 3.12, in its own
  environment: `pip install "nodekit @ git+https://github.com/intelligence-observatory/nodekit@3a13ac4"`
  then `python brainscore/benchmarks/nodekit_fitts/build_site.py`. Scoring does not need it.
- **Tests:** `pytest tests/test_cursor_traces.py tests/test_nodekit_fitts.py tests/test_nodekit_browser.py`
  (the browser tests skip when Playwright or Chromium is missing).
