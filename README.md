# brainscore (unified model interface)

The unified Brain-Score package: register a model once and score it across every
compatible benchmark domain (vision, language, audio, video, multimodal,
perturbation, embodied) through a single `process()`-based interface.

## Layout

- `brainscore/model_helpers/` — wrappers (model harnesses): image, text,
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

`brainscore` is installed alongside the three domain repositories, which provide the
model-extraction and scoring stack. All four are editable installs from one workspace.

**No checkout yet — use the bootstrap.** It clones all four repositories at the right
branch and builds the pinned environment:

```bash
mkdir brainscore-umi && cd brainscore-umi
base=https://raw.githubusercontent.com/KartikP/brainscore-unified/unified-model-interface-v2/install
curl -fsSLO "$base/setup.sh" && curl -fsSLO "$base/environment-unified.yml"
bash setup.sh
conda activate brainscore-unified
```

**Already have the four repositories side by side?** Create the environment **from the
workspace root** — the directory containing `core/`, `vision/`, `language/`, `unified/`:

```bash
cd <workspace-root>          # NOT from inside unified/
conda env create -n brainscore-unified -f unified/install/environment-unified.yml
conda activate brainscore-unified
```

The environment file's `-e ./core` entries are relative paths, so running it from
anywhere but the workspace root fails with four "path does not exist" errors. See
[install/README.md](install/README.md) for troubleshooting.

Requires Python 3.11 and `conda` (miniforge or miniconda). For notebook dependencies in
an environment you already have, `pip install -e "unified[notebooks]"` from the
workspace root.

## Usage

A model is a `Subject`; the class you construct is `BrainScoreModel`
(see [Concepts](docs/concepts.md#subject)). The everyday surface:

```python
from brainscore import load_model, load_benchmark

model = load_model("clip-vit-b-32")
score = load_benchmark("MajajHong2015public.IT-pls-unified")(model)
```

New to Brain-Score? Read [Concepts](docs/concepts.md) first — it defines the vocabulary
(subject, assembly, neuroid, `region_layer_map`, raw vs. ceiled) that everything else
assumes.

Start with the shipped [getting-started guide](docs/getting_started.md), then
use [EXTENDING.md](EXTENDING.md), [templates](templates/), the
[notebook manifest](notebooks/README.md), and the
[UMI API cookbook](docs/umi_api_reference.md).

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
